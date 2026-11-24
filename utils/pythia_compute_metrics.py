import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.lc_utils import get_intersections_for_hulls, get_ortho_hull_around_samples
from utils.data_utils import get_pile_loaders
from models.model_utils import flatten_model, add_hooks_pythia
from transformers.activations import GELUActivation
from utils.checkpoint_utils import load_pythia_checkpoint

SEQ_LEN = 64
CACHE_DIR = None

PYTHIA_CHECKPOINTS = [
    f"step{s}" for s in [
        16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000
    ]
]

PYTHIA_MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    # "EleutherAI/pythia-12b",
]

PYTHIA_SIZES = [
    70,
    160,
    410,
    1400,
    2800,
    6900,
    # 12000
]

def compute_next_token_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            preds = logits.argmax(dim=-1)
            mask = attention_mask[:, 1:].bool()
            correct += (preds == labels)[mask].sum().item()
            total += mask.sum().item()
    return correct / total

def compute_adv_acc(model, attacker, loader, device):
    model.eval()
    correct = total = 0
    for input_ids, attention_mask in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        adv_embeds = attacker.attack(input_ids, attention_mask)
        with torch.no_grad():
            outputs = model(inputs_embeds=adv_embeds, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        preds = logits.argmax(dim=-1)
        mask = attention_mask[:, 1:].bool()
        correct += (preds == labels)[mask].sum().item()
        total += mask.sum().item()
    return correct / total

class EmbeddingPGD:
    def __init__(self, model, eps, alpha, steps, device):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device
        self.embed = model.gpt_neox.embed_in

    def attack(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            clean_embeds = self.embed(input_ids)
        adv = clean_embeds.clone().detach()
        adv = adv + torch.zeros_like(adv).uniform_(-self.eps, self.eps)
        adv = adv.detach()
        for _ in range(self.steps):
            adv.requires_grad_(True)
            outputs = self.model(inputs_embeds=adv, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + self.alpha * grad.sign()
            adv = torch.clamp(adv, clean_embeds - self.eps, clean_embeds + self.eps)
            adv = adv.detach()
        return adv

def compute_avg_max_activation(model, loader, device):
    model.eval()
    total_max = 0.0
    n = 0
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            for hidden in outputs.hidden_states[1:]:
                token_maxes = hidden.abs().max(dim=-1).values
                total_max += token_maxes.mean().item()
                n += 1
    return total_max / n

def compute_residual_maxes(model, loader, device, top_k=1):
    model.eval()
    n_layers = model.config.num_hidden_layers
    layer_maxes = [[] for _ in range(n_layers)]
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            for layer_idx, hidden in enumerate(outputs.hidden_states[1:]):
                last_tok = hidden[:, -1, :]
                topk_val = last_tok.abs().topk(top_k, dim=-1).values.sum(dim=-1)
                layer_maxes[layer_idx].append(topk_val.mean().item())
    return [np.mean(layer_maxes[l]) for l in range(n_layers)]

def compute_massive_magnitudes(model, loader, device, top_k=3, max_batches=None):
    model.eval()
    n_layers = model.config.num_hidden_layers
    total_magnitudes = torch.zeros(n_layers, dtype=torch.float64)
    total_tokens = 0

    with torch.no_grad():
        for i, (input_ids, attention_mask) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            token_mask = attention_mask.bool()
            token_mask[:, 0] = False

            if token_mask.sum().item() == 0:
                continue

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            for layer_idx, hidden in enumerate(outputs.hidden_states[1:]):
                topk_vals = hidden.abs().topk(top_k, dim=-1).values.sum(dim=-1)
                total_magnitudes[layer_idx] += topk_vals[token_mask].sum().double().cpu()

            total_tokens += token_mask.sum().item()

    return (total_magnitudes / total_tokens).numpy().tolist()

def build_hulls_for_pythia(model, loader, device, n_frame, r_frame, seed, max_lc_seq_len):
    model.eval()
    all_hulls = []
    rng = torch.Generator()
    rng.manual_seed(seed)
    embed = model.gpt_neox.embed_in
    hidden_dim = model.config.hidden_size
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            B = input_ids.shape[0]
            for b in range(B):
                t = torch.randint(1, max_lc_seq_len + 1, (1,), generator=rng).item()
                x_t = input_ids[b:b+1, :t].to(device)
                embeds = embed(x_t).view(1, -1)
                hull = get_ortho_hull_around_samples(embeds, r=r_frame, n=n_frame, seed=seed)
                pad_size = max_lc_seq_len * hidden_dim - hull.shape[-1]
                hull_padded = F.pad(hull, (0, pad_size))
                all_hulls.append(hull_padded.cpu())
    return torch.cat(all_hulls, dim=0)

class PythiaEmbedWrapper(nn.Module):
    def __init__(self, model, seq_len, hidden_dim):
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embeds = x.view(x.shape[0], self.seq_len, self.hidden_dim)
        return self.model(inputs_embeds=embeds)

def compute_metrics(model, tokenizer, device, max_batches, batch_size, adv_eps, adv_alpha, adv_steps, n_frame, r_frame, seed, lc_seq_len, lc_batch_size, top_k,
                    adv_restarts=1, adv_positions=8, adv_sequences=64):
    from utils.language_attacks import evaluate_loader

    if not adv_eps:
        raise ValueError("At least one adversarial budget is required")
    train_loader, test_loader = get_pile_loaders(tokenizer, max_batches, batch_size, SEQ_LEN)

    adv_accs = {}
    attack_accs = {}
    for eps in adv_eps:
        masks = evaluate_loader(model, train_loader, eps, device, attack_steps=adv_steps,
                                restarts=adv_restarts, n_positions=adv_positions,
                                max_sequences=adv_sequences, seed=seed)
        clean_acc = float(masks['clean'].mean())
        adv_accs[eps] = float(masks['union'].mean())
        attack_accs[eps] = {name: float(masks[name].mean())
                            for name in ('apgd_ce', 'apgd_dlr', 'fab')}

    avg_max_act = compute_avg_max_activation(model, train_loader, device)
    residual_maxes = compute_residual_maxes(model, train_loader, device, top_k)

    massive_magnitudes = compute_massive_magnitudes(model, train_loader, device, top_k, max_batches)

    train_hulls = build_hulls_for_pythia(model, train_loader, device, n_frame, r_frame, seed, lc_seq_len)
    test_hulls = build_hulls_for_pythia(model, test_loader, device, n_frame, r_frame, seed, lc_seq_len)

    hidden_dim = model.config.hidden_size
    wrapped = PythiaEmbedWrapper(model, lc_seq_len, hidden_dim)
    wrapped_hooked, layer_names, activation_buffer = add_hooks_pythia(wrapped)

    n_inters_train, _ = get_intersections_for_hulls(
        hulls=train_hulls,
        model=wrapped_hooked,
        layer_names=layer_names,
        activation_buffer=activation_buffer,
        batch_size=lc_batch_size,
        verbose=False,
    )
    n_inters_test, _ = get_intersections_for_hulls(
        hulls=test_hulls,
        model=wrapped_hooked,
        layer_names=layer_names,
        activation_buffer=activation_buffer,
        batch_size=lc_batch_size,
        verbose=False,
    )

    train_lc_per_sample = n_inters_train.float().sum(dim=1)
    test_lc_per_sample = n_inters_test.float().sum(dim=1)

    return {
        'clean_acc': clean_acc,
        'adv_accs': adv_accs,
        'attack_accs': attack_accs,
        'avg_max_act': avg_max_act,
        'residual_maxes': residual_maxes,
        'massive_magnitudes': massive_magnitudes,
        'train_lc': train_lc_per_sample.mean().item(),
        'train_lc_std': train_lc_per_sample.std().item(),
        'test_lc': test_lc_per_sample.mean().item(),
        'test_lc_std': test_lc_per_sample.std().item(),
    }

def run_over_checkpoints(args):
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    checkpoints = [f"step{s}" for s in args.steps] if args.steps else PYTHIA_CHECKPOINTS

    all_clean_accs = []
    all_adv_accs = {eps: [] for eps in args.adv_eps}
    all_attack_accs = {(eps, name): [] for eps in args.adv_eps
                       for name in ('apgd_ce', 'apgd_dlr', 'fab')}
    all_avg_max_acts = []
    all_residual_maxes = []
    all_massive_magnitudes = []
    all_train_lc = []
    all_train_lc_std = []
    all_test_lc = []
    all_test_lc_std = []
    steps = []

    for ckpt in tqdm(checkpoints, desc="Checkpoints"):
        model = load_pythia_checkpoint(args.model_name, ckpt, device, args.scratch_dir)

        result = compute_metrics(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_batches=args.max_batches,
            batch_size=args.batch_size,
            adv_eps=args.adv_eps,
            adv_alpha=args.adv_alpha,
            adv_steps=args.adv_steps,
            n_frame=args.n_frame,
            r_frame=args.r_frame,
            seed=args.seed,
            lc_seq_len=args.lc_seq_len,
            lc_batch_size=args.lc_batch_size,
            top_k=args.top_k,
            adv_restarts=args.adv_restarts,
            adv_positions=args.adv_positions,
            adv_sequences=args.adv_sequences,
        )

        step = int(ckpt.replace("step", ""))
        steps.append(step)
        all_clean_accs.append(result['clean_acc'])
        all_avg_max_acts.append(result['avg_max_act'])
        all_residual_maxes.append(result['residual_maxes'])
        all_massive_magnitudes.append(result['massive_magnitudes'])
        all_train_lc.append(result['train_lc'])
        all_train_lc_std.append(result['train_lc_std'])
        all_test_lc.append(result['test_lc'])
        all_test_lc_std.append(result['test_lc_std'])
        for eps in args.adv_eps:
            all_adv_accs[eps].append(result['adv_accs'][eps])
            for name in ('apgd_ce', 'apgd_dlr', 'fab'):
                all_attack_accs[eps, name].append(result['attack_accs'][eps][name])

        del model
        torch.cuda.empty_cache()

    metrics = {
        'steps': np.array(steps),
        'clean_accs': np.array(all_clean_accs),
        'avg_max_acts': np.array(all_avg_max_acts),
        'massive_magnitudes': np.array(all_massive_magnitudes),
        'residual_maxes': np.array(all_residual_maxes),
        'train_lc': np.array(all_train_lc),
        'train_lc_std': np.array(all_train_lc_std),
        'test_lc': np.array(all_test_lc),
        'test_lc_std': np.array(all_test_lc_std),
        'adv_eps': np.array(args.adv_eps),
        'top_k': args.top_k,
        'model_name': args.model_name,
        'adv_steps': args.adv_steps,
        'adv_restarts': args.adv_restarts,
        'adv_positions': args.adv_positions,
        'adv_sequences': args.adv_sequences,
        'attack_protocol': 'shared-position-apgd-ce-apgd-dlr-binary-fab',
        'fab_conditional': True,
        **{f'adv_accs_{eps:.4f}': np.array(all_adv_accs[eps]) for eps in args.adv_eps},
        **{f'adv_accs_{name}_{eps:.4f}': np.array(values)
           for (eps, name), values in all_attack_accs.items()},
    }

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    np.savez(save_path / f"{args.model_name.replace('/', '_')}-steps-r_frame:{args.r_frame}", **metrics)
    print(f"Metrics saved to {save_path}")

def run_over_model_sizes(args):
    device = torch.device(args.device)

    for model_name in tqdm(PYTHIA_MODELS, desc="Model sizes"):
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = load_pythia_checkpoint(model_name, "step143000", device, args.scratch_dir)

        result = compute_metrics(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_batches=args.max_batches,
            batch_size=args.batch_size,
            adv_eps=args.adv_eps,
            adv_alpha=args.adv_alpha,
            adv_steps=args.adv_steps,
            n_frame=args.n_frame,
            r_frame=args.r_frame,
            seed=args.seed,
            lc_seq_len=args.lc_seq_len,
            lc_batch_size=args.lc_batch_size,
            top_k=args.top_k,
            adv_restarts=args.adv_restarts,
            adv_positions=args.adv_positions,
            adv_sequences=args.adv_sequences,
        )

        model_size = PYTHIA_SIZES[PYTHIA_MODELS.index(model_name)]
        metrics = {
            'model_size': model_size,
            'model_name': model_name,
            'clean_acc': np.array([result['clean_acc']]),
            'avg_max_act': np.array([result['avg_max_act']]),
            'massive_magnitudes': np.array([result['massive_magnitudes']]),
            'residual_maxes': np.array([result['residual_maxes']]),
            'train_lc': np.array([result['train_lc']]),
            'train_lc_std': np.array([result['train_lc_std']]),
            'test_lc': np.array([result['test_lc']]),
            'test_lc_std': np.array([result['test_lc_std']]),
            'adv_eps': np.array(args.adv_eps),
            'top_k': args.top_k,
            'adv_steps': args.adv_steps,
            'adv_restarts': args.adv_restarts,
            'adv_positions': args.adv_positions,
            'adv_sequences': args.adv_sequences,
            'attack_protocol': 'shared-position-apgd-ce-apgd-dlr-binary-fab',
            'fab_conditional': True,
            **{f'adv_accs_{eps:.4f}': np.array([result['adv_accs'][eps]]) for eps in args.adv_eps},
            **{f'adv_accs_{name}_{eps:.4f}': np.array([result['attack_accs'][eps][name]])
               for eps in args.adv_eps for name in ('apgd_ce', 'apgd_dlr', 'fab')},
        }

        save_path = Path(args.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(save_path / f"{model_name.replace('/', '_')}-r_frame:{args.r_frame}", **metrics)
        print(f"Saved metrics for {model_name}")

        del model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['steps', 'models'], default='steps')
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-1.4b')
    parser.add_argument('--save_path', type=str, default='utils/metrics')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scratch_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_batches', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--adv_eps', type=float, nargs='+', default=[0.00025, 0.0005, 0.001, 0.0025, 0.005])
    parser.add_argument('--adv_alpha', type=float, default=0.001)
    parser.add_argument('--adv_steps', type=int, default=10)
    parser.add_argument('--adv_restarts', type=int, default=1)
    parser.add_argument('--adv_positions', type=int, default=8)
    parser.add_argument('--adv_sequences', type=int, default=64)
    parser.add_argument('--n_frame', type=int, default=25)
    parser.add_argument('--r_frame', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lc_seq_len', type=int, default=64)
    parser.add_argument('--lc_batch_size', type=int, default=2)
    parser.add_argument('--steps', type=int, nargs='+', default=None)
    args = parser.parse_args()

    if args.mode == 'models':
        run_over_model_sizes(args)
    else:
        run_over_checkpoints(args)
