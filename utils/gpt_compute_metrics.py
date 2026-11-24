import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from scripts.gpt_shakespeare_text import EmbeddingPGD, GPT, get_loaders, evaluate
from utils.lc_utils import get_intersections_for_hulls, get_ortho_hull_around_samples
from models.model_utils import flatten_model
from utils.metrics import compute_jacobian_rank_gpt, compute_jacobian_alignment_gpt


def sequence_attack_loss(logits, targets, loss):
    """Return one attack objective per sequence; larger values are worse."""
    if loss == "ce":
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        return token_losses.mean(dim=1)

    if loss == "dlr":
        sorted_logits, sorted_indices = logits.sort(dim=-1)
        target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        target_is_top = sorted_indices[..., -1].eq(targets)
        best_other = torch.where(
            target_is_top,
            sorted_logits[..., -2],
            sorted_logits[..., -1],
        )
        denominator = sorted_logits[..., -1] - sorted_logits[..., -3] + 1e-12
        return (-(target_logits - best_other) / denominator).mean(dim=1)

    raise ValueError(f"Unsupported attack loss: {loss}")


class EmbeddingAPGD:
    """Adaptive projected-gradient attack on token embeddings."""

    def __init__(self, model, eps, steps, device, loss="ce", n_restarts=1, seed=0, rho=0.75):
        self.model = model
        self.eps = eps
        self.steps = steps
        self.device = device
        self.loss = loss
        self.n_restarts = n_restarts
        self.seed = seed
        self.rho = rho

    def _loss(self, adv, pos_embeds, targets):
        logits, _ = self.model(inputs_embeds=adv + pos_embeds)
        return sequence_attack_loss(logits, targets, self.loss)

    def attack(self, idx, targets):
        _, sequence_length = idx.shape
        pos = torch.arange(sequence_length, device=self.device)
        with torch.no_grad():
            clean = self.model.tok_emb(idx)
            pos_embeds = self.model.pos_emb(pos)

        overall_best = clean.clone()
        overall_best_loss = torch.full(
            (clean.size(0),), -torch.inf, device=self.device
        )

        for restart in range(self.n_restarts):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed + restart)
            delta = torch.empty_like(clean).uniform_(
                -self.eps, self.eps, generator=generator
            )
            adv = clean + delta
            previous_adv = adv.clone()
            step_size = torch.full(
                (clean.size(0), 1, 1), 2.0 * self.eps, device=self.device
            )
            best_adv = adv.clone()
            best_loss = torch.full_like(overall_best_loss, -torch.inf)
            best_loss_at_check = best_loss.clone()
            loss_history = []
            check_interval = max(int(0.22 * self.steps), 1)

            for attack_step in range(self.steps):
                adv.requires_grad_(True)
                losses = self._loss(adv, pos_embeds, targets)
                grad = torch.autograd.grad(losses.sum(), adv)[0]

                with torch.no_grad():
                    improved = losses > best_loss
                    best_loss[improved] = losses[improved]
                    best_adv[improved] = adv[improved]
                    loss_history.append(losses.detach())

                    if (attack_step + 1) % check_interval == 0 and len(loss_history) > 1:
                        recent = torch.stack(loss_history[-check_interval:])
                        improvement_rate = (recent[1:] > recent[:-1]).float().mean(dim=0)
                        stalled = best_loss <= best_loss_at_check + 1e-12
                        oscillating = improvement_rate <= self.rho
                        reduce_step = stalled | oscillating
                        step_size[reduce_step] *= 0.5
                        best_loss_at_check = best_loss.clone()

                    candidate = adv + step_size * grad.sign()
                    candidate = torch.maximum(
                        torch.minimum(candidate, clean + self.eps),
                        clean - self.eps,
                    )
                    if attack_step > 0:
                        candidate = adv + 0.75 * (candidate - adv) + 0.25 * (
                            adv - previous_adv
                        )
                        candidate = torch.maximum(
                            torch.minimum(candidate, clean + self.eps),
                            clean - self.eps,
                        )
                    previous_adv = adv.detach()
                    adv = candidate.detach()

            with torch.no_grad():
                final_loss = self._loss(adv, pos_embeds, targets)
                improved = final_loss > best_loss
                best_loss[improved] = final_loss[improved]
                best_adv[improved] = adv[improved]

                improved = best_loss > overall_best_loss
                overall_best_loss[improved] = best_loss[improved]
                overall_best[improved] = best_adv[improved]

        return overall_best + pos_embeds


class EmbeddingSquare:
    """Square-style black-box attack over the sequence-by-embedding matrix."""

    def __init__(self, model, eps, n_queries, device, seed=0, p_init=0.8):
        self.model = model
        self.eps = eps
        self.n_queries = n_queries
        self.device = device
        self.seed = seed
        self.p_init = p_init

    def _loss(self, adv, pos_embeds, targets):
        logits, _ = self.model(inputs_embeds=adv + pos_embeds)
        return sequence_attack_loss(logits, targets, "ce")

    def _p(self, query):
        progress = query / max(self.n_queries, 1)
        if progress <= 0.1:
            return self.p_init
        if progress <= 0.3:
            return self.p_init / 2
        if progress <= 0.5:
            return self.p_init / 4
        if progress <= 0.7:
            return self.p_init / 8
        if progress <= 0.9:
            return self.p_init / 16
        return self.p_init / 32

    def attack(self, idx, targets):
        batch_size, sequence_length = idx.shape
        pos = torch.arange(sequence_length, device=self.device)
        with torch.no_grad():
            clean = self.model.tok_emb(idx)
            pos_embeds = self.model.pos_emb(pos)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        signs = torch.randint(
            0, 2, clean.shape, device=self.device, generator=generator
        ).mul_(2).sub_(1)
        adv = clean + self.eps * signs

        with torch.no_grad():
            best_loss = self._loss(adv, pos_embeds, targets)

            for query in range(self.n_queries):
                candidate = adv.clone()
                side = max(
                    int(round((self._p(query) * sequence_length * clean.size(2)) ** 0.5)),
                    1,
                )
                token_side = min(side, sequence_length)
                embedding_side = min(side, clean.size(2))
                token_starts = torch.randint(
                    0,
                    sequence_length - token_side + 1,
                    (batch_size,),
                    device=self.device,
                    generator=generator,
                )
                embedding_starts = torch.randint(
                    0,
                    clean.size(2) - embedding_side + 1,
                    (batch_size,),
                    device=self.device,
                    generator=generator,
                )
                patch_signs = torch.randint(
                    0,
                    2,
                    (batch_size, token_side, embedding_side),
                    device=self.device,
                    generator=generator,
                ).mul_(2).sub_(1)

                for sample in range(batch_size):
                    token_start = token_starts[sample].item()
                    embedding_start = embedding_starts[sample].item()
                    token_slice = slice(token_start, token_start + token_side)
                    embedding_slice = slice(
                        embedding_start, embedding_start + embedding_side
                    )
                    candidate[sample, token_slice, embedding_slice] = (
                        clean[sample, token_slice, embedding_slice]
                        + self.eps * patch_signs[sample]
                    )

                candidate_loss = self._loss(candidate, pos_embeds, targets)
                improved = candidate_loss > best_loss
                adv[improved] = candidate[improved]
                best_loss[improved] = candidate_loss[improved]

        return adv + pos_embeds


def evaluate_adv_attacks(
    model,
    loader,
    eps,
    alpha,
    steps,
    n_queries,
    n_restarts,
    seed,
    device,
    max_batches=None,
    n_positions=8,
    max_sequences=64,
):
    from utils.language_attacks import evaluate_loader

    masks = evaluate_loader(model, loader, eps, device, is_gpt=True,
                            attack_steps=steps, restarts=n_restarts, seed=seed,
                            n_positions=n_positions, max_sequences=max_sequences,
                            max_batches=max_batches)
    return {name: float(values.mean()) for name, values in masks.items()}


def build_hulls_for_gpt(model, loader, device, n_frame, r_frame, seed, max_lc_seq_len, max_batches=None):
    model.eval()
    all_hulls = []
    rng = torch.Generator()
    rng.manual_seed(seed)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            B = x.shape[0]
            for b in range(B):
                t = torch.randint(1, max_lc_seq_len + 1, (1,), generator=rng).item()
                x_t = x[b:b+1, :t].to(device)
                embeds = model.tok_emb(x_t).view(1, -1)
                hull = get_ortho_hull_around_samples(
                    embeds, r=r_frame, n=n_frame, seed=seed
                )
                pad_size = max_lc_seq_len * model.config.n_embd - hull.shape[-1]
                hull_padded = F.pad(hull, (0, pad_size))
                all_hulls.append(hull_padded.cpu())
    return torch.cat(all_hulls, dim=0)


class GPTEmbedWrapper(nn.Module):
    def __init__(self, model, seq_len, n_embd):
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.n_embd = n_embd

    def forward(self, x):
        embeds = x.view(x.shape[0], self.seq_len, self.n_embd)
        pos = torch.arange(self.seq_len, device=x.device)
        inputs = self.model.drop(embeds + self.model.pos_emb(pos))
        for block in self.model.blocks:
            inputs = block(inputs)
        inputs = self.model.ln_f(inputs)
        return self.model.head(inputs)


def add_hooks_gpt(model):
    names, modules = flatten_model(model)
    layer_ids = [i for i, mod in enumerate(modules) if isinstance(mod, nn.GELU)]
    activation = {}

    def get_activation(name):
        def hook(module, input, output):
            activation[name] = input[0].detach().view(input[0].shape[0], -1)
        return hook

    for idx in layer_ids:
        modules[idx].register_forward_hook(get_activation(names[idx]))

    layer_names = [names[i] for i in layer_ids]
    return model, layer_names, activation


def compute_avg_max_activation(model, loader, device, max_batches=None):
    model.eval()
    total_max = 0.0
    n = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            B, T = x.shape
            pos = torch.arange(T, device=device)
            hidden = model.drop(model.tok_emb(x) + model.pos_emb(pos))
            for block in model.blocks:
                hidden = block(hidden)
                token_maxes = hidden.abs().max(dim=-1).values
                total_max += token_maxes.mean().item()
                n += 1
    return total_max / n


def compute_residual_maxes(model, loader, device, top_k=1, max_batches=None):
    model.eval()
    n_layers = len(model.blocks)
    layer_maxes = [[] for _ in range(n_layers)]
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            B, T = x.shape
            pos = torch.arange(T, device=device)
            h = model.drop(model.tok_emb(x) + model.pos_emb(pos))
            for layer_idx, block in enumerate(model.blocks):
                h = h + block.attn(block.ln1(h))
                last_tok = h[:, -1, :]
                topk_val = last_tok.abs().topk(top_k, dim=-1).values.sum(dim=-1)
                layer_maxes[layer_idx].append(topk_val.mean().item())
                h = h + block.mlp(block.ln2(h))
    return [np.mean(layer_maxes[l]) for l in range(n_layers)]


def compute_metrics_over_checkpoints(
    checkpoint_dir,
    data_path,
    block_size,
    batch_size,
    n_embd,
    n_head,
    n_layer,
    dropout,
    bias,
    device,
    adv_eps,
    adv_alpha,
    adv_steps,
    adv_n_queries,
    adv_restarts,
    adv_seed,
    n_frame,
    r_frame,
    seed,
    lc_seq_len,
    lc_batch_size,
    max_batches,
    top_k,
    save_path,
    adv_positions=8,
    adv_sequences=64,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pth"),
                         key=lambda p: int(p.stem.split('_')[1]))

    train_loader, test_loader, vocab_size = get_loaders(data_path, block_size, batch_size)

    class GPTConfig:
        pass
    gpt_config = GPTConfig()
    gpt_config.vocab_size = vocab_size
    gpt_config.block_size = block_size
    gpt_config.n_embd = n_embd
    gpt_config.n_head = n_head
    gpt_config.n_layer = n_layer
    gpt_config.dropout = dropout
    gpt_config.bias = bias

    dummy_model = GPT(gpt_config).to(device)
    print("Building train hulls...")
    train_hulls = build_hulls_for_gpt(dummy_model, train_loader, device, n_frame, r_frame, seed, lc_seq_len, max_batches=max_batches)
    print("Building test hulls...")
    test_hulls = build_hulls_for_gpt(dummy_model, test_loader, device, n_frame, r_frame, seed, lc_seq_len, max_batches=max_batches)
    del dummy_model
    print(f"Train hulls: {train_hulls.shape}, Test hulls: {test_hulls.shape}")

    steps = []
    all_train_accs, all_test_accs, all_train_losses, all_test_losses = [], [], [], []
    all_adv_accs = {
        name: [] for name in ["apgd_ce", "apgd_dlr", "fab", "union"]
    }
    all_avg_max_acts = []
    all_residual_maxes = []
    all_train_lc, all_train_lc_std = [], []
    all_test_lc, all_test_lc_std = [], []
    all_jac_ranks = []
    all_jac_alignments = []

    for ckpt_path in tqdm(checkpoints, desc="Checkpoints"):
        step = int(ckpt_path.stem.split('_')[1])

        model = GPT(gpt_config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        train_acc, train_loss = evaluate(model, train_loader, device, max_batches)
        test_acc, test_loss = evaluate(model, test_loader, device, max_batches)
        adv_acc = evaluate_adv_attacks(
            model,
            test_loader,
            adv_eps,
            adv_alpha,
            adv_steps,
            adv_n_queries,
            adv_restarts,
            adv_seed,
            device,
            max_batches,
            n_positions=adv_positions,
            max_sequences=adv_sequences,
        )
        test_acc = adv_acc["clean"]

        avg_max_act = compute_avg_max_activation(model, test_loader, device, max_batches)
        residual_maxes = compute_residual_maxes(model, test_loader, device, top_k, max_batches)

        wrapped = GPTEmbedWrapper(model, lc_seq_len, n_embd).to(device)
        wrapped_hooked, layer_names, activation_buffer = add_hooks_gpt(wrapped)

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

        jac_rank = compute_jacobian_rank_gpt(model, test_loader, device, max_batches)
        jac_alignment = compute_jacobian_alignment_gpt(model, test_loader, device, max_batches)

        steps.append(step)
        all_train_accs.append(train_acc * 100.0)
        all_test_accs.append(test_acc * 100.0)
        for name in all_adv_accs:
            all_adv_accs[name].append(adv_acc[name] * 100.0)
        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)
        all_avg_max_acts.append(avg_max_act)
        all_residual_maxes.append(residual_maxes)
        all_train_lc.append(train_lc_per_sample.mean().item())
        all_train_lc_std.append(train_lc_per_sample.std().item())
        all_test_lc.append(test_lc_per_sample.mean().item())
        all_test_lc_std.append(test_lc_per_sample.std().item())
        all_jac_ranks.append(jac_rank)
        all_jac_alignments.append(jac_alignment)  

        del model, wrapped, wrapped_hooked

    metrics = {
        'steps': np.array(steps),
        'train_accs': np.array(all_train_accs),
        'test_accs': np.array(all_test_accs),
        **{f'adv_accs_{name}': np.array(values) for name, values in all_adv_accs.items()},
        'train_losses': np.array(all_train_losses),
        'test_losses': np.array(all_test_losses),
        'avg_max_acts': np.array(all_avg_max_acts),
        'residual_maxes': np.array(all_residual_maxes),
        'train_lc': np.array(all_train_lc),
        'train_lc_std': np.array(all_train_lc_std),
        'test_lc': np.array(all_test_lc),
        'test_lc_std': np.array(all_test_lc_std),
        'jac_ranks': np.array(all_jac_ranks),
        'jac_alignments': np.array(all_jac_alignments),
        'n_layers': n_layer,
        'adv_eps': adv_eps,
        'adv_n_queries': adv_n_queries,
        'adv_restarts': adv_restarts,
        'adv_steps': adv_steps,
        'adv_positions': adv_positions,
        'adv_sequences': adv_sequences,
        'attack_protocol': 'shared-position-apgd-ce-apgd-dlr-binary-fab',
        'fab_conditional': True,
        'top_k': top_k,
    }

    save_path = Path(save_path) / f"{checkpoint_dir.name}-r_frame:{r_frame}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **metrics)
    print(f"Metrics saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='utils/metrics')
    parser.add_argument('--data_path', type=str, default='data/shakespeare.txt')
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_batches', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--adv_eps', type=float, default=0.01)
    parser.add_argument('--adv_alpha', type=float, default=0.01)
    parser.add_argument('--adv_steps', type=int, default=10)
    parser.add_argument('--adv_n_queries', type=int, default=200)
    parser.add_argument('--adv_restarts', type=int, default=1)
    parser.add_argument('--adv_positions', type=int, default=8)
    parser.add_argument('--adv_sequences', type=int, default=64)
    parser.add_argument('--adv_seed', type=int, default=0)
    parser.add_argument('--n_frame', type=int, default=25)
    parser.add_argument('--r_frame', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lc_seq_len', type=int, default=64)
    parser.add_argument('--lc_batch_size', type=int, default=32)
    args = parser.parse_args()

    compute_metrics_over_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        block_size=args.block_size,
        batch_size=args.batch_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        bias=args.bias,
        device=args.device,
        adv_eps=args.adv_eps,
        adv_alpha=args.adv_alpha,
        adv_steps=args.adv_steps,
        adv_n_queries=args.adv_n_queries,
        adv_restarts=args.adv_restarts,
        adv_seed=args.adv_seed,
        n_frame=args.n_frame,
        r_frame=args.r_frame,
        seed=args.seed,
        lc_seq_len=args.lc_seq_len,
        lc_batch_size=args.lc_batch_size,
        max_batches=args.max_batches,
        top_k=args.top_k,
        save_path=args.save_path,
        adv_positions=args.adv_positions,
        adv_sequences=args.adv_sequences,
    )
