from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from tqdm import tqdm
from scripts.resnet18.dataloaders import cifar10_dataloaders
from scripts.resnet18_cifar10 import evaluate, evaluate_adv
from scripts.resnet18.models import make_resnet18k
from utils.lc_utils import get_intersections_for_hulls, get_ortho_hull_around_samples
from models.model_utils import add_hooks_preact_resnet18
from utils.metrics import compute_jacobian_rank, compute_jacobian_alignment


def compute_metrics_over_checkpoints_resnet(checkpoint_dir, config, test_loader, train_loader, device,
                                            adv_eps, adv_alpha, adv_itrs, adv_min, adv_max,
                                            n_frame=25, r_frame=0.005, seed=0, lc_batch_size=32,
                                            max_batches=2, top_k=1, save_path=None):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pth"),
                         key=lambda p: int(p.stem.split('_')[1]))

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return None

    def build_hulls(loader):
        all_images = []
        for i, (ims, _) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            all_images.append(ims)
        all_images = torch.cat(all_images, dim=0)
        hulls = get_ortho_hull_around_samples(
            all_images.to(device),
            r=r_frame,
            n=n_frame,
            seed=seed,
        ).cpu()
        return hulls

    print("Building train hulls...")
    train_hulls = build_hulls(train_loader)
    print("Building test hulls...")
    test_hulls = build_hulls(test_loader)
    print(f"Train hulls: {train_hulls.shape}, Test hulls: {test_hulls.shape}")

    loss_fn = nn.CrossEntropyLoss()
    n_blocks = 8
    steps = []
    all_train_accs = []
    all_test_accs = []
    all_test_accs_std = []
    all_adv_accs = []
    all_adv_accs_std = []
    all_train_lc = []
    all_train_lc_std = []
    all_test_lc = []
    all_test_lc_std = []
    all_residual_maxes = []
    all_residual_maxes_std = []
    all_jac_ranks = []
    all_jac_alignments = []

    for ckpt_path in tqdm(checkpoints, desc="Checkpoints"):
        step = int(ckpt_path.stem.split('_')[1])

        model = make_resnet18k(num_classes=config.num_class, bn=config.use_bn).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        train_acc, _ = evaluate(model, train_loader, loss_fn, max_batches=max_batches)
        test_acc, _ = evaluate(model, test_loader, loss_fn, max_batches=max_batches)
        adv_acc = evaluate_adv(model, test_loader, adv_eps, adv_alpha, adv_itrs,
                               adv_min, adv_max, device, max_batches=max_batches)

        block_residual_maxes = [[] for _ in range(n_blocks)]
        shortcut_buffer = {}

        def make_hook(block_idx):
            def hook(module, input, output):
                shortcut = module.shortcut(F.relu(module.bn1(input[0]))) if hasattr(module, 'shortcut') else input[0]
                topk = shortcut.abs().mean(dim=[-2, -1]).topk(top_k, dim=1).values.sum(dim=1)
                shortcut_buffer[block_idx] = topk.mean().item()
            return hook

        handles = []
        block_idx = 0
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in layer:
                handles.append(block.register_forward_hook(make_hook(block_idx)))
                block_idx += 1

        for i, (ims, _) in enumerate(test_loader):
            if i >= max_batches:
                break
            ims = ims.to(device)
            with torch.no_grad():
                model(ims)
            for b in range(n_blocks):
                block_residual_maxes[b].append(shortcut_buffer[b])

        for h in handles:
            h.remove()

        model_hooked, layer_names, activation_buffer = add_hooks_preact_resnet18(model, config.use_bn, verbose=False)
        n_inters_train, _ = get_intersections_for_hulls(
            hulls=train_hulls,
            model=model_hooked,
            layer_names=layer_names,
            activation_buffer=activation_buffer,
            batch_size=lc_batch_size,
            verbose=False,
        )
        n_inters_test, _ = get_intersections_for_hulls(
            hulls=test_hulls,
            model=model_hooked,
            layer_names=layer_names,
            activation_buffer=activation_buffer,
            batch_size=lc_batch_size,
            verbose=False,
        )

        train_lc_per_sample = n_inters_train.float().sum(dim=1)
        test_lc_per_sample = n_inters_test.float().sum(dim=1)

        jac_rank = compute_jacobian_rank(model, test_loader, device, max_batches)
        jac_alignment = compute_jacobian_alignment(model, test_loader, device, max_batches)

        steps.append(step)
        all_train_accs.append(train_acc.item() * 100.0)
        all_test_accs.append(test_acc.item() * 100.0)
        all_test_accs_std.append(0.0)
        all_adv_accs.append(adv_acc.item() * 100.0)
        all_adv_accs_std.append(0.0)
        all_train_lc.append(train_lc_per_sample.mean().item())
        all_train_lc_std.append(train_lc_per_sample.std().item())
        all_test_lc.append(test_lc_per_sample.mean().item())
        all_test_lc_std.append(test_lc_per_sample.std().item())
        all_residual_maxes.append([np.mean(block_residual_maxes[b]) for b in range(n_blocks)])
        all_residual_maxes_std.append([np.std(block_residual_maxes[b]) for b in range(n_blocks)])
        all_jac_ranks.append(jac_rank)
        all_jac_alignments.append(jac_alignment)

    metrics = {
        'steps': np.array(steps),
        'train_accs': np.array(all_train_accs),
        'test_accs': np.array(all_test_accs),
        'test_accs_std': np.array(all_test_accs_std),
        'adv_accs': np.array(all_adv_accs),
        'adv_accs_std': np.array(all_adv_accs_std),
        'train_lc': np.array(all_train_lc),
        'train_lc_std': np.array(all_train_lc_std),
        'test_lc': np.array(all_test_lc),
        'test_lc_std': np.array(all_test_lc_std),
        'residual_maxes': np.array(all_residual_maxes),
        'residual_maxes_std': np.array(all_residual_maxes_std),
        'jac_ranks': np.array(all_jac_ranks),
        'jac_alignments': np.array(all_jac_alignments),
        'n_blocks': n_blocks,
        'adv_eps': adv_eps,
        'top_k': top_k,
    }

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    np.savez(save_path / f"{checkpoint_dir.name}-r_frame:{r_frame}", **metrics)
    print(f"Metrics saved to {save_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="utils/metrics")
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--classes', type=int, nargs='+', default=None)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--max_batches', type=int, default=16)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')

    # adv robustness params
    parser.add_argument('--adv_eps', type=float, default=0.2)
    parser.add_argument('--adv_alpha', type=float, default=4/255)
    parser.add_argument('--adv_itrs', type=int, default=10)
    parser.add_argument('--adv_min', type=float, default=-2.4291)
    parser.add_argument('--adv_max', type=float, default=2.7537)

    # local complexity parameters
    parser.add_argument('--r_frame', type=float, default=0.001)
    parser.add_argument('--n_frame', type=int, default=25)
    parser.add_argument('--lc_batch_size', type=int, default=8)
    args = parser.parse_args()

    if args.classes is not None:
        args.num_class = len(args.classes)

    device = torch.device(args.device)
    train_loader, test_loader = cifar10_dataloaders(args)

    compute_metrics_over_checkpoints_resnet(
        checkpoint_dir=args.checkpoint_dir,
        config=args,
        test_loader=test_loader,
        train_loader=train_loader,
        device=device,
        adv_eps=args.adv_eps,
        adv_alpha=args.adv_alpha,
        adv_itrs=args.adv_itrs,
        adv_min=args.adv_min,
        adv_max=args.adv_max,
        r_frame=args.r_frame,
        n_frame=args.n_frame,
        lc_batch_size=args.lc_batch_size,
        max_batches=args.max_batches,
        top_k=args.top_k,
        save_path=args.save_path,
    )