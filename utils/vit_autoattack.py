import torch
import torch.nn as nn
import argparse
import numpy as np
import torchattacks
from tqdm import tqdm
from pathlib import Path

from scripts.vit_imagenette import evaluate, get_loaders, make_model


def evaluate_attacks(model, loader, device, eps, n_steps, n_queries, max_batches):
    model.eval()

    apgd_ce = torchattacks.APGD(model, norm='Linf', eps=eps, steps=n_steps, loss='ce')
    apgd_dlr = torchattacks.APGD(model, norm='Linf', eps=eps, steps=n_steps, loss='dlr')
    square = torchattacks.Square(model, norm='Linf', eps=eps, n_queries=n_queries, n_restarts=1)

    total = 0
    n_clean = 0
    n_rob_ce = 0
    n_rob_dlr = 0
    n_rob_sq = 0
    n_rob_union = 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        total += x.size(0)

        with torch.no_grad():
            correct_mask = (model(x).argmax(1) == y)
        n_clean += correct_mask.sum().item()

        x_c = x[correct_mask]
        y_c = y[correct_mask]
        if x_c.size(0) == 0:
            continue

        with torch.no_grad():
            found_ce = model(apgd_ce(x_c, y_c)).argmax(1) != y_c
            found_dlr = model(apgd_dlr(x_c, y_c)).argmax(1) != y_c
            found_sq = model(square(x_c, y_c)).argmax(1) != y_c
            found_any = found_ce | found_dlr | found_sq

        n_rob_ce += (~found_ce).sum().item()
        n_rob_dlr += (~found_dlr).sum().item()
        n_rob_sq += (~found_sq).sum().item()
        n_rob_union += (~found_any).sum().item()

    denom = max(total, 1)
    return {
        'clean_acc': n_clean / total,
        'apgd_ce_rob': n_rob_ce / denom,
        'apgd_dlr_rob': n_rob_dlr / denom,
        'square_rob': n_rob_sq / denom,
        'union_rob': n_rob_union / denom,
    }


def compute_adv_robustness_over_checkpoints(
        checkpoint_dir, config, test_loader, device,
        eps, n_steps, n_queries, max_batches, save_path):

    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_*.pth"),
        key=lambda p: int(p.stem.split('_')[1])
    )

    steps = []
    clean_accs = []
    apgd_ce_robs = []
    apgd_dlr_robs = []
    square_robs = []
    union_robs = []

    for ckpt_path in tqdm(checkpoints, desc="Checkpoints"):
        step = int(ckpt_path.stem.split('_')[1])

        model = make_model(config).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        m = evaluate_attacks(model, test_loader, device,
                              eps, n_steps, n_queries, max_batches)

        steps.append(step)
        clean_accs.append(m['clean_acc'] * 100.0)
        apgd_ce_robs.append(m['apgd_ce_rob'] * 100.0)
        apgd_dlr_robs.append(m['apgd_dlr_rob'] * 100.0)
        square_robs.append(m['square_rob'] * 100.0)
        union_robs.append(m['union_rob'] * 100.0)

        tqdm.write(
            f"step {step:>8d}  "
            f"clean={m['clean_acc']:.4f}  "
            f"APGD-CE={m['apgd_ce_rob']:.4f}  "
            f"APGD-DLR={m['apgd_dlr_rob']:.4f}  "
            f"Square={m['square_rob']:.4f}  "
            f"union={m['union_rob']:.4f}"
        )

        del model
        torch.cuda.empty_cache()

    metrics = {
        'steps': np.array(steps),
        'clean_accs': np.array(clean_accs),
        'apgd_ce_robs': np.array(apgd_ce_robs),
        'apgd_dlr_robs': np.array(apgd_dlr_robs),
        'square_robs': np.array(square_robs),
        'union_robs': np.array(union_robs),
        'eps': eps,
        'n_steps': n_steps,
        'n_queries': n_queries,
    }

    out = Path(save_path) / f"{checkpoint_dir.name}-eps:{eps:.4f}"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **metrics)
    print(f"Saved to {out}.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="utils/metrics")
    parser.add_argument("--model_name", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--dataset", type=str, default="imagenette")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--max_batches", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    # attack params
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--n_queries", type=int, default=200)
    args = parser.parse_args()

    args.num_class = len(args.classes) if args.classes is not None else 10

    device = torch.device(args.device)
    _, test_loader, _, _ = get_loaders(args, root="./data")

    compute_adv_robustness_over_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        config=args,
        test_loader=test_loader,
        device=device,
        eps=args.eps,
        n_steps=args.n_steps,
        n_queries=args.n_queries,
        max_batches=args.max_batches,
        save_path=args.save_path,
    )