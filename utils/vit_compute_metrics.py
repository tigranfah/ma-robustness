import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm

from pathlib import Path

from scripts.vit_imagenette import evaluate, evaluate_adv, get_loaders, make_model
from utils.lc_utils import get_intersections_for_hulls, get_ortho_hull_around_samples
from models.model_utils import add_hooks_vit


def compute_metrics_over_checkpoints(checkpoint_dir, config, test_loader, train_loader, device,
                                     adv_eps, adv_alpha, adv_itrs, adv_min, adv_max,
                                     n_frame=24, r_frame=0.005, seed=0, lc_batch_size=32,
                                     max_batches=2, top_k=1, save_path=None):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pth"),
                         key=lambda p: int(p.stem.split('_')[1]))

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
    n_layers = None
    steps = []
    all_train_accs = []
    all_test_accs = []
    all_test_accs_std = []
    all_adv_accs = {k: [] for k in ['pgd_ce', 'apgd_ce', 'apgd_dlr', 'square', 'union']}
    all_train_lc = []
    all_train_lc_std = []
    all_test_lc = []
    all_test_lc_std = []
    all_residual_maxes = []
    all_residual_maxes_std = []

    for ckpt_path in tqdm(checkpoints, desc="Checkpoints"):
        step = int(ckpt_path.stem.split('_')[1])

        model = make_model(config).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        if n_layers is None:
            n_layers = len(model.blocks)

        train_acc, _ = evaluate(model, train_loader, loss_fn, device, max_batches=max_batches)
        test_acc, _ = evaluate(model, test_loader, loss_fn, device, max_batches=max_batches)
        adv_acc = evaluate_adv(model, test_loader, adv_eps, adv_alpha, adv_itrs, 200, adv_min, adv_max, device=device, max_batches=max_batches)

        layer_residual_maxes = [[] for _ in range(n_layers)]

        for i, (ims, _) in enumerate(test_loader):
            if i >= max_batches:
                break
            ims = ims.to(device)

            with torch.no_grad():
                x = model.patch_embed(ims)
                x = model._pos_embed(x)
                x = model.patch_drop(x)
                x = model.norm_pre(x)

                for layer_idx, block in enumerate(model.blocks):
                    x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))

                    cls_residual = x[:, 0, :]
                    topk_residual = cls_residual.abs().topk(top_k, dim=1).values.sum(dim=1)
                    layer_residual_maxes[layer_idx].append(topk_residual.mean().item())

                    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        model_hooked, layer_names, activation_buffer = add_hooks_vit(model, verbose=False)
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

        steps.append(step)
        all_train_accs.append(train_acc.item() * 100.0)
        all_test_accs.append(test_acc.item() * 100.0)
        all_test_accs_std.append(0.0)
        for k in all_adv_accs:
            all_adv_accs[k].append(adv_acc[k] * 100.0)
        all_train_lc.append(train_lc_per_sample.mean().item())
        all_train_lc_std.append(train_lc_per_sample.std().item())
        all_test_lc.append(test_lc_per_sample.mean().item())
        all_test_lc_std.append(test_lc_per_sample.std().item())
        all_residual_maxes.append([np.mean(layer_residual_maxes[l]) for l in range(n_layers)])
        all_residual_maxes_std.append([np.std(layer_residual_maxes[l]) for l in range(n_layers)])

    metrics = {
        'steps': np.array(steps),
        'train_accs': np.array(all_train_accs),
        'test_accs': np.array(all_test_accs),
        'test_accs_std': np.array(all_test_accs_std),
        **{f'adv_accs_{k}': np.array(v) for k, v in all_adv_accs.items()},
        'train_lc': np.array(all_train_lc),
        'train_lc_std': np.array(all_train_lc_std),
        'test_lc': np.array(all_test_lc),
        'test_lc_std': np.array(all_test_lc_std),
        'residual_maxes': np.array(all_residual_maxes),
        'residual_maxes_std': np.array(all_residual_maxes_std),
        'n_layers': n_layers,
        'adv_eps': adv_eps,
        'top_k': top_k,
    }

    save_path = save_path = Path(save_path) / f"{checkpoint_dir.name}-adv_eps:{adv_eps}-r_frame:{r_frame}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **metrics)
    print(f"Metrics saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="utils/metrics")
    parser.add_argument('--model_name', type=str, default='vit_tiny_patch16_224')
    parser.add_argument('--dataset', type=str, default="imagenette")
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--classes', type=int, nargs='+', default=None)
    parser.add_argument('--max_batches', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')

    # adv robustness params
    parser.add_argument('--adv_eps', type=float, default=0.2)
    parser.add_argument('--adv_alpha', type=float, default=4/255)
    parser.add_argument('--adv_itrs', type=int, default=10)
    parser.add_argument("--adv_min", type=float, default=-2.1179)
    parser.add_argument("--adv_max", type=float, default=2.6400)

    # local complexity parameters
    parser.add_argument('--r_frame', type=float, default=0.005)
    parser.add_argument('--n_frame', type=int, default=25)
    parser.add_argument('--lc_batch_size', type=int, default=32)
    args = parser.parse_args()

    if args.classes is not None:
        args.num_class = len(args.classes)
    else:
        args.num_class = 10

    device = torch.device(args.device)

    train_loader, test_loader, mean, std = get_loaders(args, root="./data")

    compute_metrics_over_checkpoints(
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