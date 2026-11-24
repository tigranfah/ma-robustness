import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from models.fcn import FCN
from utils.lc_utils import get_intersections_for_hulls, get_ortho_hull_around_samples
from models.model_utils import add_hooks_fcn
from scripts.script_utils import get_hulls
from utils.metrics import compute_jacobian_rank, compute_jacobian_alignment, compute_alignment


def get_loaders(args):
    from scripts.fcn_mnist_grokking import get_mnist_loaders, evaluate_adv, compute_accuracy
    loaders = get_mnist_loaders(args.train_points, args.test_points, args.batch_size, classes=list(range(10)))
    return loaders['train'], loaders['test'], 784, 10, evaluate_adv, compute_accuracy


def compute_metrics_over_checkpoints(
    checkpoint_dir,
    config,
    train_loader,
    test_loader,
    input_dim,
    output_dim,
    evaluate_adv,
    compute_accuracy,
    device,
    adv_eps,
    adv_alpha,
    adv_itrs,
    dmin,
    dmax,
    lc_batch_size,
    max_batches,
    top_k,
    save_path,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pth"),
                         key=lambda p: int(p.stem.split('_')[1]))

    hulls = get_hulls(config, train_loader, test_loader, normalize=True)
    train_hulls = hulls['train']
    test_hulls = hulls['test']
    rand_hulls = hulls['rand']
    print(f"Train hulls: {train_hulls.shape}, Test hulls: {test_hulls.shape}, Rand hulls: {rand_hulls.shape}")

    n_layers = None
    steps = []
    all_train_accs = []
    all_test_accs = []
    all_adv_accs = []
    all_train_lc = []
    all_train_lc_std = []
    all_test_lc = []
    all_test_lc_std = []
    all_rand_lc = []
    all_rand_lc_std = []
    all_residual_maxes = []
    all_residual_maxes_std = []
    all_jac_ranks = []
    all_jac_alignments = []
    all_alignments = []

    for ckpt_path in tqdm(checkpoints, desc="Checkpoints"):
        step = int(ckpt_path.stem.split('_')[1])

        model = FCN(
            depth=config.depth,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            bias=config.bias,
            act_fn=config.act_fn,
            norm_type=config.norm_type,
            residual_stream=config.residual_stream,
            init_scale=config.init_scale,
            block_type=config.block_type,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        if n_layers is None:
            n_layers = len(model.layers)

        train_acc = compute_accuracy(model, train_loader, device)
        test_acc = compute_accuracy(model, test_loader, device)
        adv_acc = evaluate_adv(model, test_loader, adv_eps, adv_alpha, adv_itrs, dmin, dmax, device)

        layer_residual_maxes = [[] for _ in range(n_layers)]
        with torch.no_grad():
            for i, (x, _) in enumerate(test_loader):
                if max_batches is not None and i >= max_batches:
                    break
                x = x.to(device)
                h = x
                for layer_idx, layer in enumerate(model.layers):
                    h = layer(h)
                    topk_val = h.abs().topk(top_k, dim=-1).values.sum(dim=-1)
                    layer_residual_maxes[layer_idx].append(topk_val.mean().item())

        model_hooked, layer_names, activation_buffer = add_hooks_fcn(model, verbose=False)
        n_inters_train, _ = get_intersections_for_hulls(
            hulls=train_hulls, model=model_hooked, layer_names=layer_names,
            activation_buffer=activation_buffer, batch_size=lc_batch_size, verbose=False)
        n_inters_test, _ = get_intersections_for_hulls(
            hulls=test_hulls, model=model_hooked, layer_names=layer_names,
            activation_buffer=activation_buffer, batch_size=lc_batch_size, verbose=False)
        n_inters_rand, _ = get_intersections_for_hulls(
            hulls=rand_hulls, model=model_hooked, layer_names=layer_names,
            activation_buffer=activation_buffer, batch_size=lc_batch_size, verbose=False)

        train_lc_per_sample = n_inters_train.float().sum(dim=1)
        test_lc_per_sample = n_inters_test.float().sum(dim=1)
        rand_lc_per_sample = n_inters_rand.float().sum(dim=1)

        jac_rank = compute_jacobian_rank(model, test_loader, device, max_batches)
        jac_alignment = compute_jacobian_alignment(model, test_loader, device, max_batches)
        alignment = compute_alignment(model, test_loader, device, max_batches)

        steps.append(step)
        all_train_accs.append(train_acc * 100.0)
        all_test_accs.append(test_acc * 100.0)
        all_adv_accs.append(adv_acc.item() * 100.0)
        all_train_lc.append(train_lc_per_sample.mean().item())
        all_train_lc_std.append(train_lc_per_sample.std().item())
        all_test_lc.append(test_lc_per_sample.mean().item())
        all_test_lc_std.append(test_lc_per_sample.std().item())
        all_rand_lc.append(rand_lc_per_sample.mean().item())
        all_rand_lc_std.append(rand_lc_per_sample.std().item())
        all_residual_maxes.append([np.mean(layer_residual_maxes[l]) for l in range(n_layers)])
        all_residual_maxes_std.append([np.std(layer_residual_maxes[l]) for l in range(n_layers)])
        all_jac_ranks.append(jac_rank)
        all_jac_alignments.append(jac_alignment)
        all_alignments.append(alignment)

        del model, model_hooked
        torch.cuda.empty_cache()

    metrics = {
        'steps': np.array(steps),
        'train_accs': np.array(all_train_accs),
        'test_accs': np.array(all_test_accs),
        'adv_accs': np.array(all_adv_accs),
        'train_lc': np.array(all_train_lc),
        'train_lc_std': np.array(all_train_lc_std),
        'test_lc': np.array(all_test_lc),
        'test_lc_std': np.array(all_test_lc_std),
        'rand_lc': np.array(all_rand_lc),
        'rand_lc_std': np.array(all_rand_lc_std),
        'residual_maxes': np.array(all_residual_maxes),
        'residual_maxes_std': np.array(all_residual_maxes_std),
        'jac_ranks': np.array(all_jac_ranks),
        'jac_alignments': np.array(all_jac_alignments),
        'alignments': np.array(all_alignments),
        'n_layers': n_layers,
        'adv_eps': adv_eps,
        'top_k': top_k
    }

    save_path = Path(save_path) / f"{checkpoint_dir.name}-r_frame:{config.r_frame}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **metrics)
    print(f"Metrics saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='utils/metrics')
    parser.add_argument('--dataset', choices=['mnist'], default='mnist')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_batches', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--train_points', type=int, default=1024)
    parser.add_argument('--test_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--residual_stream', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--act_fn', type=str, default='relu')
    parser.add_argument('--norm_type', type=str, default=None)
    parser.add_argument('--block_type', type=str, default='pre_act')
    parser.add_argument('--init_scale', type=float, default=8.0)
    parser.add_argument('--adv_eps', type=float, default=0.2)
    parser.add_argument('--adv_alpha', type=float, default=4/255)
    parser.add_argument('--adv_itrs', type=int, default=10)
    parser.add_argument('--dmin', type=float, default=None)
    parser.add_argument('--dmax', type=float, default=None)
    parser.add_argument('--n_frame', type=int, default=25)
    parser.add_argument('--r_frame', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lc_batch_size', type=int, default=256)
    args = parser.parse_args()

    if args.dmin is None or args.dmax is None:
        args.dmin = -0.4242
        args.dmax = 2.8215

    device = torch.device(args.device)
    train_loader, test_loader, input_dim, output_dim, evaluate_adv, compute_accuracy = get_loaders(args)

    compute_metrics_over_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        config=args,
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        output_dim=output_dim,
        evaluate_adv=evaluate_adv,
        compute_accuracy=compute_accuracy,
        device=device,
        adv_eps=args.adv_eps,
        adv_alpha=args.adv_alpha,
        adv_itrs=args.adv_itrs,
        dmin=args.dmin,
        dmax=args.dmax,
        lc_batch_size=args.lc_batch_size,
        max_batches=args.max_batches,
        top_k=args.top_k,
        save_path=args.save_path,
    )
