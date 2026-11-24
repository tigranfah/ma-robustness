import argparse
import random
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.func import vmap, jacrev
import torchvision
import wandb as wb
from tqdm import tqdm

from models.fcn import FCN
from scripts.resnet18.attacks import PGD
from scripts.resnet18.utils import add_hooks_fcn
from scripts.resnet18.local_complexity import get_intersections_for_hulls
from scripts.script_utils import GrokAlign, get_hulls


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_mnist_loaders(train_points, test_points, batch_size, classes, data_dir='./data'):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        torchvision.transforms.Lambda(torch.flatten)
    ])
    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    train_idx = [i for i, (_, label) in enumerate(train_set) if label in classes]
    test_idx  = [i for i, (_, label) in enumerate(test_set) if label in classes]

    train_subset = torch.utils.data.Subset(train_set, train_idx)
    test_subset = torch.utils.data.Subset(test_set, test_idx)

    train_subset = torch.utils.data.Subset(train_subset, range(train_points))
    test_subset = torch.utils.data.Subset(test_subset, range(test_points))
    
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return {'train': train_loader, 'test': test_loader}


def evaluate_adv(
    model,
    dloader,
    eps,
    alpha,
    itrs,
    dmin,
    dmax,
    device
):
    model.eval()
    atk = PGD(
        model,
        eps=eps,
        alpha=alpha,
        steps=itrs,
        dmin=dmin,
        dmax=dmax,
        device=device
    )

    acc = 0
    nsamples = 0
    for ims, labs in dloader:
        ims = ims.to(device)
        labs = labs.to(device)

        adv_images = atk(ims, labs)

        with torch.no_grad():
            adv_pred = model(adv_images).argmax(dim=-1)

        acc += torch.sum(labs == adv_pred).cpu()
        nsamples += adv_pred.shape[0]

    model.train()
    return acc / nsamples


def compute_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += x.size(0)
    model.train()
    return correct / total


def save_model(exp_id, model, optimizer, step):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    model_save_path = f"checkpoints/{exp_id}"
    os.makedirs(model_save_path, exist_ok=True)
    checkpoint_path = f"{model_save_path}/checkpoint_{step}.pth"
    torch.save(checkpoint, checkpoint_path)


def effective_rank(x, eps=1e-12):
    s = x.abs()
    p = s / s.sum(dim=-1, keepdim=True)
    H = -(p * torch.log(p + eps)).sum(dim=-1)
    return torch.exp(H)


def ejac_rank(model, x, device, power_iters=5):
    Js = vmap(jacrev(lambda inp: model(inp.unsqueeze(0)).squeeze(0)))(x.to(device))
    F_norm_sq = (Js ** 2).sum(dim=(-2, -1)).mean()
    
    B = Js.shape[0]
    v = torch.randn(B, Js.shape[2], device=device)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    
    with torch.no_grad():
        for _ in range(power_iters):
            u = torch.bmm(Js.detach(), v.unsqueeze(-1)).squeeze(-1)
            v = torch.bmm(Js.detach().transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1)
            v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    
    u = torch.bmm(Js, v.detach().unsqueeze(-1)).squeeze(-1)
    spec_norm_sq = (u ** 2).sum(dim=-1).mean()
    
    return F_norm_sq / (spec_norm_sq + 1e-8)


def compute_alignment(model, loader, device):
    model.eval()
    total_alignment = 0.0
    total_samples = 0
    
    for x, _ in loader:
        x = x.to(device)
        x.requires_grad_(True)
        logits = model(x)
        c_star = logits.argmax(dim=1)
        
        batch_indices = torch.arange(x.size(0), device=device)
        pred_logits = logits[batch_indices, c_star]
        
        w = torch.autograd.grad(
            outputs=pred_logits, 
            inputs=x, 
            grad_outputs=torch.ones_like(pred_logits),
            create_graph=False,
            retain_graph=False
        )[0]
        
        x_flat = x.view(x.size(0), -1)
        w_flat = w.view(w.size(0), -1)
        
        w_norm = torch.norm(w_flat, p=2, dim=1) + 1e-8
        dot_product = torch.sum(x_flat * w_flat, dim=1)
        alignment = torch.abs(dot_product) / w_norm
        
        total_alignment += alignment.sum().item()
        total_samples += x.size(0)
        
    model.train()
    return total_alignment / total_samples


@torch.no_grad
def layer_stats(model, loader, device):
    model.eval()
    repr_dict = {}
    max_act_dict = {}
    x, _ = next(iter(loader))
    x = x.to(device)
    for i, layer in tqdm(enumerate(model.layers)):
        x = layer(x)
        repr_dict[f"layer/{i+1} repr eff. rank"] = effective_rank(x).mean().item()
        max_act_dict[f"layer/{i+1} repr max act"] = x.max(dim=-1).values.mean().item()
    model.train()
    return repr_dict, max_act_dict


def train(config, device='cuda', data_dir='./data', dtype=torch.float32):
    run_name = f"{config.exp_id}-{config.loss_fn}"
    run_name += f"-D={config.depth}"
    run_name += f"-H={config.hidden_dim}"
    run_name += f"-BT={config.block_type}"
    run_name += f"{f'-JnormLam={config.lambda_jac}' if config.lambda_jac > 0 else ''}"
    run_name += f"{f'-JnormLvl={config.jac_norm_level}' if config.jac_norm_level > 0 else ''}"
    run_name += f"{f'-JnormLam={config.lambda_jac_erank}' if config.lambda_jac_erank > 0 else ''}"
    run_name += f"{f'-JrankLvl={config.jac_erank_level}' if config.jac_erank_level > 0 else ''}"
    run_name += f"{f'-Wd={config.weight_decay}' if config.weight_decay > 0 else ''}"
    run_name += f"{'-RS' if config.residual_stream else ''}"
    run_name += f"{'-BN' if config.norm_type == 'batch_norm' else ''}"
    run_name += f"{'-LN' if config.norm_type == 'layer_norm' else ''}"
    run_name += f"{'-nB' if not config.bias else ''}"
    run_name += f"-init_sc={config.init_scale}"
    run_name += f"{'-At' if config.adv_training else ''}-{config.seed}"
    run = wb.init(project='fcn_mnist_grokking', config=vars(config), name=run_name, mode="disabled")

    torch.set_default_dtype(dtype)
    set_seed(config.seed)

    loaders = get_mnist_loaders(config.train_points, config.test_points, config.batch_size, classes=config.mnist_classes, data_dir=data_dir)
    hulls = get_hulls(config, loaders['train'], loaders['test'], normalize=True)
    model = FCN(
        depth=config.depth,
        input_dim=784,
        output_dim=len(config.mnist_classes),
        hidden_dim=config.hidden_dim,
        bias=config.bias,
        act_fn=config.act_fn,
        norm_type=config.norm_type,
        residual_stream=config.residual_stream,
        init_scale=config.init_scale,
        block_type=config.block_type
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    grokalign = GrokAlign(model)
    one_hots = torch.eye(len(config.mnist_classes), device=device)

    wb.log({
        "depth": config.depth,
    })

    if config.compute_LC:
        model, layer_names, activation_buffer = add_hooks_fcn(model, verbose=False)

    total_time = 0
    pbar = tqdm(range(0, config.steps))
    log_steps = np.unique(np.logspace(0, np.log10(config.steps), 30).astype(int)).tolist()

    for step in pbar:
        model.train()
        start_time = time.time()

        if step in log_steps:
            save_model(run_name, model, optimizer, step)

        if config.adv_training:
            train_acc = compute_accuracy(model, loaders['train'], device)
            sigma = max(0.06 * (1 - train_acc), 0.03)

        for x, labels in loaders['train']:
            x, labels = x.to(device), labels.to(device)
            if config.adv_training:
                x += torch.randn_like(x) * sigma

            optimizer.zero_grad()
            outputs = model(x)

            if config.loss_fn == 'CrossEntropy':
                task_loss = nn.CrossEntropyLoss()(outputs, labels)
            elif config.loss_fn == 'MSE':
                task_loss = nn.MSELoss()(outputs, one_hots[labels])
            else:
                raise ValueError(f"Unsupported loss function: {config.loss_fn}")

            loss = task_loss
            if config.lambda_jac > 0.0:
                loss += config.lambda_jac * torch.abs(grokalign(x) - config.jac_norm_level)
            if config.lambda_jac_erank > 0.0:
                loss += config.lambda_jac_erank * torch.abs(ejac_rank(model, x, device=device) - config.jac_erank_level)

            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

        total_time += time.time() - start_time
        pbar.set_description(f"{loss.item():.4f}")

        if step % 100 == 0:
            stats = {
                'step': step,
                'loss': loss.item(),
                'task_loss': task_loss.item(),
                'optimization_time': total_time
            }
            for split in ['train', 'test']:
                stats[f'{split}_accuracy'] = compute_accuracy(model, loaders[split], device)
                stats[f'{split}_alignment'] = compute_alignment(model, loaders[split], device)
                
                sample_x, _ = next(iter(loaders[split]))
                stats[f'{split}_jac_norm'] = grokalign(sample_x)
                stats[f'{split}_jac_erank'] = ejac_rank(model, sample_x, device=device)

            if len(config.adv_eps) > 0:
                for eps in config.adv_eps:
                    adv_acc = evaluate_adv(model, loaders['test'], eps, config.adv_alpha, config.adv_itrs, config.dmin, config.dmax, device)
                    stats[f'test_adv_accuracy_{eps:.2f}'] = adv_acc

            repr_dict, max_act_dict = layer_stats(model, loaders['test'], device)
            stats.update(repr_dict)
            stats.update(max_act_dict)

            if config.compute_LC:
                if hulls is not None:
                    for k in hulls.keys():
                        # compute number of neurons that intersect hulls
                        # using network activations
                        with torch.no_grad():
                            n_inters, _ = get_intersections_for_hulls(
                                hulls[k],
                                model=model,
                                batch_size=config.LC_batch_size,
                                layer_names=layer_names,
                                activation_buffer=activation_buffer
                            )

                        stats[k+'_LC'] = n_inters.float().mean().item()

            wb.log(stats)

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default="none")
    parser.add_argument('--train_points', type=int, default=1024)
    parser.add_argument('--test_points', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--residual_stream', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--norm_type', type=str, choices=['batch_norm', 'layer_norm'], default=None)
    parser.add_argument('--act_fn', type=str, choices=['relu'], default="relu")
    parser.add_argument('--block_type', type=str, choices=["pre_act", "pre", "post"], default="pre_act")
    parser.add_argument('--lambda_jac', type=float, default=0.0)
    parser.add_argument('--jac_norm_level', type=float, default=0.0)
    parser.add_argument('--lambda_jac_erank', type=float, default=0.0)
    parser.add_argument('--jac_erank_level', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--adv_training', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--init_scale', type=float, default=8.0)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--loss_fn', type=str, choices=['MSE', 'CrossEntropy'], default='MSE')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mnist_classes', type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--dmin', type=float, default=-0.4242)
    parser.add_argument('--dmax', type=float, default=2.8215)

    # adversarial attack parameters
    parser.add_argument('--adv_eps', type=float, nargs="+", default=[0.03, 0.06, 0.1, 0.13, 0.16, 0.2])
    parser.add_argument('--adv_alpha', type=float, default=4/255)
    parser.add_argument('--adv_itrs', type=int, default=10)

    # local complexity parameters
    parser.add_argument('--compute_LC', action='store_true') # number of samples to use for approximation
    parser.add_argument('--approx_n', type=int, default=1024) # number of samples to use for approximation
    parser.add_argument('--n_frame', type=int, default=25) # number of vertices for neighborhood
    parser.add_argument('--r_frame', type=float, default=0.005) # radius of \ell-1 ball neighborhood
    parser.add_argument('--LC_batch_size', type=int, default=256)
    parser.add_argument('--inc_centroid', action='store_true') # include original sample as neighborhood vertex

    args = parser.parse_args()
    train(args, device=args.device)
