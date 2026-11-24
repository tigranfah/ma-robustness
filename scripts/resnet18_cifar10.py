from tqdm import tqdm
import argparse
import os
import random

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import numpy as np
import wandb as wb

from scripts.resnet18.models import make_resnet18k
from scripts.resnet18.dataloaders import cifar10_dataloaders
from scripts.resnet18.attacks import PGD


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def evaluate(model, dloader, loss_fn=None, max_batches=None):

    acc = 0
    loss = 0
    nsamples = 0
    nbatch = 0

    for i, (ims, labs) in enumerate(dloader):
        if max_batches is not None and i >= max_batches:
            break

        ims = ims.cuda()
        labs = labs.cuda()

        outs = model(ims)

        if loss_fn is not None:
            loss += loss_fn(outs, labs)
            nbatch += 1

        acc += torch.sum(labs == outs.argmax(dim=-1)).cpu()
        nsamples += outs.shape[0]

    return acc/nsamples, loss/nbatch


def evaluate_adv(
    model,
    dloader,
    eps,
    alpha,
    itrs,
    dmin,
    dmax,
    device,
    max_batches=None
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
    for i, (ims, labs) in enumerate(dloader):
        if max_batches is not None and i >= max_batches:
            break
        ims = ims.to(device)
        labs = labs.to(device)

        adv_images = atk(ims, labs)

        with torch.no_grad():
            adv_pred = model(adv_images).argmax(dim=-1)

        acc += torch.sum(labs == adv_pred).cpu()
        nsamples += adv_pred.shape[0]

    model.train()
    return acc / nsamples


def effective_rank(x, eps=1e-12):
    B, C, H, W = x.shape
    z = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
    s = torch.linalg.svdvals(z)
    eigvals = s**2
    p = eigvals / (eigvals.sum(dim=-1, keepdim=True) + eps)
    H = -(p * torch.log(p + eps)).sum(dim=-1)
    return torch.exp(H)


@torch.no_grad()
def layer_repr(model, loader, device):
    model.eval()
    repr_dict = {}
    x, _ = next(iter(loader))
    x = x.to(device)
    
    # Initial conv layer
    x = model.conv1(x)
    
    # Go through each main layer group
    for layer_idx, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        x = layer(x)
        repr_dict[f"layer/{layer_idx} repr eff. rank"] = effective_rank(x).mean().item()
    
    model.train()
    return repr_dict

def save_model(exp_id, model, optimizer, step):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    model_save_path = f"checkpoints/{exp_id}"
    os.makedirs(model_save_path, exist_ok=True)
    checkpoint_path = f"{model_save_path}/checkpoint_{step}.pth"
    torch.save(checkpoint, checkpoint_path)

def train(config, device='cuda', dtype=torch.float32):
    run_name = f"{config.exp_id}"
    run_name += f"{'-GF' if config.grokfast else ''}"
    run_name += f"{f'-GA={config.lambda_jac}' if config.lambda_jac > 0 else ''}"
    run_name += f"{f'-Wd={config.weight_decay}' if config.weight_decay > 0 else ''}"
    wb.init(project="resnet18_cifar10_grokking", name=run_name, config=vars(config), mode="disabled")

    torch.set_default_dtype(dtype)
    set_seed(config.seed)

    train_loader, test_loader = cifar10_dataloaders(config)

    loaders = {
        'train' : train_loader,
        'test' : test_loader
    }

    model = make_resnet18k(num_classes=config.num_class, bn=config.use_bn)

    model.cuda()

    opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    iters_per_epoch = len(loaders['train'])
    epochs = np.floor(config.num_steps/iters_per_epoch)

    loss_fn = CrossEntropyLoss(label_smoothing=config.label_smoothing)

    train_step = 0

    log_steps = np.unique(np.logspace(0,5.7,30).astype(int).clip(0,config.num_steps))

    for step in range(config.num_steps):
        
        for ims, labs in tqdm(loaders['train'], desc=f"train_step:{train_step}-{train_step+iters_per_epoch}"):

            ims = ims.cuda()
            labs = labs.cuda()

            if config.use_ffcv and labs.max()>config.num_class-1:
              labs = torch.clip(labs,0,config.num_class-1) ## weird ffcv bug

            opt.zero_grad()
            out = model(ims)
            loss = loss_fn(out, labs)
            loss.backward()
            opt.step()
            train_step += 1

            if train_step in log_steps:
                model.eval()

                train_acc, train_loss = evaluate(model, loaders['train'], loss_fn, max_batches=5)
                test_acc, test_loss = evaluate(model,loaders['test'], loss_fn, max_batches=5)

                stats = {}
                stats['step'] = train_step
                stats['train_acc'] = train_acc
                stats['test_acc'] = test_acc
                stats['train_loss'] = train_loss
                stats['test_loss'] = test_loss

                if len(config.adv_eps) > 0:
                    for eps in config.adv_eps:
                        adv_acc = evaluate_adv(model, loaders['test'], eps, config.adv_alpha, config.adv_itrs, config.adv_min, config.adv_max, config.device, max_batches=5)
                        stats[f'test_adv_acc_{eps:.2f}'] = adv_acc

                stats.update(layer_repr(model, loaders['test'], device))

                wb.log(stats)
                model.train()

                if train_step in log_steps:
                    save_model(run_name, model, opt, train_step)

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default="none")
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lambda_jac', type=float, default=0.0)
    parser.add_argument('--grokfast', action='store_true')
    parser.add_argument('--adv_training', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--use_ffcv', action='store_true')
    parser.set_defaults(use_ffcv=True)
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')

    # adversarial robustness parameters
    parser.add_argument('--adv_eps', type=float, nargs="+", default=[0.03, 0.06, 0.1, 0.13, 0.16, 0.2])
    parser.add_argument('--adv_min', type=float, default=-2.4291)
    parser.add_argument('--adv_max', type=float, default=2.8215)
    parser.add_argument('--adv_alpha', type=float, default=4/255)
    parser.add_argument('--adv_itrs', type=int, default=10)

    args = parser.parse_args()
    train(args)
