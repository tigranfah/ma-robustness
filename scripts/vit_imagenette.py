from tqdm import tqdm
import argparse
import os
import random

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import numpy as np
import wandb as wb
import torchattacks
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import timm

from scripts.script_utils import GrokAlign, SplitLayerNorm
from scripts.resnet18.attacks import APGD, PGD, Square


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]


class FilteredDataset(Dataset):
    def __init__(self, original_dataset, allowed_classes):
        self.original_dataset = original_dataset
        self.allowed_classes = sorted(list(set(allowed_classes)))
        self.class_mapping = {orig: new for new, orig in enumerate(self.allowed_classes)}

        if hasattr(original_dataset, 'targets'):
            targets = original_dataset.targets
        elif hasattr(original_dataset, '_samples'):
            targets = [label for _, label in original_dataset._samples]
        else:
            raise AttributeError("Dataset does not have 'targets' or '_samples' attribute")

        self.indices = [i for i, target in enumerate(targets) if target in self.allowed_classes]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, original_target = self.original_dataset[original_idx]
        mapped_target = self.class_mapping[original_target]
        return img, mapped_target


def get_loaders(config, root="./data"):
    if config.dataset == "imagenette":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        train_dataset = datasets.Imagenette(root=root, split='train', size='full', download=True, transform=transform)
        test_dataset = datasets.Imagenette(root=root, split='val', size='full', download=True, transform=transform)

        if config.classes is not None:
            train_dataset = FilteredDataset(train_dataset, config.classes)
            test_dataset = FilteredDataset(test_dataset, config.classes)

    elif config.dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

        if config.classes is not None:
            train_dataset = FilteredDataset(train_dataset, config.classes)
            test_dataset = FilteredDataset(test_dataset, config.classes)

    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=True, num_workers=4)

    return train_loader, test_loader, mean, std


@torch.no_grad()
def evaluate(model, dloader, loss_fn, device, max_batches=None):
    acc = 0
    loss = 0
    nsamples = 0
    nbatch = 0

    for i, (ims, labs) in enumerate(dloader):
        if max_batches is not None and i >= max_batches:
            break
        ims = ims.to(device)
        labs = labs.to(device)
        outs = model(ims)
        loss += loss_fn(outs, labs)
        nbatch += 1
        acc += torch.sum(labs == outs.argmax(dim=-1)).cpu()
        nsamples += outs.shape[0]

    return acc / nsamples, loss / nbatch


def compute_jacobian_norm(model, dloader, device, max_batches=None):
    model.eval()

    flash_was_enabled = torch.backends.cuda.flash_sdp_enabled()
    mem_eff_was_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
    math_was_enabled = torch.backends.cuda.math_sdp_enabled()

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    total_norm = 0.0
    samples_processed = 0

    for i, (ims, _) in enumerate(dloader):
        if max_batches is not None and i >= max_batches:
            break
        ims = ims.to(device)
        ims.requires_grad_(True)
        logits = model(ims)
        B, K = logits.shape
        batch_norm_sq = torch.zeros(B, device=device)

        for k in range(K):
            grad_x, = torch.autograd.grad(
                outputs=logits[:, k],
                inputs=ims,
                grad_outputs=torch.ones_like(logits[:, k]),
                create_graph=False,
                retain_graph=True,
            )
            batch_norm_sq += (grad_x ** 2).sum(dim=(1, 2, 3))

        total_norm += torch.sqrt(batch_norm_sq).sum().item()
        samples_processed += B

    torch.backends.cuda.enable_flash_sdp(flash_was_enabled)
    torch.backends.cuda.enable_mem_efficient_sdp(mem_eff_was_enabled)
    torch.backends.cuda.enable_math_sdp(math_was_enabled)

    return total_norm / samples_processed


def compute_alignment(model, loader, device, max_batches=None):
    model.eval()

    flash_was_enabled = torch.backends.cuda.flash_sdp_enabled()
    mem_eff_was_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
    math_was_enabled = torch.backends.cuda.math_sdp_enabled()

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    total_alignment = 0.0
    total_samples = 0

    for i, (x, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        x.requires_grad_(True)
        logits = model(x)
        c_star = logits.argmax(dim=1)
        pred_logits = logits[torch.arange(x.size(0), device=device), c_star]

        w = torch.autograd.grad(
            outputs=pred_logits.sum(),
            inputs=x,
            create_graph=False,
            retain_graph=False,
        )[0]

        x_flat = x.view(x.size(0), -1)
        w_flat = w.view(w.size(0), -1)
        w_norm = torch.norm(w_flat, p=2, dim=1) + 1e-8
        dot_product = torch.sum(x_flat * w_flat, dim=1)
        alignment = torch.abs(dot_product) / w_norm

        total_alignment += alignment.sum().item()
        total_samples += x.size(0)

    torch.backends.cuda.enable_flash_sdp(flash_was_enabled)
    torch.backends.cuda.enable_mem_efficient_sdp(mem_eff_was_enabled)
    torch.backends.cuda.enable_math_sdp(math_was_enabled)

    model.train()
    return total_alignment / total_samples


def evaluate_adv(model, dloader, eps, alpha, steps, n_queries, adv_min, adv_max, device, max_batches=None):
    model.eval()

    pgd_ce = PGD(model, eps=eps, steps=steps, alpha=alpha, dmin=adv_min, dmax=adv_max, device=device)
    apgd_ce = APGD(model, norm='Linf', eps=eps, steps=steps, loss='ce', dmin=adv_min, dmax=adv_max, device=device)
    apgd_dlr = APGD(model, norm='Linf', eps=eps, steps=steps, loss='dlr', dmin=adv_min, dmax=adv_max, device=device)
    square = Square(model, norm='Linf', eps=eps, n_queries=n_queries, n_restarts=1, dmin=adv_min, dmax=adv_max)

    n_rob_pgd_ce = 0
    n_rob_ce = 0
    n_rob_dlr = 0
    n_rob_sq = 0
    n_rob_union = 0
    nsamples = 0

    for i, (ims, labs) in enumerate(dloader):
        if max_batches is not None and i >= max_batches:
            break
        ims = ims.to(device)
        labs = labs.to(device)

        with torch.no_grad():
            correct_mask = model(ims).argmax(1) == labs

        ims_c = ims[correct_mask]
        labs_c = labs[correct_mask]
        nsamples += ims.size(0)

        if ims_c.size(0) == 0:
            continue

        with torch.no_grad():
            found_pgd_ce = model(pgd_ce(ims_c, labs_c)).argmax(1) != labs_c
            found_ce = model(apgd_ce(ims_c, labs_c)).argmax(1) != labs_c
            found_dlr = model(apgd_dlr(ims_c, labs_c)).argmax(1) != labs_c
            found_sq = model(square(ims_c, labs_c)).argmax(1) != labs_c
            found_any = found_pgd_ce | found_ce | found_dlr | found_sq

        n_rob_pgd_ce += (~found_pgd_ce).sum().item()
        n_rob_ce += (~found_ce).sum().item()
        n_rob_dlr += (~found_dlr).sum().item()
        n_rob_sq += (~found_sq).sum().item()
        n_rob_union += (~found_any).sum().item()

    denom = max(nsamples, 1)
    model.train()
    return {
        'pgd_ce': n_rob_pgd_ce / denom,
        'apgd_ce': n_rob_ce / denom,
        'apgd_dlr': n_rob_dlr / denom,
        'square': n_rob_sq / denom,
        'union': n_rob_union / denom,
    }


def save_model(checkpoint_dir, exp_id, model, optimizer, step):
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    model_save_path = f"{checkpoint_dir}/checkpoints/{exp_id}"
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(checkpoint, f"{model_save_path}/checkpoint_{step}.pth")


def make_model(config):
    if config.dataset == "imagenette":
        model = timm.create_model(
            config.model_name,
            pretrained=False,
            num_classes=config.num_class
        )
    elif config.dataset == "cifar10":
        model = timm.create_model(
            config.model_name,
            pretrained=False,
            num_classes=config.num_class,
            img_size=32,
            patch_size=4
        )

    return model


def train(config, dtype=torch.float32):
    if config.lambda_jac > 0:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    run_name = f"{config.exp_id}-{config.model_name.split('_')[1]}-{config.dataset}"
    run_name += f"{'-splitLN' if config.split_ln else '-LN'}"
    run_name += f"{'-noMlpLN' if config.disable_mlp_ln else ''}"
    run_name += f"{f'-GA={config.lambda_jac}' if config.lambda_jac > 0 else ''}"
    run_name += f"{f'-Wd={config.weight_decay}' if config.weight_decay > 0 else ''}"
    run_name += f"{f'-At={config.adv_train_eps}' if config.adv_training else ''}"
    wb.init(project="vit_grokking", name=run_name, config=vars(config), mode="disabled")

    torch.set_default_dtype(dtype)
    set_seed(config.seed)

    train_loader, test_loader, mean, std = get_loaders(config)

    loaders = {
        'train': train_loader,
        'test': test_loader,
    }

    model = make_model(config).to(config.device)

    opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    grokalign = GrokAlign(model)

    iters_per_epoch = len(loaders['train'])
    loss_fn = CrossEntropyLoss(label_smoothing=config.label_smoothing)

    train_step = 0

    if config.resume_checkpoint is not None:
        if os.path.isfile(config.resume_checkpoint):
            print(f"=> Loading checkpoint '{config.resume_checkpoint}'")
            checkpoint = torch.load(config.resume_checkpoint, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'step' in checkpoint:
                train_step = checkpoint['step']
            else:
                try:
                    filename = os.path.basename(config.resume_checkpoint)
                    train_step = int(filename.split('_')[1].split('.')[0])
                except Exception as e:
                    print(f"Could not parse step from filename, defaulting to 0. Error: {e}")

            print(f"Resuming the training from {train_step} step")
        else:
            print(f"No checkpoint found at '{config.resume_checkpoint}', starting from scratch")

    log_steps = np.unique(np.logspace(0, 6.7, 30).astype(int).clip(1, config.num_steps)).tolist()
    print(f"Total steps: {config.num_steps}, log_steps: {sorted(log_steps)}")

    stats = {}

    while train_step < config.num_steps:

        for ims, labs in tqdm(loaders['train'], desc=f"train_step:{train_step}-{train_step+iters_per_epoch}"):
            ims = ims.to(config.device)
            labs = labs.to(config.device)

            if config.adv_training:
                model.eval()
                atk = torchattacks.APGD(model, norm='Linf', eps=config.adv_train_eps,
                                         steps=config.adv_itrs, loss='ce')
                ims = atk(ims, labs).detach()
                model.train()

            opt.zero_grad()
            out = model(ims)

            task_loss = loss_fn(out, labs)
            loss = task_loss

            if config.lambda_jac > 0.0:
                loss += config.lambda_jac * grokalign(ims)

            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()
            train_step += 1

            if train_step in log_steps:
                model.eval()

                train_acc, train_loss = evaluate(model, loaders['train'], loss_fn, device=config.device, max_batches=5)
                test_acc, test_loss = evaluate(model, loaders['test'], loss_fn, device=config.device, max_batches=5)
                jac_norm = compute_jacobian_norm(model, loaders['test'], device=config.device, max_batches=5)
                alignment = compute_alignment(model, loaders['test'], device=config.device, max_batches=5)

                stats = {
                    'step': train_step,
                    'loss': loss.item(),
                    'task_loss': task_loss.item(),
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'test_jac_norm': jac_norm,
                    'test_alignment': alignment,
                }

                if len(config.adv_eps) > 0:
                    for eps in config.adv_eps:
                        adv = evaluate_adv(model, loaders['test'], eps, config.adv_alpha,
                                           config.adv_itrs, config.adv_n_queries,
                                           adv_min=config.adv_min, adv_max=config.adv_max,
                                           device=config.device, max_batches=5)
                        stats[f'test_adv_apgd_ce_{eps:.2f}']  = adv['apgd_ce']
                        stats[f'test_adv_apgd_dlr_{eps:.2f}'] = adv['apgd_dlr']
                        stats[f'test_adv_square_{eps:.2f}']   = adv['square']
                        stats[f'test_adv_union_{eps:.2f}']    = adv['union']

                wb.log(stats)
                model.train()

                save_model(config.checkpoint_dir, run_name, model, opt, train_step)

            if train_step >= config.num_steps:
                break

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default="none")
    parser.add_argument('--model_name', type=str, default='vit_tiny_patch16_224')
    parser.add_argument('--dataset', type=str, default="imagenette", choices=["imagenette", "cifar10"])
    parser.add_argument('--split_ln', action='store_true')
    parser.add_argument('--disable_mlp_ln', action='store_true')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lambda_jac', type=float, default=0.0)
    parser.add_argument('--adv_training', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--classes', type=int, nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default=".")
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--adv_train_eps', type=float, default=0.03)
    parser.add_argument('--adv_eps', type=float, nargs="+", default=[0.03, 0.06, 0.1, 0.13, 0.16, 0.2])
    parser.add_argument('--adv_alpha', type=float, default=4/255)
    parser.add_argument('--adv_itrs', type=int, default=10)
    parser.add_argument("--adv_min", type=float, default=-2.1179)
    parser.add_argument("--adv_max", type=float, default=2.6400)
    parser.add_argument('--adv_n_queries', type=int, default=200)
    args = parser.parse_args()

    if args.classes is not None:
        args.num_class = len(args.classes)
    else:
        args.num_class = 10

    train(args)
