import argparse
import fcntl
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.datasets import CIFAR10, Imagenette, MNIST
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from models.fcn import FCN
from scripts.resnet18.attacks import APGD, PGD, Square
from scripts.resnet18.models import make_resnet18k


def main(args):
    torch.set_num_threads(1)
    device = torch.device(args.device)
    is_mlp = args.model == 'mlp'
    is_cifar_mlp = args.model == 'mlp_cifar10'
    is_vit = args.model == 'vit'
    is_vit_tiny = args.model == 'vit_tiny'
    mean, std = ((.1307,), (.3081,)) if is_mlp else (
        (.4914, .4822, .4465), (.2023, .1994, .2010))
    if is_vit:
        from scripts.vit_imagenette import IMAGENET_MEAN, IMAGENET_STD, make_model

        mean, std = IMAGENET_MEAN, IMAGENET_STD
        test = Imagenette(args.data_dir, split='val', size='full', download=False,
                          transform=Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize(mean, std)]))
    else:
        test = (MNIST if is_mlp else CIFAR10)(
            args.data_dir, train=False, transform=Compose([ToTensor(), Normalize(mean, std)]))
    indices = np.random.default_rng(0).permutation(len(test))[:args.samples]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test, indices.tolist()),
                                         batch_size=args.batch_size, num_workers=0)
    low = torch.tensor([-m/s for m, s in zip(mean, std)], device=device)[None, :, None, None]
    high = torch.tensor([(1-m)/s for m, s in zip(mean, std)], device=device)[None, :, None, None]

    def clip(x):
        return torch.maximum(low, torch.minimum(x, high))

    def step(p):
        return int(p.stem.split('_')[-1])

    paths = sorted(Path(args.checkpoints).glob('checkpoint_*.pth'), key=step)
    if not paths:
        raise ValueError('No checkpoints found')
    if args.steps:
        by_step = {step(path): path for path in paths}
        paths = [by_step[value] for value in args.steps]
    if args.max_checkpoints and len(paths) > args.max_checkpoints:
        positions = np.rint(np.linspace(0, len(paths)-1, args.max_checkpoints)).astype(int)
        paths = [paths[i] for i in positions]
    args.output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = args.output / f'step_{step(path)}.npz'
        if target.exists():
            continue
        state = torch.load(path, map_location='cpu')['model_state_dict']
        if is_vit:
            net = make_model(SimpleNamespace(dataset='imagenette', model_name='vit_small_patch16_224', num_class=10))
        elif is_vit_tiny:
            from scripts.vit_imagenette import make_model
            net = make_model(SimpleNamespace(dataset='cifar10', model_name='vit_tiny_patch16_224', num_class=10))
        elif is_mlp or is_cifar_mlp:
            net = FCN(depth=args.mlp_depth,
                  input_dim=3072 if is_cifar_mlp else 784, output_dim=10, hidden_dim=args.mlp_width,
                  bias=True, act_fn='relu', norm_type=None, residual_stream=True,
                  init_scale=1.0,
                  block_type=args.mlp_block_type)
        else:
            net = make_resnet18k(k=state['conv1.weight'].shape[0],
                                bn=any('running_mean' in key for key in state))
        net.load_state_dict(state)
        net.to(device).eval().requires_grad_(False)

        class InputModel(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, images):
                images = clip(images)
                return self.net(images.flatten(1) if is_mlp or is_cifar_mlp else images)

        model = InputModel(net).eval()
        options = dict(eps=args.eps, dmin=low.min().item(), dmax=high.max().item())
        attacks = {
            'pgd_ce': PGD(model, **options, steps=10, alpha=.01, device=device),
            'apgd_ce': APGD(model, **options, norm='Linf', steps=10, loss='ce', device=device),
            'apgd_dlr': APGD(model, **options, norm='Linf', steps=10, loss='dlr', device=device),
            'square': Square(model, **options, norm='Linf', n_queries=200, n_restarts=1),
        }
        if args.autoattack:
            from torchattacks import AutoAttack

            # A scalar change of coordinates preserves the normalized Linf budget.
            offset = options['dmin']
            span = options['dmax'] - offset
            if args.pixel_eps:
                channel_span = high - low
                offset, span = low, channel_span
                encoded_eps = args.eps
            else:
                encoded_eps = args.eps / span

            class UnitBoxModel(torch.nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.base = base

                def forward(self, x):
                    return self.base(x * span + offset)

            aa = AutoAttack(UnitBoxModel(model).eval(), norm='Linf', eps=encoded_eps,
                            version='standard', n_classes=10, seed=0, verbose=False)
            restarts = getattr(args, 'restarts', 1)
            if args.aa_steps is not None or args.aa_square_queries is not None or restarts != 1:
                stages = aa._autoattack.attacks
                if [type(stage).__name__ for stage in stages] != ['APGD', 'APGDT', 'FAB', 'Square']:
                    raise RuntimeError('Unexpected AutoAttack stage configuration')
                for stage in stages[:3]:
                    stage.n_restarts = restarts
                stages[3].n_restarts = restarts
                if args.aa_steps is not None:
                    for stage in stages[:3]:
                        stage.steps = args.aa_steps
                if args.aa_square_queries is not None:
                    stages[3].n_queries = args.aa_square_queries
                print(f'{args.model} step {step(path)}: '
                      f'APGD/APGDT/FAB steps={[stage.steps for stage in stages[:3]]}, '
                      f'Square queries={stages[3].n_queries}', flush=True)

            def run_autoattack(images, labels):
                encoded = ((images - offset) / span).clamp(0, 1)
                return aa(encoded, labels) * span + offset

            attacks = {'autoattack': run_autoattack}
        keys = ['clean', *attacks] + ([] if args.autoattack else ['union'])
        masks = {name: [] for name in keys}
        for batch, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                clean = model(x).argmax(1).eq(y)
            masks['clean'].append(clean.cpu().numpy())
            alive = clean.clone()
            for name, attack in attacks.items():
                survived = clean.clone()
                if clean.any():
                    torch.manual_seed(batch)
                    adv = clip(attack(x[clean], y[clean]).detach())
                    delta = (adv-x[clean]).abs()
                    if args.pixel_eps:
                        delta = delta / (high - low)
                    assert delta.max() <= args.eps+1e-5
                    with torch.no_grad():
                        survived[clean] = model(adv).argmax(1).eq(y[clean])
                masks[name].append(survived.cpu().numpy())
                alive &= survived
            if not args.autoattack:
                masks['union'].append(alive.cpu().numpy())
        arrays = {name: np.concatenate(values) for name, values in masks.items()}
        temporary = target.with_suffix('.tmp.npz')
        np.savez(temporary, step=step(path), eps=args.eps, indices=indices,
                 eps_space='pixel' if args.pixel_eps else 'normalized',
                 aa_square_queries=stages[3].n_queries if args.autoattack and
                 (args.aa_steps is not None or args.aa_square_queries is not None) else 0,
                 aa_steps=args.aa_steps if args.aa_steps is not None else 0,
                 aa_restarts=getattr(args, 'restarts', 1), **arrays)
        temporary.replace(target)
        print(f'{args.model} {step(path)}: ' + ' '.join(
            f'{k}={100*v.mean():.2f}%' for k, v in arrays.items()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['mlp', 'mlp_cifar10', 'resnet', 'vit', 'vit_tiny'])
    parser.add_argument('checkpoints')
    parser.add_argument('output', type=Path)
    parser.add_argument('--eps', type=float, default=.0025)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--mlp-depth', type=int, default=8)
    parser.add_argument('--mlp-width', type=int, default=200)
    parser.add_argument('--mlp-block-type', choices=['pre', 'pre_act', 'post'], default='pre')
    parser.add_argument('--max-checkpoints', type=int)
    parser.add_argument('--steps', type=int, nargs='+')
    parser.add_argument('--autoattack', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--aa-steps', type=int, default=10)
    parser.add_argument('--aa-square-queries', type=int, default=2000)
    parser.add_argument('--restarts', type=int, default=1,
                        help='Number of runs per gradient attack (per target for targeted attacks).')
    parser.add_argument('--pixel-eps', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.restarts < 1:
        parser.error('--restarts must be positive')
    if args.aa_steps < 1 or args.aa_square_queries < 1:
        parser.error('Attack counts must be positive')
    if args.pixel_eps and not args.autoattack:
        parser.error('--pixel-eps requires --autoattack')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.eval.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main(args)

