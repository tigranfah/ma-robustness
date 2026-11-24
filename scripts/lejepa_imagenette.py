import argparse
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision.ops import MLP
from torchvision.transforms import v2

from scripts.resnet18.attacks import APGD, PGD, Square
from scripts.script_utils import GrokAlign


# class SIGReg(torch.nn.Module):
#     def __init__(self, knots=17):
#         super().__init__()
#         t = torch.linspace(0, 3, knots, dtype=torch.float32)
#         dt = 3 / (knots - 1)
#         weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
#         weights[[0, -1]] = dt
#         window = torch.exp(-t.square() / 2.0)
#         self.register_buffer("t", t)
#         self.register_buffer("phi", window)
#         self.register_buffer("weights", weights * window)

#     def forward(self, proj):
#         A = torch.randn(proj.size(-1), 256, device=proj.device)
#         A = A.div_(A.norm(p=2, dim=0).clamp_min(1e-8))
#         x_t = (proj @ A).unsqueeze(-1) * self.t
#         err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
#         statistic = (err @ self.weights) * proj.size(-2)
#         return statistic.mean()


class SIGReg(torch.nn.Module):
    def __init__(self, sigma=1.0, knots=17):
        super().__init__()
        alpha = 1 / sigma
        t = torch.linspace(0, 3 / (sigma * alpha), knots, dtype=torch.float32)
        dt = (3 / (sigma * alpha)) / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-t.square() * sigma**2 / 2.0)
        window = torch.exp(-t.square() * alpha**2 * sigma**2 / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128, reg_tokens=None):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
            reg_tokens=reg_tokens
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        n, v = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(n, v, -1).transpose(0, 1)


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, split, size=128, V=1, root="./data"):
        self.V = V
        from torchvision import datasets

        tv_split = "train" if split == "train" else "val"
        self.ds = datasets.Imagenette(root=root, split=tv_split, size="160px", download=True)
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(size, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=size)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test = v2.Compose(
            [
                v2.Resize(size),
                v2.CenterCrop(size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        img, label = self.ds[i]
        transform = self.aug if self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), label

    def __len__(self):
        return len(self.ds)


def make_adversarial_view(net, images, args):
    was_training = net.training
    net.eval()

    clean = images.detach()

    # Fixed clean projection target.
    with torch.no_grad():
        clean_emb = net.backbone(clean)
        target = net.proj(clean_emb).detach()

    # random start
    delta = torch.empty_like(clean).uniform_(
        -args.train_adv_eps,
        args.train_adv_eps,
    )
    adv = (clean + delta).clamp(args.adv_min, args.adv_max)

    for _ in range(args.train_adv_itrs):
        adv = adv.detach().requires_grad_(True)

        with autocast("cuda", enabled=False):
            adv_emb = net.backbone(adv.float())
            adv_proj = net.proj(adv_emb)
            attack_loss = F.mse_loss(adv_proj, target.float())

        # Gradient ascent: maximize projection-space distance.
        grad = torch.autograd.grad(attack_loss, adv, only_inputs=True)[0]
        adv = adv.detach() + args.train_adv_alpha * grad.sign()

        # Project back to L-inf eps ball around clean image.
        adv = torch.max(torch.min(adv, clean + args.train_adv_eps), clean - args.train_adv_eps)
        adv = adv.clamp(args.adv_min, args.adv_max)

    net.train(was_training)
    return adv.detach()


def maybe_add_adversarial_view(net, probe, vs, y, args):
    if not args.use_adv_view:
        return vs

    source = vs[:, args.adv_view_source]
    adv = make_adversarial_view(net, source, args)
    return torch.cat([vs, adv[:, None]], dim=1)


# def evaluate_adv_probe(net, probe, loader, eps, alpha, itrs, adv_min, adv_max, device, max_batches=None):
#     net.eval()
#     probe.eval()
#     atk = PGD(
#         nn.Sequential(net.backbone, probe),
#         eps=eps,
#         alpha=alpha,
#         steps=itrs,
#         dmin=adv_min,
#         dmax=adv_max,
#         device=device,
#     )
#     correct = 0
#     total = 0
#     for i, (vs, y) in enumerate(loader):
#         if max_batches is not None and i >= max_batches:
#             break
#         vs = vs.to(device)
#         y = y.to(device)
#         adv_ims = atk(vs[:, 0], y)
#         with torch.no_grad():
#             adv_emb = net.backbone(adv_ims)
#             preds = probe(adv_emb).argmax(1)
#         correct += (preds == y).sum().item()
#         total += y.size(0)
#     return correct / total

def evaluate_adv_probe(net, probe, loader, eps, alpha, steps, n_queries, adv_min, adv_max, device, max_batches=None):
    net.eval()
    probe.eval()

    attack_model = nn.Sequential(net.backbone, probe)
    attack_model.eval()

    pgd_ce = PGD(attack_model, eps=eps, alpha=alpha, steps=steps, dmin=adv_min, dmax=adv_max, device=device)
    apgd_ce = APGD(attack_model, norm="Linf", eps=eps, steps=steps, loss="ce", dmin=adv_min, dmax=adv_max, device=device)
    apgd_dlr = APGD(attack_model, norm="Linf", eps=eps, steps=steps, loss="dlr", dmin=adv_min, dmax=adv_max, device=device)
    square = Square(attack_model, norm="Linf", eps=eps, n_queries=n_queries, n_restarts=1, dmin=adv_min, dmax=adv_max)

    n_rob_pgd_ce = 0
    n_rob_apgd_ce = 0
    n_rob_apgd_dlr = 0
    n_rob_square = 0
    n_rob_union = 0
    nsamples = 0

    for i, (vs, labs) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        ims = vs[:, 0].to(device)
        labs = labs.to(device)
        nsamples += ims.size(0)

        with torch.no_grad():
            correct_mask = attack_model(ims).argmax(1).eq(labs)

        ims_c = ims[correct_mask]
        labs_c = labs[correct_mask]

        if ims_c.size(0) == 0:
            continue

        adv_pgd_ce = pgd_ce(ims_c, labs_c)
        adv_apgd_ce = apgd_ce(ims_c, labs_c)
        adv_apgd_dlr = apgd_dlr(ims_c, labs_c)
        adv_square = square(ims_c, labs_c)

        with torch.no_grad():
            found_pgd_ce = attack_model(adv_pgd_ce).argmax(1).ne(labs_c)
            found_apgd_ce = attack_model(adv_apgd_ce).argmax(1).ne(labs_c)
            found_apgd_dlr = attack_model(adv_apgd_dlr).argmax(1).ne(labs_c)
            found_square = attack_model(adv_square).argmax(1).ne(labs_c)

        found_any = found_pgd_ce | found_apgd_ce | found_apgd_dlr | found_square

        n_rob_pgd_ce += (~found_pgd_ce).sum().item()
        n_rob_apgd_ce += (~found_apgd_ce).sum().item()
        n_rob_apgd_dlr += (~found_apgd_dlr).sum().item()
        n_rob_square += (~found_square).sum().item()
        n_rob_union += (~found_any).sum().item()

    denom = max(nsamples, 1)

    return {
        "pgd_ce": n_rob_pgd_ce / denom,
        "apgd_ce": n_rob_apgd_ce / denom,
        "apgd_dlr": n_rob_apgd_dlr / denom,
        "square": n_rob_square / denom,
        "union": n_rob_union / denom,
    }


def save_model(checkpoint_dir, exp_id, net, probe, opt, epoch):
    os.makedirs(f"{checkpoint_dir}/checkpoints/{exp_id}", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "probe_state_dict": probe.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        },
        f"{checkpoint_dir}/checkpoints/{exp_id}/checkpoint_{epoch}.pth",
    )


def train(args):
    exp_id = args.exp_id
    exp_id += f"-V{args.V}"
    exp_id += f"-Lambda{args.lamb}"
    exp_id += f"-Wd{args.net_weight_decay}"
    exp_id += f"-JacNormLambda{args.jac_norm_lamb}"
    exp_id += f"-Sigma{args.sigma}"
    if args.use_adv_view:
        exp_id += f"-AdvView{args.train_adv_eps}"
    wandb.init(project="LeJEPA", name=exp_id, config=vars(args), mode="disabled")
    torch.manual_seed(args.seed)

    train_ds = HFDataset("train", V=args.V, root=args.data_root)
    test_ds = HFDataset("validation", V=1, root=args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=8)

    net = ViTEncoder(proj_dim=args.proj_dim, reg_tokens=args.reg_tokens).to("cuda")
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).to("cuda")
    sigreg = SIGReg(args.sigma).to("cuda")
    grokalign = GrokAlign(net.backbone, device="cuda") if args.jac_norm_lamb > 0 else None

    g1 = {"params": net.parameters(), "lr": args.lr, "weight_decay": args.net_weight_decay}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": args.probe_weight_decay}
    opt = torch.optim.AdamW([g1, g2])

    warmup_steps = len(train_loader)
    total_steps = len(train_loader) * args.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
    scaler = GradScaler(enabled=True)

    log_epochs = np.unique(np.logspace(0, np.log10(args.epochs), 30).astype(int).clip(1, args.epochs)).tolist()
    max_batches = 4

    for epoch in range(1, args.epochs + 1):
        net.train()
        probe.train()
        for vs, y in tqdm.tqdm(train_loader, total=len(train_loader)):
            vs = vs.to("cuda", non_blocking=True)
            y = y.to("cuda", non_blocking=True)
            vs = maybe_add_adversarial_view(net, probe, vs, y, args)
            effective_v = vs.size(1)

            with autocast("cuda", dtype=torch.bfloat16):
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * args.lamb + inv_loss * (1 - args.lamb)
                y_rep = y.repeat_interleave(effective_v)
                probe_loss = F.cross_entropy(probe(emb.detach()), y_rep)

            jac_norm_loss = None
            if grokalign is not None:
                jac_imgs = vs.flatten(0, 1).detach().float().requires_grad_(True)
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_mem_efficient=False,
                    enable_math=True,
                ):
                    with autocast("cuda", enabled=False):
                        jac_norm_loss = grokalign(jac_imgs)
                lejepa_loss = lejepa_loss + jac_norm_loss * args.jac_norm_lamb

            loss = lejepa_loss + probe_loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            log_dict = {
                "train/probe": probe_loss.item(),
                "train/lejepa": lejepa_loss.item(),
                "train/sigreg": sigreg_loss.item(),
                "train/inv": inv_loss.item(),
            }
            if jac_norm_loss is not None:
                log_dict["train/jacnorm"] = jac_norm_loss.item()
            wandb.log(log_dict)

        if epoch in log_epochs:
            log_dict = {}
            net.eval()
            probe.eval()
            correct = 0
            total = 0
            with torch.inference_mode():
                for i, (vs, y) in enumerate(test_loader):
                    if max_batches is not None and i >= max_batches:
                        break
                    vs = vs.to("cuda", non_blocking=True)
                    y = y.to("cuda", non_blocking=True)
                    with autocast("cuda", dtype=torch.bfloat16):
                        correct += (probe(net(vs)[0]).argmax(1) == y).sum().item()
                    total += y.size(0)

            log_dict["test/acc"] = correct / total
            log_dict["test/epoch"] = epoch
            
            if not args.no_adv_eval:
                for eps in args.adv_eps:
                    adv_acc = evaluate_adv_probe(
                        net,
                        probe,
                        test_loader,
                        eps,
                        args.adv_alpha,
                        args.adv_itrs,
                        args.adv_n_queries,
                        args.adv_min,
                        args.adv_max,
                        args.device,
                        max_batches,
                    )
                    log_dict[f"test/adv_pgd_ce_{eps:.2f}"] = adv_acc["pgd_ce"]
                    log_dict[f"test/adv_apgd_ce_{eps:.2f}"] = adv_acc["apgd_ce"]
                    log_dict[f"test/adv_apgd_dlr_{eps:.2f}"] = adv_acc["apgd_dlr"]
                    log_dict[f"test/adv_square_{eps:.2f}"] = adv_acc["square"]
                    log_dict[f"test/adv_union_{eps:.2f}"] = adv_acc["union"]

            wandb.log(log_dict)
            net.train()
            probe.train()
            save_model(args.checkpoint_dir, exp_id, net, probe, opt, epoch)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default="none")
    parser.add_argument("--V", type=int, default=4)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--net_weight_decay", type=float, default=5e-2)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-7)
    parser.add_argument("--proj_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lamb", type=float, default=0.02)
    parser.add_argument("--jac_norm_lamb", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--checkpoint_dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reg_tokens", type=int, default=0)

    parser.add_argument("--use_adv_view", action="store_true")
    parser.add_argument("--adv_view_source", type=int, default=0)
    parser.add_argument("--train_adv_eps", type=float, default=0.06)
    parser.add_argument("--train_adv_alpha", type=float, default=4 / 255)
    parser.add_argument("--train_adv_itrs", type=int, default=10)

    parser.add_argument("--no_adv_eval", action="store_true")
    parser.add_argument("--adv_eps", type=float, nargs="+", default=[0.03, 0.06, 0.1, 0.13, 0.16, 0.2])
    parser.add_argument("--adv_alpha", type=float, default=4 / 255)
    parser.add_argument("--adv_itrs", type=int, default=10)
    parser.add_argument("--adv_n_queries", type=int, default=200)
    parser.add_argument("--adv_min", type=float, default=-2.1179)
    parser.add_argument("--adv_max", type=float, default=2.6400)

    args = parser.parse_args()
    train(args)
