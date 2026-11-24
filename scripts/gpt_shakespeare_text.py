from tqdm import tqdm
import argparse
import os
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb as wb


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class ShakespeareDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, i):
        x = self.tokens[i:i + self.block_size]
        y = self.tokens[i + 1:i + self.block_size + 1]
        return x, y


def get_loaders(data_path, block_size, batch_size, train_split=0.9):
    with open(data_path, 'r') as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    tokens = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(tokens) * train_split)
    train_loader = DataLoader(ShakespeareDataset(tokens[:n], block_size), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ShakespeareDataset(tokens[n:], block_size), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, vocab_size


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx=None, inputs_embeds=None, targets=None):
        if inputs_embeds is not None:
            x = self.drop(inputs_embeds)
        else:
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class EmbeddingPGD:
    def __init__(self, model, eps, alpha, steps, device):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device

    def attack(self, idx, targets):
        B, T = idx.shape
        pos = torch.arange(T, device=self.device)
        with torch.no_grad():
            clean_tok_embeds = self.model.tok_emb(idx)
            pos_embeds = self.model.pos_emb(pos)
        adv = clean_tok_embeds.clone().detach()
        adv = adv + torch.zeros_like(adv).uniform_(-self.eps, self.eps)
        for _ in range(self.steps):
            adv.requires_grad_(True)
            _, loss = self.model(inputs_embeds=adv + pos_embeds, targets=targets)
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + self.alpha * grad.sign()
            adv = torch.clamp(adv, clean_tok_embeds - self.eps, clean_tok_embeds + self.eps)
            adv = adv.detach()
        return adv + pos_embeds


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = total = loss_sum = nbatch = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)
        loss_sum += loss.item()
        nbatch += 1
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.numel()
    return correct / total, loss_sum / nbatch


def evaluate_adv(model, loader, eps, alpha, steps, device, max_batches=None):
    model.eval()
    attacker = EmbeddingPGD(model, eps=eps, alpha=alpha, steps=steps, device=device)
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        adv_embeds = attacker.attack(x, y)
        with torch.no_grad():
            logits, _ = model(inputs_embeds=adv_embeds, targets=y)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.numel()
    return correct / total


def save_model(checkpoint_dir, exp_id, model, optimizer, step):
    path = f"{checkpoint_dir}/checkpoints/{exp_id}"
    os.makedirs(path, exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{path}/checkpoint_{step}.pth")


def train(config):
    wb.init(project="gpt_shakespeare", name=config.exp_id, config=vars(config), mode="disabled")
    set_seed(config.seed)

    train_loader, test_loader, vocab_size = get_loaders(config.data_path, config.block_size, config.batch_size)
    print(f"vocab_size={vocab_size}")

    class GPTConfig:
        pass
    gpt_config = GPTConfig()
    gpt_config.vocab_size = vocab_size
    gpt_config.block_size = config.block_size
    gpt_config.n_embd = config.n_embd
    gpt_config.n_head = config.n_head
    gpt_config.n_layer = config.n_layer
    gpt_config.dropout = config.dropout
    gpt_config.bias = config.bias

    model = GPT(gpt_config).to(config.device)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.99))

    train_step = 0
    if config.resume_checkpoint is not None and os.path.isfile(config.resume_checkpoint):
        ckpt = torch.load(config.resume_checkpoint, map_location=config.device)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        train_step = ckpt['step']
        print(f"resumed from step {train_step}")

    log_steps = np.unique(np.logspace(0, np.log10(config.num_steps), 90).astype(int).clip(1, config.num_steps)).tolist()
    max_batches = 5

    while train_step < config.num_steps:
        model.train()
        for x, y in tqdm(train_loader, desc=f"step {train_step}"):
            x, y = x.to(config.device), y.to(config.device)
            opt.zero_grad()
            _, loss = model(x, targets=y)
            loss.backward()
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()
            train_step += 1

            if train_step in log_steps:
                train_acc, train_loss = evaluate(model, train_loader, config.device, max_batches)
                test_acc, test_loss = evaluate(model, test_loader, config.device, max_batches)
                stats = {
                    'step': train_step,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                }
                for eps in config.adv_eps:
                    stats[f'test_adv_acc_{eps:.2f}'] = evaluate_adv(model, test_loader, eps, config.adv_alpha, config.adv_steps, config.device, max_batches)
                wb.log(stats)
                save_model(config.checkpoint_dir, config.exp_id, model, opt, train_step)
                model.train()

            if train_step >= config.num_steps:
                break

    wb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='none')
    parser.add_argument('--data_path', type=str, default='data/shakespeare.txt')
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='.')
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--adv_eps', type=float, nargs='+', default=[0.03, 0.06, 0.1, 0.13, 0.16, 0.2])
    parser.add_argument('--adv_alpha', type=float, default=0.01)
    parser.add_argument('--adv_steps', type=int, default=10)
    args = parser.parse_args()
    train(args)
