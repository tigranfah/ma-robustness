import torch
import torch.nn.functional as F
from torch.func import vmap, jacrev


def compute_jacobian_rank(model, dataloader, device, max_batches=None, power_iters=5):
    model.eval()
    total_rank = 0.0
    total = 0
    for i, (x, _) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        Js = vmap(jacrev(lambda inp: model(inp.unsqueeze(0)).squeeze(0)))(x)
        B, K = Js.shape[0], Js.shape[1]
        Js = Js.view(B, K, -1)
        F_norm_sq = (Js ** 2).sum(dim=(-2, -1))
        B = Js.shape[0]
        v = torch.randn(B, Js.shape[2], device=device)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        with torch.no_grad():
            for _ in range(power_iters):
                u = torch.bmm(Js.detach(), v.unsqueeze(-1)).squeeze(-1)
                v = torch.bmm(Js.detach().transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1)
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        u = torch.bmm(Js, v.detach().unsqueeze(-1)).squeeze(-1)
        spec_norm_sq = (u ** 2).sum(dim=-1)
        total_rank += (F_norm_sq / (spec_norm_sq + 1e-8)).sum().item()
        total += B
    return total_rank / total


def compute_jacobian_alignment(model, dataloader, device, max_batches=None):
    model.eval()
    total_alignment = 0.0
    total = 0
    for i, (x, _) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        x.requires_grad_(True)
        out = model(x)
        B, K = out.shape
        x_flat = x.view(B, -1)
        x_unit = x_flat / x_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        row_alignments = torch.zeros(B, K, device=device)
        for k in range(K):
            grad = torch.autograd.grad(out[:, k].sum(), x, retain_graph=True, create_graph=False)[0]
            g_flat = grad.view(B, -1)
            g_norm = g_flat.norm(dim=-1).clamp(min=1e-8)
            row_alignments[:, k] = (g_flat * x_unit).sum(dim=-1) / g_norm
        total_alignment += row_alignments.abs().mean(dim=-1).sum().item()
        total += B
    return total_alignment / total


def compute_jacobian_rank_gpt(model, dataloader, device, max_batches=None, power_iters=10):
    model.eval()
    total_rank = 0.0
    total = 0
    for i, (x, _) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        with torch.no_grad():
            clean_embeds = model.tok_emb(x)
        B, T, D = clean_embeds.shape

        def forward(embeds):
            pos = torch.arange(T, device=device)
            h = model.drop(embeds + model.pos_emb(pos))
            for block in model.blocks:
                h = block(h)
            h = model.ln_f(h)
            return model.head(h)[:, -1, :]

        embeds = clean_embeds.detach().requires_grad_(True)
        out = forward(embeds)
        K = out.shape[1]

        frob_sq = torch.zeros(B, device=device)
        JT = []
        for k in range(K):
            g = torch.autograd.grad(out[:, k].sum(), embeds, retain_graph=True, create_graph=False)[0]
            g_flat = g.view(B, -1)
            frob_sq += g_flat.norm(dim=-1) ** 2
            JT.append(g_flat.detach())
        JT = torch.stack(JT, dim=1)

        v = torch.randn(B, T * D, device=device)
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        for _ in range(power_iters):
            u = torch.bmm(JT, v.unsqueeze(-1)).squeeze(-1)
            v = torch.bmm(JT.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1)
            v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        u = torch.bmm(JT, v.unsqueeze(-1)).squeeze(-1)
        spec_norm_sq = u.norm(dim=-1) ** 2

        rank = frob_sq / spec_norm_sq.clamp(min=1e-8)
        total_rank += rank.sum().item()
        total += B
    return total_rank / total


def compute_jacobian_alignment_gpt(model, dataloader, device, max_batches=None):
    model.eval()
    total_alignment = 0.0
    total = 0
    for i, (x, _) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        with torch.no_grad():
            clean_embeds = model.tok_emb(x)
        embeds = clean_embeds.detach().requires_grad_(True)  # [B, T, D]
        B, T, D = embeds.shape
        pos = torch.arange(T, device=device)
        h = model.drop(embeds + model.pos_emb(pos))
        for block in model.blocks:
            h = block(h)
        h = model.ln_f(h)
        out = model.head(h)[:, -1, :]  # [B, vocab_size]
        K = out.shape[1]
        x_flat = embeds.detach().view(B, -1)
        x_unit = x_flat / x_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        row_alignments = torch.zeros(B, K, device=device)
        for k in range(K):
            grad = torch.autograd.grad(out[:, k].sum(), embeds, retain_graph=True, create_graph=False)[0]
            g_flat = grad.view(B, -1)
            g_norm = g_flat.norm(dim=-1).clamp(min=1e-8)
            row_alignments[:, k] = (g_flat * x_unit).sum(dim=-1) / g_norm
        total_alignment += row_alignments.abs().mean(dim=-1).sum().item()
        total += B
    return total_alignment / total


def compute_alignment(model, dataloader, device, max_batches=None):
    model.eval()
    total_alignment = 0.0
    total = 0
    for i, (x, _) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        x.requires_grad_(True)
        out = model(x)
        B, K = out.shape
        c_i = out.argmax(dim=1)
        x_flat = x.detach().view(B, -1)

        grad_i = torch.autograd.grad(out[torch.arange(B, device=device), c_i].sum(), x, retain_graph=True)[0]
        grad_i_flat = grad_i.view(B, -1)

        # compute (J_i - J_j)x / ||J_i - J_j|| for all j != c_i
        margins = torch.full((B, K), float('inf'), device=device)
        for k in range(K):
            is_ci = (c_i == k)
            if is_ci.all():
                continue
            mask = ~is_ci
            grad_k = torch.autograd.grad(out[mask, k].sum(), x, retain_graph=True)[0]
            grad_k_flat = grad_k.view(B, -1)[mask]
            w_diff = grad_i_flat[mask] - grad_k_flat
            w_diff_norm = w_diff.norm(dim=-1).clamp(min=1e-8)
            margin = (w_diff * x_flat[mask]).sum(dim=-1) / w_diff_norm
            margins[mask, k] = margin

        margins[torch.arange(B, device=device), c_i] = float('inf')
        c_j = margins.argmin(dim=1)

        grad_j = torch.autograd.grad(out[torch.arange(B, device=device), c_j].sum(), x)[0]
        w_diff_final = (grad_i - grad_j).view(B, -1)
        cos_sim = F.cosine_similarity(w_diff_final, x_flat, dim=-1).abs()

        total_alignment += cos_sim.sum().item()
        total += B
    return total_alignment / total