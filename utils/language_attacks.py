from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchattacks import FAB

from utils.gpt_compute_metrics import EmbeddingAPGD


class CausalAdapter(nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.model = model
        self.mask = mask

    def tok_emb(self, ids):
        return self.model.get_input_embeddings()(ids)

    def pos_emb(self, positions):
        return 0

    def forward(self, inputs_embeds):
        return self.model(inputs_embeds=inputs_embeds, attention_mask=self.mask,
                          use_cache=False).logits, None


class GPTBackbone(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, inputs_embeds, attention_mask=None, use_cache=False):
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        hidden = self.base.drop(inputs_embeds + self.base.pos_emb(positions))
        for block in self.base.blocks:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=self.base.ln_f(hidden))


class GPTCausalAdapter(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.gpt_neox = GPTBackbone(base)
        self.embed_out = base.head

    def get_input_embeddings(self):
        return self.base.tok_emb

    def forward(self, inputs_embeds, attention_mask=None, use_cache=False):
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        logits, _ = self.base(inputs_embeds=inputs_embeds + self.base.pos_emb(positions))
        return SimpleNamespace(logits=logits)


class MaskedAPGD(EmbeddingAPGD):
    def _loss(self, adv, pos_embeds, targets):
        logits, _ = self.model(inputs_embeds=adv)
        valid = targets.ne(-100)
        if self.loss == "ce":
            values = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        else:
            top, indices = logits.topk(3, dim=-1)
            labels = targets.masked_fill(~valid, 0)
            target = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            other = torch.where(indices[..., 0].eq(labels), top[..., 1], top[..., 0])
            values = (other - target) / (top[..., 0] - top[..., 2]).clamp_min(1e-12)
        return values.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1)


class TokenBoundary(nn.Module):
    def __init__(self, model, clean, mask, position, label, competitor, eps):
        super().__init__()
        self.model = model
        self.register_buffer("clean", clean)
        self.register_buffer("mask", mask)
        self.position = position
        self.label = label
        self.competitor = competitor
        self.eps = eps

    def forward(self, encoded):
        embeds = self.clean + (encoded[:, 0] - 0.5) * (2 * self.eps)
        hidden = self.model.gpt_neox(inputs_embeds=embeds, attention_mask=self.mask,
                                     use_cache=False).last_hidden_state[:, self.position]
        weights = self.model.embed_out.weight[[self.label, self.competitor]]
        return F.linear(hidden, weights)


class BoundaryFAB(FAB):
    def get_diff_logits_grads_batch(self, imgs, labels):
        if not labels.eq(0).all():
            raise ValueError("The binary boundary adapter expects label zero")
        with torch.enable_grad():
            inputs = imgs.detach().requires_grad_(True)
            logits = self.get_logits(inputs)
            margin = logits[:, 1] - logits[:, 0]
            grad = torch.autograd.grad(margin.sum(), inputs)[0]
        difference = torch.stack((torch.full_like(margin, 1e10), margin.detach()), dim=1)
        gradients = torch.stack((torch.zeros_like(grad), grad.detach()), dim=1)
        return difference, gradients


def sample_positions(mask, seed, count=8):
    candidates = mask[0, 1:].bool().nonzero().flatten().cpu()
    generator = torch.Generator().manual_seed(10000 + seed)
    return candidates[torch.randperm(len(candidates), generator=generator)[:count]].sort().values


def evaluate_sequence(model, ids, mask, eps, seed, n_positions=8, attack_steps=10,
                      restarts=1, consistency=False):
    if ids.shape[0] != 1 or eps <= 0 or min(n_positions, attack_steps, restarts) < 1:
        raise ValueError("Use one sequence, a positive budget, and positive attack counts")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("The shared-position attack protocol requires FP32 weights")
    x, targets = ids[:, :-1], ids[:, 1:]
    selected = sample_positions(mask, seed, n_positions)
    valid = torch.zeros_like(targets, dtype=torch.bool)
    valid[0, selected.to(ids.device)] = True
    if not valid.any():
        raise ValueError("Sequence has no valid next-token targets")
    adapter = CausalAdapter(model, mask[:, :-1])
    with torch.no_grad():
        clean_embeds = adapter.tok_emb(x)
        top = adapter(clean_embeds)[0].topk(2, dim=-1).indices
        clean = top[..., 0].eq(targets)
        attack_labels = top[..., 0] if consistency else targets
        competitors = torch.where(top[..., 0].eq(attack_labels), top[..., 1], top[..., 0])
    base = torch.ones_like(clean) if consistency else clean
    masks = {"clean": clean.clone()}
    masked_targets = attack_labels.masked_fill(~valid, -100)
    perturbations = []

    def prediction(candidate):
        if candidate.dtype != torch.float32 or not torch.isfinite(candidate).all():
            raise RuntimeError("Non-finite or non-FP32 adversarial embeddings")
        magnitude = (candidate - clean_embeds).abs().max().item()
        if magnitude > eps + 1e-6:
            raise RuntimeError(f"Perturbation exceeds budget: {magnitude} > {eps}")
        perturbations.append(magnitude)
        with torch.no_grad():
            return adapter(candidate)[0].argmax(-1).eq(attack_labels)

    for name, loss in (("apgd_ce", "ce"), ("apgd_dlr", "dlr")):
        attack = MaskedAPGD(adapter, eps, attack_steps, ids.device, loss=loss,
                            n_restarts=restarts, seed=seed)
        masks[name] = base & prediction(attack.attack(x, masked_targets).detach())
    union = base & masks["apgd_ce"] & masks["apgd_dlr"]
    fab = base.clone()
    fab_positions = []
    # FAB uses the same sampled targets and checks every candidate on their union.
    for position in selected.tolist():
        if not (valid & union)[0, position]:
            continue
        boundary = TokenBoundary(model, clean_embeds, mask[:, :-1], position,
                                 int(attack_labels[0, position]),
                                 int(competitors[0, position]), eps)
        attack = BoundaryFAB(boundary, norm="Linf", eps=0.5, steps=attack_steps,
                             n_restarts=restarts, n_classes=2, seed=seed, verbose=False)
        encoded = torch.full_like(clean_embeds[:, None], 0.5)
        perturbed = attack(encoded, torch.zeros(1, dtype=torch.long, device=ids.device))
        candidate = clean_embeds + (perturbed[:, 0] - 0.5) * (2 * eps)
        survived = prediction(candidate)
        fab &= survived
        union &= survived
        fab_positions.append(position)
    masks["fab"] = fab
    masks["union"] = union
    return {**{key: value[valid].cpu().numpy() for key, value in masks.items()},
            "evaluated_positions": selected.numpy(),
            "fab_positions": np.asarray(fab_positions, dtype=int),
            "max_perturbation": max(perturbations, default=0)}


def evaluate_loader(model, loader, eps, device, is_gpt=False, attack_steps=10,
                    restarts=1, n_positions=8, max_sequences=64, seed=0,
                    consistency=False, max_batches=None):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.eval()
    adapted = GPTCausalAdapter(model) if is_gpt else model
    values = {key: [] for key in ("clean", "apgd_ce", "apgd_dlr", "fab", "union")}
    sequence_index = 0
    for batch_index, (inputs, labels_or_mask) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        for index in range(len(inputs)):
            if max_sequences is not None and sequence_index >= max_sequences:
                break
            if is_gpt:
                ids = torch.cat((inputs[index], labels_or_mask[index, -1:])).unsqueeze(0)
                mask = torch.ones_like(ids)
            else:
                ids = inputs[index:index + 1]
                mask = labels_or_mask[index:index + 1]
            result = evaluate_sequence(adapted, ids.to(device), mask.to(device), eps,
                                       seed + sequence_index, n_positions, attack_steps,
                                       restarts, consistency)
            for key in values:
                values[key].append(result[key])
            sequence_index += 1
        if max_sequences is not None and sequence_index >= max_sequences:
            break
    if not sequence_index:
        raise ValueError("No evaluation sequences")
    return {key: np.concatenate(parts) for key, parts in values.items()}
