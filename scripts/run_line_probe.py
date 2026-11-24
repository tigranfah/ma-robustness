#!/usr/bin/env python3
"""
Linear probes from all layers — MCQ, CLS, and NTP (HF, Gemma-7B friendly)
========================================================================

Downstream tasks:
  Multiple-Choice: ARC-Easy, ARC-Challenge, HellaSwag, MMLU, WinoGrande
  Classification:  SST-2
  NTP (LM):        WikiText-2 (default) or WikiText-103

Backbone is **frozen**. Only probe heads train.

Outputs (CSV, appended):
  <outdir>/results_mc_probe_<dataset>.csv   # accuracy (val/test)
  <outdir>/results_cls_probe_<dataset>.csv  # accuracy (val/test)
  <outdir>/results_lm_probe_<dataset>.csv   # perplexity (val/test)

Run everything:
  python probes_all_layers.py --model google/gemma-7b --run_all
"""

from __future__ import annotations
import argparse, csv, math, os, gc, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, Dataset as HFDataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    DataCollatorForMultipleChoice,
    Trainer,
    TrainingArguments,
)
import random, numpy as np, torch
from resnet18.attacks import EmbeddingPGD
from tqdm import tqdm


def set_seed(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)



# ------------------------- Global speed/memory knobs -------------------------
torch.backends.cuda.matmul.allow_tf32 = True
warnings.filterwarnings("ignore", category=UserWarning)


class MultipleChoiceProbe(nn.Module):
    """Multiple-choice probe: score each option with a scalar from pooled reps."""
    def __init__(self, backbone: AutoModelForCausalLM, layer_idx: int):
        super().__init__()
        self.backbone = backbone
        self.layer_idx = layer_idx
        hidden = backbone.config.hidden_size
        self.norm = nn.LayerNorm(hidden, eps=1e-5, elementwise_affine=False)
        self.head = nn.Linear(hidden, 1)  # scalar per option
        for p in self.backbone.parameters():
            p.requires_grad = False

    # def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
    #     # Expect shapes (B, C, T) from DataCollatorForMultipleChoice
    #     bsz, nchoice, seqlen = input_ids.shape
    #     flat_ids = input_ids.view(bsz * nchoice, seqlen)
    #     flat_mask = None
    #     if attention_mask is not None:
    #         flat_mask = attention_mask.view(bsz * nchoice, seqlen)

    #     out = self.backbone(
    #         input_ids=flat_ids,
    #         attention_mask=flat_mask,
    #         output_hidden_states=True,
    #         use_cache=False,
    #     )
    #     h = out.hidden_states[self.layer_idx + 1]   # (B*C, T, d)
    #     if flat_mask is not None:
    #         mask = flat_mask.unsqueeze(-1)
    #         pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-5)
    #     else:
    #         pooled = h.mean(1)
    #     scores = self.head(self.norm(pooled)).view(bsz, nchoice)  # (B, C)
    #     loss = None
    #     if labels is not None:
    #         loss = nn.functional.cross_entropy(scores, labels)
    #     return {"loss": loss, "logits": scores}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        **kwargs
    ):
        if input_ids is not None:
            bsz, nchoice, seqlen = input_ids.shape
            flat_ids = input_ids.view(bsz * nchoice, seqlen)
            flat_mask = attention_mask.view(bsz * nchoice, seqlen) if attention_mask is not None else None

            outputs = self.backbone(
                input_ids=flat_ids,
                attention_mask=flat_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        elif inputs_embeds is not None:
            # inputs_embeds: [B*C, T, d]
            flat_mask = attention_mask  # already flat

            outputs = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=flat_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            # need to infer B and C later
            bsz = labels.shape[0] if labels is not None else None
            nchoice = inputs_embeds.shape[0] // bsz

        else:
            raise ValueError("Must provide input_ids or inputs_embeds")

        h = outputs.hidden_states[self.layer_idx + 1]  # [B*C, T, d]
        last_hidden = h[:, -1, :]
        
        scores = self.head(self.norm(last_hidden))  # [B*C, 1]

        if input_ids is not None:
            scores = scores.view(bsz, nchoice)
        else:
            scores = scores.view(bsz, nchoice)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(scores, labels)

        return {"loss": loss, "logits": scores}

# ------------------------------- Datasets -----------------------------------

def _pad_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# ---- Language Modeling (NTP) ----

def _tokenise_text_blocks(ds: HFDataset, tokenizer, text_key: str, block_size: int):
    ds_tok = ds.map(
        lambda ex: tokenizer(ex[text_key], add_special_tokens=False),
        remove_columns=[text_key],
        batched=True
    )
    def group(ex):
        ids = sum(ex["input_ids"], [])
        n = len(ids) // block_size * block_size
        ids = ids[:n]
        blocks = [ids[i:i+block_size] for i in range(0, n, block_size)]
        return {"input_ids": blocks, "labels": blocks.copy()}
    return ds_tok.map(group, batched=True, batch_size=1000, remove_columns=ds_tok.column_names)

def load_wikitext(tokenizer, which="wikitext-2-raw-v1", block_size=512):
    wt = load_dataset("wikitext", which)
    return (
        _tokenise_text_blocks(wt["train"], tokenizer, "text", block_size),
        _tokenise_text_blocks(wt["validation"], tokenizer, "text", block_size),
        _tokenise_text_blocks(wt["test"], tokenizer, "text", block_size),
    )

# ---- Multiple Choice helpers ----
def load_arce(tokenizer, max_length=256):
    ds = load_dataset("ai2_arc", "ARC-Easy")
    def map_split(split):
        def to_mc(ex):
            choices = ex["choices"]["text"]
            labels = ex["choices"]["label"]
            # gold answerKey is A/B/C/D...
            gold_letter = ex["answerKey"]
            label_idx = labels.index(gold_letter)
            return {
                "question": ex["question"].strip(),
                "choices": [c.strip() for c in choices],
                "label": label_idx
            }
        return split.map(to_mc, remove_columns=split.column_names)
    return {k: map_split(v) for k, v in ds.items()}, 4

def load_arcc(tokenizer, max_length=256):
    ds = load_dataset("ai2_arc", "ARC-Challenge")
    def map_split(split):
        def to_mc(ex):
            choices = ex["choices"]["text"]
            labels = ex["choices"]["label"]
            gold_letter = ex["answerKey"]
            label_idx = labels.index(gold_letter)
            return {
                "question": ex["question"].strip(),
                "choices": [c.strip() for c in choices],
                "label": label_idx
            }
        return split.map(to_mc, remove_columns=split.column_names)
    return {k: map_split(v) for k, v in ds.items()}, 4

def load_hellaswag_mc(tokenizer, max_length=128):
    hs = load_dataset("hellaswag")
    def prep(split):
        def row(ex):
            ctx = ex.get("ctx") or (ex.get("ctx_a","") + " " + ex.get("ctx_b","")).strip()
            ends = ex["endings"]
            lab = ex.get("label", 0)
            try:
                lab = int(lab)
            except Exception:
                lab = 0
            return {
                "question": ctx,
                "choices": [ctx + " " + e for e in ends],
                "label": lab
            }
        return split.map(row, remove_columns=split.column_names)
    return {k: prep(v) for k, v in hs.items()}, 4

def _default_dtype():
    # Prefer bf16 on Ampere+ (A100); otherwise fp16 if requested
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
    return torch.float16

def _load_backbone(model_name: str, torch_dtype=None, device_map="auto"):
    if torch_dtype is None:
        torch_dtype = _default_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def _trainer_args(out_dir: Path, epochs: int, lr: float, batch_size: int, eff_batch: int, bf16: bool):
    grad_acc = max(1, eff_batch // max(1, batch_size))
    return TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        optim="adamw_torch",
        bf16=bf16,
        fp16=not bf16,
        report_to=[],
    )

def _maybe_subsample(ds: HFDataset, max_samples: int | None):
    if max_samples is None or len(ds) <= max_samples:
        return ds
    return ds.select(range(max_samples))


@torch.no_grad()
def _num_layers(backbone) -> int:
    return getattr(backbone.config, "num_hidden_layers", None) or \
           len(getattr(getattr(backbone, "model", backbone), "layers", [])) or \
           len(getattr(getattr(backbone, "transformer", backbone), "h", []))


def tokenize_mc_dataset(ds_raw, tokenizer, max_length=256):
    def preprocess(example):
        texts = [f"{example['question']} {opt}" for opt in example['choices']]
        enc = tokenizer(texts, truncation=True, padding=False, max_length=max_length)
        return {
            "input_ids": enc["input_ids"],            # list[list[int]] (C×T)
            "attention_mask": enc["attention_mask"],  # list[list[int]]
            "labels": int(example["label"]),
        }

    ds_tok = {}
    for split, split_ds in ds_raw.items():
        tmp = split_ds.map(
            preprocess,
            remove_columns=[c for c in split_ds.column_names if c not in ("question","choices","label")]
        )
        keep = {"input_ids", "attention_mask", "labels"}
        ds_tok[split] = tmp.remove_columns([c for c in tmp.column_names if c not in keep])
    return ds_tok


def _collate_mc_adv(batch, pad_id: int):
    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
    nchoices = [len(ex["input_ids"]) for ex in batch]
    C = nchoices[0]
    B = len(batch)

    flat_ids = []
    flat_masks = []
    for ex in batch:
        for ids, mask in zip(ex["input_ids"], ex["attention_mask"]):
            flat_ids.append(torch.tensor(ids, dtype=torch.long))
            flat_masks.append(torch.tensor(mask, dtype=torch.long))

    # ENFORCE LEFT PADDING: Flip, pad (which pads the "right"), then flip back
    reversed_ids = [t.flip(0) for t in flat_ids]
    reversed_masks = [t.flip(0) for t in flat_masks]
    
    padded_ids = rnn.pad_sequence(reversed_ids, batch_first=True, padding_value=pad_id).flip(1)
    padded_masks = rnn.pad_sequence(reversed_masks, batch_first=True, padding_value=0).flip(1)

    # RESHAPE TO 3D [B, C, T] TO MATCH THE PROBE
    input_ids = padded_ids.view(B, C, -1)
    attention_mask = padded_masks.view(B, C, -1)

    return {
        "input_ids": input_ids,          # [B, C, T]
        "attention_mask": attention_mask,# [B, C, T]
        "labels": labels,                # [B]
        "nchoice": C,
    }


def evaluate_adv(model, dataset, eps, alpha, itrs, device, tokenizer, batch_size):
    model.eval()
    device = torch.device(device)

    atk = EmbeddingPGD(model, eps=eps, alpha=alpha, steps=itrs, device=device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate_mc_adv(batch, pad_id),
    )

    correct = 0
    nsamples = 0

    for batch in tqdm(loader, desc="Adv eval"):
        input_ids = batch["input_ids"].to(device)           # [B, C, T]
        attention_mask = batch["attention_mask"].to(device) # [B, C, T]
        labels = batch["labels"].to(device)                 # [B]
        B = labels.size(0)

        # Generate adversarial embeddings
        adv_embeds = atk.attack(input_ids, attention_mask, labels)  # [B*C, T, d]

        # Flatten mask for the probe
        flat_mask = attention_mask.view(B * batch["nchoice"], -1)

        with torch.no_grad():
            # Pass the adv_embeds into the probe. The probe returns a dictionary.
            outputs = model(
                inputs_embeds=adv_embeds,
                attention_mask=flat_mask,
                labels=labels
            )

            # The probe already did the hidden_states/pooling math! Just grab the scores.
            scores = outputs["logits"]                 # [B, C]
            pred = scores.argmax(dim=-1)               # [B]

        correct += (pred == labels).sum().item()
        nsamples += B

        del input_ids, attention_mask, labels, adv_embeds, outputs, scores, flat_mask
        torch.cuda.empty_cache()

    model.train()
    return correct / max(nsamples, 1)


def run_mc_probe_all_layers(
    backbone, tokenizer, ds: Dict[str, HFDataset], n_choices: int,
    out_dir: Path, batch_size: int, eff_batch: int, epochs: int,
    max_train_samples: int | None, max_eval_samples: int | None,
    result_csv: Path, adv_eps, adv_alpha, adv_itrs
):
    bf16 = (backbone.dtype == torch.bfloat16)
    device = next(backbone.parameters()).device

    def _filter_by_num_choices(ds, n_choices):
        return {k: v.filter(lambda ex: len(ex["choices"]) == n_choices) for k, v in ds.items()}
    
    ds = _filter_by_num_choices(ds,n_choices)
    ds = tokenize_mc_dataset(ds, tokenizer, max_length=1024)
    # collator = DataCollatorForMultipleChoice(tokenizer, padding=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    collator = lambda batch: _collate_mc_adv(batch, pad_id)
    if not result_csv.exists():
        with result_csv.open("w", newline="") as f:
            csv.writer(f).writerow(["layer_idx", "val_acc", "test_acc"])

    # Convert into MC input format (B, C, T) expected by collator
    def to_features(example):
        # already structured by our loaders as question/choices/label
        return example
    ds_feats = {k: v.map(to_features) for k, v in ds.items() if k in ("train", "validation", "test")}

    L = _num_layers(backbone)
    for l in range(L):
        model = MultipleChoiceProbe(backbone, l).to(device)
        args = _trainer_args(out_dir / f"mc_layer{l}", epochs, 5e-4, batch_size, eff_batch, bf16)

        def acc_metric(eval_pred):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
            pred = logits.argmax(-1)
            return {"eval_acc": (pred == labels).mean().item()}

        trainer = Trainer(
            model=model, args=args,
            train_dataset=_maybe_subsample(ds_feats["train"], max_train_samples),
            eval_dataset=_maybe_subsample(ds_feats.get("validation", ds_feats.get("test")), max_eval_samples),
            data_collator=collator,
            compute_metrics=acc_metric,
        )
        trainer.train()

        def eval_acc(dataset):
            ds_eval = _maybe_subsample(dataset, max_eval_samples)
            return trainer.evaluate(eval_dataset=ds_eval)["eval_acc"]

        val_acc = eval_acc(ds_feats.get("validation", ds_feats.get("test")))
        test_acc = eval_acc(ds_feats.get("test", ds_feats.get("validation")))
        adv_acc = {}
        for eps in tqdm(adv_eps):
            adv_acc[eps] = evaluate_adv(
                model=model,
                dataset=_maybe_subsample(ds_feats.get("test", ds_feats.get("validation")), max_eval_samples),
                eps=eps,
                alpha=adv_alpha,
                itrs=adv_itrs,
                device=device,
                tokenizer=tokenizer,
                batch_size=batch_size // 2
            )

        with result_csv.open("a", newline="") as f:
            csv.writer(f).writerow([l, f"{val_acc:.4f}", f"{test_acc:.4f}"] + [f"adv-{eps}-acc={adv_acc[eps]}" for eps in adv_acc])

        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

def load_lambada(tokenizer, which="standard", block_size=512):
    """
    which: "openai" or "standard".
    Returns (train, val, test) tokenized into fixed-size blocks for LM probing.
    """
    ds = load_dataset("cimec/lambada")
    # Reuse validation as a tiny train + val; test is the real evaluation set
    train_ds = _tokenise_text_blocks(ds["validation"], tokenizer, "text", block_size)
    val_ds   = _tokenise_text_blocks(ds["validation"], tokenizer, "text", block_size)
    test_ds  = _tokenise_text_blocks(ds["test"], tokenizer, "text", block_size)
    return train_ds, val_ds, test_ds

def load_ptb(tokenizer, block_size=512):
    """
    Penn Treebank (text only) — splits: train/validation/test.
    Uses the 'sentence' field and packs into fixed-length blocks
    compatible with the LM probe.
    """
    ptb = load_dataset("ptb_text_only")
    train = _tokenise_text_blocks(ptb["train"], tokenizer, text_key="sentence", block_size=block_size)
    val   = _tokenise_text_blocks(ptb["validation"], tokenizer, text_key="sentence", block_size=block_size)
    test  = _tokenise_text_blocks(ptb["test"], tokenizer, text_key="sentence", block_size=block_size)
    return train, val, test


def _filter_labeled(ds_dict):
    out = {}
    for k, v in ds_dict.items():
        out[k] = v.filter(lambda ex: isinstance(ex["label"], int) and ex["label"] >= 0)
    return out

def main():
    set_seed(123)
    p = argparse.ArgumentParser("probes_all_layers.py — linear probes from all layers")
    p.add_argument("--model", default="EleutherAI/pythia-1.4b")
    p.add_argument("--device_map", default="auto")
    # Which tasks
    p.add_argument("--mc_datasets", default="hellaswag") # arc_easy,arc_challenge,hellaswag,mmlu,winogrande
    p.add_argument("--cls_datasets", default="sst2")
    p.add_argument("--lm_datasets", default="wikitext2,wikitext103")  # or "wikitext103"

    # Train settings
    p.add_argument("--epochs_mc", type=int, default=1)
    p.add_argument("--epochs_cls", type=int, default=1)
    p.add_argument("--epochs_lm", type=int, default=1)
    p.add_argument("--batch_mc", type=int, default=16)
    p.add_argument("--batch_cls", type=int, default=32)
    p.add_argument("--batch_lm", type=int, default=16)
    p.add_argument("--eff_batch_mc", type=int, default=64)
    p.add_argument("--eff_batch_cls", type=int, default=64)
    p.add_argument("--eff_batch_lm", type=int, default=64)

    # Subsampling (to control runtime/memory)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=128)

    p.add_argument('--adv_eps', type=float, nargs="+", default=[0.03, 0.06, 0.1, 0.13, 0.16, 0.2])
    p.add_argument('--adv_alpha', type=float, default=1e-2)
    p.add_argument('--adv_itrs', type=int, default=10)

    args = p.parse_args()

    outdir = Path(f"probe_runs/{args.model.split('/')[-1].lower()}")
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    _pad_tokenizer(tokenizer)

    print(f"Loading backbone {args.model} ...")
    backbone = _load_backbone(args.model, torch_dtype=None, device_map=args.device_map)

    
    mc_list  = [x.strip() for x in args.mc_datasets.split(",") if x.strip()]
    cls_list = [x.strip() for x in args.cls_datasets.split(",") if x.strip()]
    lm_list  = [x.strip() for x in args.lm_datasets.split(",") if x.strip()]

    # -------------------- Run Multiple Choice tasks --------------------
    for name in mc_list:
        print(f"\n[MC] {name}")
        if name == "arc_easy":
            ds, nC = load_arce(tokenizer)
        elif name == "arc_challenge":
            ds, nC = load_arcc(tokenizer)
        elif name == "hellaswag":
            ds, nC = load_hellaswag_mc(tokenizer)
            ds = _filter_labeled(ds)
        elif name == "mmlu":
            ds, nC = load_mmlu_mc(tokenizer)
        elif name == "winogrande":
            ds, nC = load_winogrande_mc(tokenizer)
        else:
            print(f"Unknown MC dataset: {name} — skipping")
            continue

        result_csv = outdir / f"results_mc_probe_{name}.csv"
        run_mc_probe_all_layers(
            backbone, tokenizer, ds, nC,
            out_dir=outdir / f"mc_{name}",
            batch_size=args.batch_mc, eff_batch=args.eff_batch_mc, epochs=args.epochs_mc,
            max_train_samples=args.max_train_samples, max_eval_samples=args.max_eval_samples,
            result_csv=result_csv, adv_eps=args.adv_eps, adv_alpha=args.adv_alpha, adv_itrs=args.adv_itrs
        )

    print("\nAll done. CSVs written under:", outdir.resolve())


if __name__ == "__main__":
    main()