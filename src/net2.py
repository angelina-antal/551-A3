import argparse
import copy
import csv
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"device: {device}")

from src.folds import make_stratified_folds
from src.augment_edge import CenteredAugConfig, CenteredOddOneOutAugment
from src.preprocess_edges import (
    PreprocessConfig,
    apply_normalization,
    compute_train_stats,
    preprocess_dataset,
)


# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def maybe_to_tensor(x: np.ndarray) -> torch.Tensor:
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    if x.dtype == np.int32:
        x = x.astype(np.int64)
    return torch.from_numpy(x)


def fold_save_path(path: str, fold_id: int, multi_fold: bool) -> str:
    if not multi_fold:
        return path
    root, ext = os.path.splitext(path)
    if not ext:
        ext = ".pt"
    return f"{root}_fold{fold_id}{ext}"


# -----------------------------
# Datasets
# -----------------------------

class GroupDataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> None:
        self.centered_raw = arrays["centered_raw"].astype(np.float32)   # [N,5,H,W]
        self.edge_stack = arrays["edge_stack"].astype(np.float32)       # [N,5,3,H,W]
        self.meta = arrays["meta"].astype(np.float32)                   # [N,5,M]
        self.labels = None if labels is None else labels.astype(np.int64)

        assert self.centered_raw.ndim == 4, f"Expected [N,5,H,W], got {self.centered_raw.shape}"
        assert self.edge_stack.ndim == 5, f"Expected [N,5,3,H,W], got {self.edge_stack.shape}"
        assert self.meta.ndim == 3, f"Expected [N,5,M], got {self.meta.shape}"

        assert self.centered_raw.shape[:2] == self.meta.shape[:2]
        assert self.centered_raw.shape[:2] == self.edge_stack.shape[:2]
        assert self.edge_stack.shape[2] == 3
        assert self.centered_raw.shape[-2:] == self.edge_stack.shape[-2:]

    def __len__(self) -> int:
        return self.centered_raw.shape[0]

    def __getitem__(self, idx: int):
        sample = {
            "centered_raw": maybe_to_tensor(self.centered_raw[idx]),
            "edge_stack": maybe_to_tensor(self.edge_stack[idx]),
            "meta": maybe_to_tensor(self.meta[idx]),
        }
        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample


class AugmentedTrainDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: CenteredOddOneOutAugment):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        imgs = self.x[idx]
        label = int(self.y[idx])
        imgs, edge_stack, meta, label = self.augment(imgs, label)

        return {
            "centered_raw": torch.from_numpy(imgs),
            "edge_stack": torch.from_numpy(edge_stack),
            "meta": torch.from_numpy(meta),
            "label": torch.tensor(label, dtype=torch.long),
        }


# -----------------------------
# Model
# -----------------------------

class StemBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw(y)
        y = self.bn(y)

        s = x if self.skip is None else self.skip(x)
        return self.act(y + s)


class PrecomputedEdgeTokenEncoder(nn.Module):
    """
    Input: [B, 4, H, W] where channels are:
      raw, sobel_x, sobel_y, laplacian

    Output:
      token feature map [B, C, Ht, Wt]
    """
    def __init__(self, out_ch: int = 32):
        super().__init__()
        self.stem = StemBlock(4, 12)
        self.blocks = nn.Sequential(
            DSResBlock(12, 16, stride=1),
            DSResBlock(16, 24, stride=2),
            DSResBlock(24, 24, stride=1),
            DSResBlock(24, out_ch, stride=1),
            DSResBlock(out_ch, out_ch, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return x


class SetAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop1(y)

        y = self.norm2(x)
        y = self.ff(y)
        x = x + self.drop2(y)
        return x


class CrossAttentionPool(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_slots: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, kv_tokens: torch.Tensor) -> torch.Tensor:
        b = kv_tokens.size(0)
        q = self.slot_queries.unsqueeze(0).expand(b, -1, -1)
        q = self.norm_q(q)
        kv = self.norm_kv(kv_tokens)
        slots, _ = self.attn(q, kv, kv, need_weights=False)
        return self.out_norm(slots)


class EdgeAwareLeaveOneOutTokenNet(nn.Module):
    def __init__(
        self,
        meta_dim: int,
        cnn_dim: int = 32,
        meta_out: int = 8,
        d_model: int = 48,
        rel_dim: int = 96,
        score_hidden: int = 64,
        dropout: float = 0.08,
        set_heads: int = 4,
        set_layers: int = 2,
        pattern_slots: int = 4,
        token_topk: int = 4,
    ) -> None:
        super().__init__()
        if d_model % set_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by set_heads ({set_heads})."
            )
        if pattern_slots < 1:
            raise ValueError("pattern_slots must be >= 1.")
        if token_topk < 1:
            raise ValueError("token_topk must be >= 1.")

        self.token_encoder = PrecomputedEdgeTokenEncoder(out_ch=cnn_dim)
        self.token_proj = nn.Conv2d(cnn_dim, d_model, kernel_size=1, bias=False)
        self.token_norm = nn.LayerNorm(d_model)

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, meta_out),
            nn.GELU(),
            nn.LayerNorm(meta_out),
        )
        self.meta_to_token = nn.Linear(meta_out, d_model)

        ff_hidden = max(2 * d_model, rel_dim)
        self.set_blocks = nn.ModuleList(
            [
                SetAttentionBlock(
                    d_model=d_model,
                    num_heads=set_heads,
                    ff_hidden=ff_hidden,
                    dropout=dropout,
                )
                for _ in range(set_layers)
            ]
        )

        self.pattern_pool = CrossAttentionPool(
            d_model=d_model,
            num_heads=set_heads,
            num_slots=pattern_slots,
            dropout=dropout,
        )
        self.slot_refine = SetAttentionBlock(
            d_model=d_model,
            num_heads=set_heads,
            ff_hidden=ff_hidden,
            dropout=dropout,
        )

        self.pattern_slots = pattern_slots
        self.token_topk = token_topk
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

        feature_dim = 5 * d_model + 6 * pattern_slots
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, score_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden, score_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden, 1),
        )

    def encode_item_tokens(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_raw = batch["centered_raw"]   # [B,5,H,W]
        x_edge = batch["edge_stack"]    # [B,5,3,H,W]
        meta = batch["meta"]            # [B,5,M]

        b, k, h, w = x_raw.shape
        assert x_edge.shape == (b, k, 3, h, w), f"Got edge_stack shape {x_edge.shape}"

        x = torch.cat([x_raw.unsqueeze(2), x_edge], dim=2).contiguous()  # [B,5,4,H,W]
        x = x.view(b * k, 4, h, w)

        fmap = self.token_encoder(x)               # [B*K,C,Ht,Wt]
        fmap = self.token_proj(fmap)               # [B*K,D,Ht,Wt]
        _, d, ht, wt = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)  # [B*K,T,D]
        tokens = tokens.view(b, k, ht * wt, d)    # [B,5,T,D]

        meta_feat = self.meta_mlp(meta)            # [B,5,M2]
        meta_bias = self.meta_to_token(meta_feat).unsqueeze(2)  # [B,5,1,D]
        tokens = self.token_norm(tokens + meta_bias)
        return tokens

    def discover_patterns(self, other_tokens: torch.Tensor) -> torch.Tensor:
        # other_tokens: [B,4,T,D]
        b, g, t, d = other_tokens.shape
        x = other_tokens.reshape(b, g * t, d)
        for block in self.set_blocks:
            x = block(x)
        slots = self.pattern_pool(x)
        slots = self.slot_refine(slots)
        return slots

    def token_support(self, slots: torch.Tensor, img_tokens: torch.Tensor) -> torch.Tensor:
        # slots: [B,S,D]
        # img_tokens: [B,G,T,D]
        s = F.normalize(slots, dim=-1)
        t = F.normalize(img_tokens, dim=-1)
        scale = self.logit_scale.exp().clamp(max=50.0)
        sim = torch.einsum("bsd,bgtd->bgst", s, t) * scale  # [B,G,S,T]
        k = min(self.token_topk, img_tokens.size(2))
        return sim.topk(k=k, dim=-1).values.mean(dim=-1)    # [B,G,S]

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = self.encode_item_tokens(batch)  # [B,5,T,D]
        b, k, t, d = tokens.shape
        assert k == 5, f"Expected K=5 items, got {k}"

        logits = []
        for i in range(k):
            held = tokens[:, i, :, :]  # [B,T,D]
            others = torch.cat([tokens[:, :i, :, :], tokens[:, i + 1 :, :, :]], dim=1)  # [B,4,T,D]

            slots = self.discover_patterns(others)  # [B,S,D]

            other_support = self.token_support(slots, others)              # [B,4,S]
            held_support = self.token_support(slots, held.unsqueeze(1)).squeeze(1)  # [B,S]

            other_mean = other_support.mean(dim=1)                         # [B,S]
            other_min = other_support.min(dim=1).values                    # [B,S]
            other_std = other_support.std(dim=1, unbiased=False)           # [B,S]
            gap_mean = other_mean - held_support                           # [B,S]
            gap_min = other_min - held_support                             # [B,S]
            held_gap = other_support.max(dim=1).values - held_support      # [B,S]

            held_global = held.mean(dim=1)                                 # [B,D]
            others_global = others.mean(dim=(1, 2))                        # [B,D]
            slot_global = slots.mean(dim=1)                                # [B,D]

            score_in = torch.cat(
                [
                    held_global,
                    others_global,
                    slot_global,
                    torch.abs(held_global - slot_global),
                    held_global * slot_global,
                    other_mean,
                    other_min,
                    other_std,
                    held_support,
                    gap_mean,
                    gap_min + held_gap,
                ],
                dim=-1,
            )
            logits.append(self.scorer(score_in).squeeze(-1))

        return torch.stack(logits, dim=1)  # [B,5]


# -----------------------------
# Training / eval
# -----------------------------

def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict_logits_tta(model: nn.Module, batch: Dict[str, torch.Tensor], tta_perms: int = 1) -> torch.Tensor:
    if tta_perms <= 1:
        return model(batch)

    x = batch["centered_raw"]
    e = batch["edge_stack"]
    meta = batch["meta"]

    device = x.device
    b, k = x.shape[:2]
    acc = torch.zeros(b, k, device=device)

    arange_b = torch.arange(b, device=device)[:, None]
    for _ in range(tta_perms):
        perm = torch.stack([torch.randperm(k, device=device) for _ in range(b)], dim=0)
        inv = torch.argsort(perm, dim=1)

        xb = x[arange_b, perm]
        eb = e[arange_b, perm]
        mb = meta[arange_b, perm]

        logits = model({"centered_raw": xb, "edge_stack": eb, "meta": mb})
        logits = logits[arange_b, inv]
        acc += logits

    return acc / float(tta_perms)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, tta_perms: int = 1) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        batch = move_batch(batch, device)
        y = batch["label"]
        logits = predict_logits_tta(model, batch, tta_perms=tta_perms)
        loss = ce(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def predict_logits_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta_perms: int = 4,
) -> np.ndarray:
    model.eval()
    logits_all = []

    for batch in loader:
        batch = move_batch(batch, device)
        logits = predict_logits_tta(model, batch, tta_perms=tta_perms)
        logits_all.append(logits.cpu().numpy())

    return np.concatenate(logits_all, axis=0)


@torch.no_grad()
def collect_per_sample_losses(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="none")
    losses = []

    for batch in loader:
        batch = move_batch(batch, device)
        y = batch["label"]
        logits = model(batch)
        loss = ce(logits, y)
        losses.append(loss.cpu().numpy())

    return np.concatenate(losses, axis=0)


def make_stage2_sampler(
    losses: np.ndarray,
    hard_fraction: float,
    hard_weight: float,
    easy_weight: float = 1.0,
) -> Tuple[WeightedRandomSampler, np.ndarray, float]:
    n = len(losses)
    k = max(1, int(np.ceil(n * hard_fraction)))
    hard_idx = np.argsort(losses)[-k:]

    is_hard = np.zeros(n, dtype=bool)
    is_hard[hard_idx] = True

    weights = np.full(n, easy_weight, dtype=np.float64)
    weights[is_hard] = hard_weight

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=n,
        replacement=True,
    )
    threshold = float(losses[hard_idx].min())
    return sampler, is_hard, threshold


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    hard_mining_frac: float = 1.0,
    accum_steps: int = 8
) -> Tuple[float, float]:
    model.train()
    ce = nn.CrossEntropyLoss(reduction="none")
    total_loss = total_correct = total = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        y = batch["label"]

        logits = model(batch)
        per_sample_loss = ce(logits, y)

        if 0.0 < hard_mining_frac < 1.0:
            k = max(1, int(np.ceil(per_sample_loss.numel() * hard_mining_frac)))
            loss = torch.topk(per_sample_loss, k=k, largest=True).values.mean()
        else:
            loss = per_sample_loss.mean()

        (loss / accum_steps).backward()

        if step % accum_steps == 0:
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += per_sample_loss.mean().item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    # flush last partial accumulation
    if step % accum_steps != 0:
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total, 1), total_correct / max(total, 1)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-train", type=str, default="datasets/x_train.npy")
    parser.add_argument("--x-test", type=str, default="datasets/x_test.npy")
    parser.add_argument("--y-train", type=str, default="datasets/y_train.npy")
    parser.add_argument("--y-test", type=str, default="datasets/y_test.npy")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1.3e-3)
    parser.add_argument("--stage2-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--cnn-dim", type=int, default=16)
    parser.add_argument("--meta-out", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--rel-dim", type=int, default=48)
    parser.add_argument("--score-hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--set-heads", type=int, default=2)
    parser.add_argument("--set-layers", type=int, default=2)
    parser.add_argument("--pattern-slots", type=int, default=4)
    parser.add_argument("--token-topk", type=int, default=1)

    parser.add_argument("--tta-perms-val", type=int, default=2)
    parser.add_argument("--tta-perms-test", type=int, default=4)

    parser.add_argument("--out-size", type=int, default=36)
    parser.add_argument("--inner-size", type=int, default=26)
    parser.add_argument("--crop-pad", type=int, default=2)

    parser.add_argument("--p-permute", type=float, default=1.0)
    parser.add_argument("--p-geom", type=float, default=0.6)
    parser.add_argument("--max-rotate-deg", type=float, default=60.0)
    parser.add_argument("--max-translate-px", type=float, default=0.0)
    parser.add_argument("--max-scale-frac", type=float, default=0.0)
    parser.add_argument("--p-noise", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--p-brightness", type=float, default=0.0)
    parser.add_argument("--brightness-frac", type=float, default=0.0)

    parser.add_argument("--stage2-hard-fraction", type=float, default=0.25)
    parser.add_argument("--stage2-hard-weight", type=float, default=2.0)
    parser.add_argument("--stage2-ohem-frac", type=float, default=0.75)

    parser.add_argument("--save-model", type=str, default="best_model.pt")
    parser.add_argument("--save-csv", type=str, default="predicted_labels.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=-1, help="Run one fold only; -1 means run all folds")
    parser.add_argument("--accum-steps", type=int, default=8)

    args = parser.parse_args()
    seed_everything(args.seed)

    if not (1 <= args.stage1_epochs <= args.epochs):
        raise ValueError("--stage1-epochs must be between 1 and --epochs.")
    if not (0.0 < args.stage2_hard_fraction <= 1.0):
        raise ValueError("--stage2-hard-fraction must be in (0, 1].")
    if not (0.0 < args.stage2_ohem_frac <= 1.0):
        raise ValueError("--stage2-ohem-frac must be in (0, 1].")
    if args.stage2_hard_weight < 1.0:
        raise ValueError("--stage2-hard-weight must be >= 1.0.")
    if args.d_model % args.set_heads != 0:
        raise ValueError("--d-model must be divisible by --set-heads.")
    if args.pattern_slots < 1:
        raise ValueError("--pattern-slots must be >= 1.")
    if args.token_topk < 1:
        raise ValueError("--token-topk must be >= 1.")

    print(f"device: {device}")

    x_train = np.load(args.x_train).astype(np.float32)
    x_test = np.load(args.x_test).astype(np.float32)
    y_train = np.load(args.y_train).astype(np.int64)

    y_test = None
    public_test_n = 0
    if args.y_test and os.path.exists(args.y_test):
        y_test = np.load(args.y_test).astype(np.int64)
        public_test_n = len(y_test)

        if public_test_n > len(x_test):
            raise ValueError(
                f"y_test has length {public_test_n}, but x_test has only {len(x_test)} samples."
            )

    if len(x_train) != len(y_train):
        raise ValueError("Length mismatch between x_train and y_train.")
    if x_train.ndim != 4 or x_train.shape[1] != 5:
        raise ValueError(f"Expected x_train shape [N,5,H,W], got {x_train.shape}")
    if x_test.ndim != 4 or x_test.shape[1] != 5:
        raise ValueError(f"Expected x_test shape [N,5,H,W], got {x_test.shape}")

    folds = make_stratified_folds(
        y_train,
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )
    fold_ids = list(range(args.n_splits)) if args.fold < 0 else [args.fold]

    preprocess_cfg = PreprocessConfig(
        out_size=args.out_size,
        inner_size=args.inner_size,
        crop_pad=args.crop_pad,
    )

    aug_cfg = CenteredAugConfig(
        p_permute=args.p_permute,
        p_geom=args.p_geom,
        max_rotate_deg=args.max_rotate_deg,
        max_translate_px=args.max_translate_px,
        max_scale_frac=args.max_scale_frac,
        p_noise=args.p_noise,
        noise_std=args.noise_std,
        p_brightness=args.p_brightness,
        brightness_frac=args.brightness_frac,
    )
    all_fold_scores = []
    all_fold_test_losses = []
    all_fold_test_accs = []
    test_logits_sum = None
    multi_fold = len(fold_ids) > 1

    for fold_id in fold_ids:
        print(f"\n========== fold {fold_id} ==========")
        tr_idx, va_idx = folds[fold_id]

        x_tr = x_train[tr_idx]
        y_tr = y_train[tr_idx]
        x_va = x_train[va_idx]
        y_va = y_train[va_idx]

        processed_tr_for_stats, meta_names, edge_names = preprocess_dataset(x_tr, preprocess_cfg)
        norm_stats = compute_train_stats(processed_tr_for_stats)
        processed_tr_for_eval = apply_normalization(processed_tr_for_stats, norm_stats)

        processed_val, _, _ = preprocess_dataset(x_va, preprocess_cfg)
        processed_val = apply_normalization(processed_val, norm_stats)

        processed_test, _, _ = preprocess_dataset(x_test, preprocess_cfg)
        processed_test = apply_normalization(processed_test, norm_stats)
        processed_public_test = None
        if y_test is not None:
            processed_public_test = {
                k: v[:public_test_n] for k, v in processed_test.items()
            }

        train_augment = CenteredOddOneOutAugment(
            cfg=aug_cfg,
            preprocess_cfg=preprocess_cfg,
            norm_stats=norm_stats,
        )

        train_ds = AugmentedTrainDataset(x=x_tr, y=y_tr, augment=train_augment)
        train_eval_ds = GroupDataset(arrays=processed_tr_for_eval, labels=y_tr)
        val_ds = GroupDataset(arrays=processed_val, labels=y_va)
        test_ds = GroupDataset(arrays=processed_test, labels=None)

        stage1_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        train_eval_loader = DataLoader(
            train_eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        meta_dim = processed_tr_for_stats["meta"].shape[-1]
        print(f"meta_dim: {meta_dim}")
        print("meta names:", ", ".join(meta_names))
        print("edge names:", ", ".join(edge_names))

        model = EdgeAwareLeaveOneOutTokenNet(
            meta_dim=meta_dim,
            cnn_dim=args.cnn_dim,
            meta_out=args.meta_out,
            d_model=args.d_model,
            rel_dim=args.rel_dim,
            score_hidden=args.score_hidden,
            dropout=args.dropout,
            set_heads=args.set_heads,
            set_layers=args.set_layers,
            pattern_slots=args.pattern_slots,
            token_topk=args.token_topk,
        ).to(device)
        print(f"trainable params: {count_trainable_params(model):,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        current_loader = stage1_loader
        current_hard_frac = 1.0

        best_state = None
        best_val_acc = -1.0
        best_val_loss = float("inf")
        wait = 0

        for epoch in range(1, args.epochs + 1):
            if epoch == args.stage1_epochs + 1 and args.stage1_epochs < args.epochs:
                train_losses = collect_per_sample_losses(model, train_eval_loader, device)
                stage2_sampler, is_hard, threshold = make_stage2_sampler(
                    train_losses,
                    hard_fraction=args.stage2_hard_fraction,
                    hard_weight=args.stage2_hard_weight,
                    easy_weight=1.0,
                )

                current_loader = DataLoader(
                    train_ds,
                    batch_size=args.batch_size,
                    sampler=stage2_sampler,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"),
                )

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.stage2_lr,
                    weight_decay=args.weight_decay,
                )
                current_hard_frac = args.stage2_ohem_frac
                wait = 0

                print(
                    f"stage2: hard={is_hard.sum()}/{len(is_hard)} "
                    f"threshold={threshold:.6f} "
                    f"ohem_frac={current_hard_frac:.2f}"
                )

            tr_loss, tr_acc = train_one_epoch(
                model,
                current_loader,
                optimizer,
                device,
                grad_clip=args.grad_clip,
                hard_mining_frac=current_hard_frac,
                accum_steps=args.accum_steps
            )
            va_loss, va_acc = evaluate(
                model,
                val_loader,
                device,
                tta_perms=args.tta_perms_val,
            )

            print(
                f"epoch {epoch:03d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.4f}"
            )

            better = (
                va_acc > best_val_acc
                or (abs(va_acc - best_val_acc) < 1e-12 and va_loss < best_val_loss)
            )

            if better:
                best_val_acc = va_acc
                best_val_loss = va_loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    print("early stopping")
                    break

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)

        final_val_loss, final_val_acc = evaluate(
            model,
            val_loader,
            device,
            tta_perms=args.tta_perms_val,
        )
        print(f"best fold val loss {final_val_loss:.4f} acc {final_val_acc:.4f}")
        all_fold_scores.append(final_val_acc)

        save_path = fold_save_path(args.save_model, fold_id, multi_fold)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "meta_dim": meta_dim,
                "meta_names": meta_names,
                "edge_names": edge_names,
                "norm_stats": norm_stats,
                "args": vars(args),
            },
            save_path,
        )
        print(f"saved model -> {save_path}")

        if y_test is not None:
            public_test_ds = GroupDataset(arrays=processed_public_test, labels=y_test)
            public_test_loader = DataLoader(
                public_test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            public_loss, public_acc = evaluate(
                model,
                public_test_loader,
                device,
                tta_perms=args.tta_perms_test,
            )
            print(
                f"fold {fold_id} public test (first {public_test_n}) "
                f"loss {public_loss:.4f} acc {public_acc:.4f}"
            )

            all_fold_test_losses.append(public_loss)
            all_fold_test_accs.append(public_acc)

        fold_test_logits = predict_logits_dataset(
            model,
            test_loader,
            device,
            tta_perms=args.tta_perms_test,
        )
        if test_logits_sum is None:
            test_logits_sum = fold_test_logits
        else:
            test_logits_sum += fold_test_logits

    if all_fold_test_losses:
        print("\nCV fold public test losses:", ", ".join(f"{x:.4f}" for x in all_fold_test_losses))
        print(f"CV mean public test loss: {np.mean(all_fold_test_losses):.4f}")
        print("CV fold public test accuracies:", ", ".join(f"{x:.4f}" for x in all_fold_test_accs))
        print(f"CV mean public test acc: {np.mean(all_fold_test_accs):.4f}")

    if test_logits_sum is None:
        raise RuntimeError("No test logits were produced.")

    final_test_logits = test_logits_sum / float(len(fold_ids))
    final_test_pred = final_test_logits.argmax(axis=1)

    os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
    with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Label"])
        for i, pred in enumerate(final_test_pred):
            writer.writerow([i, int(pred)])

    print(f"saved predictions -> {args.save_csv}")


if __name__ == "__main__":
    main()
