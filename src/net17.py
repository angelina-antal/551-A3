import argparse
import copy
import csv
import os
import random
from typing import Dict, Optional, Tuple

from src.folds import make_stratified_folds

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.augment import CenteredAugConfig, CenteredOddOneOutAugment
from src.preprocess import (
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


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def fold_checkpoint_path(base_path: str, fold_id: int) -> str:
    root, ext = os.path.splitext(base_path)
    ext = ext if ext else ".pt"
    return f"{root}_fold{fold_id}{ext}"


def cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    a = F.normalize(a, dim=dim, eps=eps)
    b = F.normalize(b, dim=dim, eps=eps)
    return (a * b).sum(dim=dim)


def softmin(x: torch.Tensor, tau: float = 0.35, dim: int = -1) -> torch.Tensor:
    return -tau * torch.logsumexp(-x / tau, dim=dim)

# -----------------------------
# Datasets
# -----------------------------


class GroupDataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> None:
        self.centered_raw = arrays["centered_raw"].astype(np.float32)
        self.meta = arrays["meta"].astype(np.float32)
        self.labels = None if labels is None else labels.astype(np.int64)

        assert self.centered_raw.ndim == 4, f"Expected [N, 5, H, W], got {self.centered_raw.shape}"
        assert self.meta.ndim == 3, f"Expected [N, 5, M], got {self.meta.shape}"
        assert self.centered_raw.shape[:2] == self.meta.shape[:2]
        assert self.meta.shape[-1] == 1, f"Expected only one metadata feature, got {self.meta.shape[-1]}"

    def __len__(self) -> int:
        return self.centered_raw.shape[0]

    def __getitem__(self, idx: int):
        sample = {
            "centered_raw": maybe_to_tensor(self.centered_raw[idx]),
            "meta": maybe_to_tensor(self.meta[idx]),
        }
        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample


class AugmentedTrainDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        augment: CenteredOddOneOutAugment,
    ):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        imgs = self.x[idx]
        label = int(self.y[idx])

        imgs, meta, label = self.augment(imgs, label)

        return {
            "centered_raw": torch.from_numpy(imgs),
            "meta": torch.from_numpy(meta),
            "label": torch.tensor(label, dtype=torch.long),
        }

# -----------------------------
# Model
# -----------------------------


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyGridEncoder(nn.Module):
    """
    Very small encoder that still keeps a little spatial information via 2x2 pooling.
    """

    def __init__(self, in_ch: int, emb_dim: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, 8, stride=1),
            ConvBNAct(8, 10, stride=1),
            ConvBNAct(10, 10, stride=2),
            ConvBNAct(10, 10, stride=1),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.proj = nn.Linear(10 * 3 * 3, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x).flatten(1)
        return self.proj(h)


class SetBlock(nn.Module):
    def __init__(self, d_model: int = 16, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class RuleHead(nn.Module):
    def __init__(
        self,
        d_model: int = 16,
        hidden: int = 16,
        rel_dim: int = 6,
        rel_hidden: int = 12,
        rel_input_dim: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.pair_rel = nn.Sequential(
            nn.Linear(4 * rel_input_dim, rel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rel_hidden, rel_dim),
        )
        self.quartet_rel_score = nn.Sequential(
            nn.Linear(2 * rel_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.cand_score = nn.Sequential(
            nn.Linear(4 * rel_input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.refine = nn.Sequential(
            nn.Linear(3 * d_model + 2 * rel_dim + 9, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )


class MultiRuleQuartetRankNet(nn.Module):
    def __init__(
        self,
        meta_dim: int,
        cnn_dim: int = 8,
        meta_out: int = 4,
        d_model: int = 16,
        num_rules: int = 4,
        set_blocks: int = 2,
        rel_dim: int = 6,
        rel_hidden: int = 12,
        rel_proj_dim: int = 6,
        dropout: float = 0.10,
        tau_softmin: float = 0.35,
        route_temp: float = 0.70,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.tau_softmin = tau_softmin
        self.route_temp = route_temp

        self.raw_encoder = TinyGridEncoder(in_ch=1, emb_dim=cnn_dim)

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, meta_out),
            nn.GELU(),
            nn.LayerNorm(meta_out),
            nn.Linear(meta_out, meta_out),
        )

        self.fuse = nn.Sequential(
            nn.Linear(cnn_dim + meta_out, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.set_blocks = nn.ModuleList(
            [SetBlock(d_model=d_model, heads=4, dropout=dropout) for _ in range(set_blocks)]
        )

        self.rel_proj = nn.Linear(d_model, rel_proj_dim, bias=False)

        hidden = max(12, d_model)
        self.rule_heads = nn.ModuleList(
            [
                RuleHead(
                    d_model=d_model,
                    hidden=hidden,
                    rel_dim=rel_dim,
                    rel_hidden=rel_hidden,
                    rel_input_dim=rel_proj_dim,
                    dropout=dropout,
                )
                for _ in range(num_rules)
            ]
        )
        self.router = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_rules),
        )

    def _pairwise_six(self, x: torch.Tensor) -> torch.Tensor:
        d01 = torch.norm(x[:, 0] - x[:, 1], dim=-1)
        d02 = torch.norm(x[:, 0] - x[:, 2], dim=-1)
        d03 = torch.norm(x[:, 0] - x[:, 3], dim=-1)
        d12 = torch.norm(x[:, 1] - x[:, 2], dim=-1)
        d13 = torch.norm(x[:, 1] - x[:, 3], dim=-1)
        d23 = torch.norm(x[:, 2] - x[:, 3], dim=-1)
        return torch.stack([d01, d02, d03, d12, d13, d23], dim=-1)

    def _pairwise_cat_six(self, x: torch.Tensor) -> torch.Tensor:
        pair_idx = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        feats = []
        for a, b in pair_idx:
            xa = x[:, a, :]
            xb = x[:, b, :]
            feats.append(torch.cat([xa, xb, torch.abs(xa - xb), xa * xb], dim=-1))
        return torch.stack(feats, dim=1)

    def encode_items(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["centered_raw"]   # [B, 5, H, W]
        meta = batch["meta"]        # [B, 5, 1]

        b, k, h, w = x.shape
        assert k == 5, f"Expected K=5 items, got {k}"

        x = x.reshape(b * k, 1, h, w)

        raw_feat = self.raw_encoder(x).reshape(b, k, -1)
        meta_feat = self.meta_mlp(meta)

        h0 = self.fuse(torch.cat([raw_feat, meta_feat], dim=-1))
        for block in self.set_blocks:
            h0 = block(h0)
        return h0

    def forward_with_details(self, batch: Dict[str, torch.Tensor]):
        h = self.encode_items(batch)
        b, k, d = h.shape
        assert k == 5, f"Expected K=5 items, got {k}"

        logits = []
        quartet_qualities = []
        gate_penalties = []
        route_probs_all = []
        gate_vectors_all = []

        for i in range(k):
            mask = [j for j in range(k) if j != i]
            others = h[:, mask, :]
            cand = h[:, i, :]

            mu = others.mean(dim=1)
            var = ((others - mu[:, None, :]) ** 2).mean(dim=1)
            mx = others.max(dim=1).values
            mn = others.min(dim=1).values
            route_in = torch.cat([mu, var, mx, mn], dim=-1)
            route_logits = self.router(route_in)
            route_probs = F.softmax(route_logits / self.route_temp, dim=-1)

            per_rule_final = []
            per_rule_quartet = []
            per_rule_gate = []
            per_rule_gatevec = []

            for head in self.rule_heads:
                gate = torch.sigmoid(head.gate(route_in))
                per_rule_gatevec.append(gate)

                others_g = others * gate[:, None, :]
                proto = others_g.mean(dim=1)

                compat_each = 0.5 * (cosine_sim(others_g, proto[:, None, :], dim=-1) + 1.0)
                compat_soft = softmin(compat_each, tau=self.tau_softmin, dim=1)
                compat_mean = compat_each.mean(dim=1)
                compat_var = compat_each.var(dim=1, unbiased=False)

                pair_d = self._pairwise_six(others_g)
                pair_mean = pair_d.mean(dim=1)
                pair_max = pair_d.max(dim=1).values

                loo_scores = []
                for t in range(4):
                    tri_idx = [j for j in range(4) if j != t]
                    tri = others_g[:, tri_idx, :]
                    tri_proto = tri.mean(dim=1)
                    s_t = 0.5 * (cosine_sim(others_g[:, t, :], tri_proto, dim=-1) + 1.0)
                    loo_scores.append(s_t)
                loo_scores = torch.stack(loo_scores, dim=1)
                triad_soft = softmin(loo_scores, tau=self.tau_softmin, dim=1)
                triad_var = loo_scores.var(dim=1, unbiased=False)

                others_r = self.rel_proj(others_g)
                pair_rel_in = self._pairwise_cat_six(others_r)
                rel_pairs = head.pair_rel(pair_rel_in)
                rel_mean = rel_pairs.mean(dim=1)
                rel_max = rel_pairs.max(dim=1).values
                rel_spread = rel_pairs.var(dim=1, unbiased=False).mean(dim=-1)
                rel_strength = torch.sigmoid(
                    head.quartet_rel_score(torch.cat([rel_mean, rel_max], dim=-1)).squeeze(-1)
                )

                cand_g = cand * gate
                cand_compat = 0.5 * (cosine_sim(cand_g, proto, dim=-1) + 1.0)
                cand_r = self.rel_proj(cand_g)
                proto_r = self.rel_proj(proto)
                cand_rel_in = torch.cat([cand_r, proto_r, torch.abs(cand_r - proto_r), cand_r * proto_r], dim=-1)
                cand_rel_score = torch.sigmoid(head.cand_score(cand_rel_in).squeeze(-1))

                quartet_quality = (
                    0.90 * compat_soft
                    + 0.25 * compat_mean
                    - 0.20 * compat_var
                    - 0.45 * pair_mean
                    - 0.15 * pair_max
                    + 0.70 * triad_soft
                    - 0.20 * triad_var
                    + 0.30 * rel_strength
                    - 0.10 * rel_spread
                )

                refine_feat = torch.cat(
                    [
                        mu,
                        var,
                        gate,
                        rel_mean,
                        rel_max,
                        (compat_soft - cand_compat).unsqueeze(-1),
                        pair_mean.unsqueeze(-1),
                        pair_max.unsqueeze(-1),
                        compat_mean.unsqueeze(-1),
                        triad_soft.unsqueeze(-1),
                        triad_var.unsqueeze(-1),
                        rel_strength.unsqueeze(-1),
                        rel_spread.unsqueeze(-1),
                        cand_rel_score.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                learned = head.refine(refine_feat).squeeze(-1)

                final_rule_score = quartet_quality - 0.20 * cand_compat - 0.20 * cand_rel_score + learned

                per_rule_final.append(final_rule_score)
                per_rule_quartet.append(quartet_quality)
                per_rule_gate.append(gate.mean(dim=-1))

            per_rule_final = torch.stack(per_rule_final, dim=1)
            per_rule_quartet = torch.stack(per_rule_quartet, dim=1)
            per_rule_gate = torch.stack(per_rule_gate, dim=1)
            per_rule_gatevec = torch.stack(per_rule_gatevec, dim=1)

            logits.append((route_probs * per_rule_final).sum(dim=1))
            quartet_qualities.append((route_probs * per_rule_quartet).sum(dim=1))
            gate_penalties.append(per_rule_gate.mean(dim=1))
            route_probs_all.append(route_probs)
            gate_vectors_all.append(per_rule_gatevec)

        return {
            "logits": torch.stack(logits, dim=1),
            "quartet_qualities": torch.stack(quartet_qualities, dim=1),
            "gate_penalty": torch.stack(gate_penalties, dim=1),
            "route_probs": torch.stack(route_probs_all, dim=1),
            "gate_vectors": torch.stack(gate_vectors_all, dim=1),
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_with_details(batch)["logits"]


# -----------------------------
# Training / eval
# -----------------------------


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def compute_train_loss(
    out: Dict[str, torch.Tensor],
    y: torch.Tensor,
    ce_weight: float = 1.0,
    quartet_margin_weight: float = 0.25,
    gate_penalty_weight: float = 0.002,
    route_entropy_weight: float = 0.010,
    route_balance_weight: float = 0.020,
    gate_diversity_weight: float = 0.005,
    margin: float = 0.12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = out["logits"]
    quartet_qualities = out["quartet_qualities"]
    gate_penalty = out["gate_penalty"]
    route_probs = out["route_probs"]
    gate_vectors = out["gate_vectors"]

    ce_per = F.cross_entropy(logits, y, reduction="none")

    b = y.size(0)
    row_idx = torch.arange(b, device=y.device)
    true_q = quartet_qualities[row_idx, y]

    wrong_mask = torch.ones_like(quartet_qualities, dtype=torch.bool)
    wrong_mask[row_idx, y] = False
    best_wrong_q = quartet_qualities.masked_fill(~wrong_mask, float("-inf")).max(dim=1).values

    quartet_margin = F.relu(margin - true_q + best_wrong_q)

    eps = 1e-8
    route_entropy = -(route_probs * (route_probs + eps).log()).sum(dim=-1)
    route_entropy_per = route_entropy.mean(dim=1)

    usage = route_probs.mean(dim=(0, 1))
    target = torch.full_like(usage, 1.0 / usage.numel())
    route_balance = ((usage - target) ** 2).mean()

    g = F.normalize(gate_vectors, dim=-1, eps=1e-6)
    sim = torch.matmul(g, g.transpose(-1, -2))
    r = sim.size(-1)
    offdiag_mask = ~torch.eye(r, device=sim.device, dtype=torch.bool)
    gate_diversity = sim[..., offdiag_mask].pow(2).mean()

    total_per = (
        ce_weight * ce_per
        + quartet_margin_weight * quartet_margin
        + gate_penalty_weight * gate_penalty.mean(dim=1)
        + route_entropy_weight * route_entropy_per
    )
    total_per = total_per + route_balance_weight * route_balance + gate_diversity_weight * gate_diversity
    return total_per, logits


@torch.no_grad()
def predict_logits_tta(model: nn.Module, batch: Dict[str, torch.Tensor], tta_perms: int = 1) -> torch.Tensor:
    if tta_perms <= 1:
        return model(batch)

    x = batch["centered_raw"]
    meta = batch["meta"]

    device = x.device
    b, k = x.shape[:2]
    acc = torch.zeros(b, k, device=device)
    arange_b = torch.arange(b, device=device)[:, None]

    for _ in range(tta_perms):
        perm = torch.stack([torch.randperm(k, device=device) for _ in range(b)], dim=0)
        inv = torch.argsort(perm, dim=1)

        xb = x[arange_b, perm]
        mb = meta[arange_b, perm]

        logits = model({"centered_raw": xb, "meta": mb})
        logits = logits[arange_b, inv]
        acc += logits

    return acc / float(tta_perms)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, tta_perms: int = 1) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        batch = move_batch(batch, device)
        y = batch["label"]
        logits = predict_logits_tta(model, batch, tta_perms=tta_perms)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, tta_perms: int = 8) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = predict_logits_tta(model, batch, tta_perms=tta_perms)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def collect_per_sample_losses(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    route_entropy_weight: float = 0.010,
    route_balance_weight: float = 0.020,
    gate_diversity_weight: float = 0.005,
) -> np.ndarray:
    model.eval()
    losses = []

    for batch in loader:
        batch = move_batch(batch, device)
        y = batch["label"]
        out = model.forward_with_details(batch)
        per_sample_loss, _ = compute_train_loss(
            out,
            y,
            route_entropy_weight=route_entropy_weight,
            route_balance_weight=route_balance_weight,
            gate_diversity_weight=gate_diversity_weight,
        )
        losses.append(per_sample_loss.cpu().numpy())

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


def build_stage2_loader(
    model: nn.Module,
    train_eval_loader: DataLoader,
    train_ds: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    hard_fraction: float,
    hard_weight: float,
    route_entropy_weight: float = 0.010,
    route_balance_weight: float = 0.020,
    gate_diversity_weight: float = 0.005,
) -> Tuple[DataLoader, np.ndarray, float]:
    train_losses = collect_per_sample_losses(
        model,
        train_eval_loader,
        device,
        route_entropy_weight=route_entropy_weight,
        route_balance_weight=route_balance_weight,
        gate_diversity_weight=gate_diversity_weight,
    )
    stage2_sampler, is_hard, hard_threshold = make_stage2_sampler(
        losses=train_losses,
        hard_fraction=hard_fraction,
        hard_weight=hard_weight,
    )
    stage2_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=stage2_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return stage2_loader, is_hard, hard_threshold


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    hard_mining_frac: float = 1.0,
    route_entropy_weight: float = 0.010,
    route_balance_weight: float = 0.020,
    gate_diversity_weight: float = 0.005,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        batch = move_batch(batch, device)
        y = batch["label"]

        optimizer.zero_grad(set_to_none=True)
        out = model.forward_with_details(batch)
        per_sample_loss, logits = compute_train_loss(
            out,
            y,
            route_entropy_weight=route_entropy_weight,
            route_balance_weight=route_balance_weight,
            gate_diversity_weight=gate_diversity_weight,
        )

        if 0.0 < hard_mining_frac < 1.0:
            k = max(1, int(np.ceil(per_sample_loss.numel() * hard_mining_frac)))
            loss = torch.topk(per_sample_loss, k=k, largest=True).values.mean()
        else:
            loss = per_sample_loss.mean()

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += per_sample_loss.mean().item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def evaluate_public_logits(logits: torch.Tensor, y_public: np.ndarray) -> float:
    n_public = len(y_public)
    if n_public == 0:
        return float("nan")
    if n_public > logits.shape[0]:
        raise ValueError(
            f"y_test has {n_public} rows but logits only has {logits.shape[0]} rows."
        )
    public_preds = logits[:n_public].argmax(dim=1).cpu().numpy()
    return float((public_preds == y_public).mean())


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--x-train", type=str, default="datasets/x_train.npy")
    parser.add_argument("--x-test", type=str, default="datasets/x_test.npy")
    parser.add_argument("--y-train", type=str, default="datasets/y_train.npy")
    parser.add_argument(
        "--y-test",
        type=str,
        default="datasets/y_test.npy",
        help="Optional labels for test/public subset evaluation",
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--stage2-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--cnn-dim", type=int, default=10)
    parser.add_argument("--meta-out", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--num-rules", type=int, default=3)
    parser.add_argument("--set-blocks", type=int, default=2)
    parser.add_argument("--rel-dim", type=int, default=8)
    parser.add_argument("--rel-hidden", type=int, default=16)
    parser.add_argument("--rel-proj-dim", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--tau-softmin", type=float, default=0.35)
    parser.add_argument("--route-temp", type=float, default=0.70)

    parser.add_argument("--tta-perms-val", type=int, default=4)
    parser.add_argument("--tta-perms-test", type=int, default=8)

    parser.add_argument("--out-size", type=int, default=32)
    parser.add_argument("--inner-size", type=int, default=26)
    parser.add_argument("--crop-pad", type=int, default=2)

    parser.add_argument("--p-permute", type=float, default=1.0)
    parser.add_argument("--p-geom", type=float, default=0.9)
    parser.add_argument("--max-rotate-deg", type=float, default=180.0)

    parser.add_argument("--route-entropy-weight", type=float, default=0.010)
    parser.add_argument("--route-balance-weight", type=float, default=0.020)
    parser.add_argument("--gate-diversity-weight", type=float, default=0.005)

    parser.add_argument("--stage2-hard-fraction", type=float, default=0.30)
    parser.add_argument("--stage2-hard-weight", type=float, default=3.0)
    parser.add_argument("--stage2-ohem-frac", type=float, default=0.5)
    parser.add_argument("--stage2-remine-every", type=int, default=4)

    parser.add_argument("--save-model", type=str, default="best_model.pt")
    parser.add_argument("--save-csv", type=str, default="predicted_labels.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=-1, help="Run one fold only; -1 means run all folds")

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
    if args.fold >= args.n_splits and args.fold >= 0:
        raise ValueError("--fold must be -1 or an integer in [0, n_splits - 1].")
    if args.set_blocks < 1:
        raise ValueError("--set-blocks must be >= 1.")
    if args.num_rules < 2:
        raise ValueError("--num-rules must be >= 2 for specialization to make sense.")
    if args.rel_dim < 1:
        raise ValueError("--rel-dim must be >= 1.")
    if args.rel_hidden < 1:
        raise ValueError("--rel-hidden must be >= 1.")
    if args.rel_proj_dim < 1:
        raise ValueError("--rel-proj-dim must be >= 1.")
    if args.stage2_remine_every < 0:
        raise ValueError("--stage2-remine-every must be >= 0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    x_train = np.load(args.x_train).astype(np.float32)
    x_test = np.load(args.x_test).astype(np.float32)
    y_train = np.load(args.y_train).astype(np.int64)

    y_public = None
    n_public = 0
    if args.y_test and os.path.exists(args.y_test):
        y_public = np.load(args.y_test).astype(np.int64)
        n_public = len(y_public)
        if n_public > len(x_test):
            raise ValueError(
                f"y_test has {n_public} rows but x_test only has {len(x_test)} rows."
            )
        print(f"found public test labels for first {n_public}/{len(x_test)} test samples")

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

    fold_ids = range(args.n_splits) if args.fold < 0 else [args.fold]

    preprocess_cfg = PreprocessConfig(
        out_size=args.out_size,
        inner_size=args.inner_size,
        crop_pad=args.crop_pad,
    )

    aug_cfg = CenteredAugConfig(
    p_permute=args.p_permute,
    p_geom=args.p_geom,
    max_rotate_deg=args.max_rotate_deg,
)

    all_fold_scores = []
    test_logits_sum = None

    ensure_parent_dir(args.save_model)
    ensure_parent_dir(args.save_csv)

    for fold_id in fold_ids:
        tr_idx, va_idx = folds[fold_id]

        x_tr = x_train[tr_idx]
        y_tr = y_train[tr_idx]
        x_va = x_train[va_idx]
        y_va = y_train[va_idx]

        processed_tr_for_stats, meta_names = preprocess_dataset(x_tr, preprocess_cfg)
        selected_meta_name = meta_names[0]

        norm_stats = compute_train_stats(processed_tr_for_stats)
        processed_tr_for_eval = apply_normalization(processed_tr_for_stats, norm_stats)

        processed_val, _ = preprocess_dataset(x_va, preprocess_cfg)
        processed_val = apply_normalization(processed_val, norm_stats)

        processed_test, _ = preprocess_dataset(x_test, preprocess_cfg)
        processed_test = apply_normalization(processed_test, norm_stats)

        train_augment = CenteredOddOneOutAugment(
            cfg=aug_cfg,
            preprocess_cfg=preprocess_cfg,
            norm_stats=norm_stats,
        )

        train_ds = AugmentedTrainDataset(
            x=x_tr,
            y=y_tr,
            augment=train_augment,
        )
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
        print(f"\n===== fold {fold_id} =====")
        print(f"meta_dim: {meta_dim}")
        print("meta names:", selected_meta_name)

        model = MultiRuleQuartetRankNet(
            meta_dim=meta_dim,
            cnn_dim=args.cnn_dim,
            meta_out=args.meta_out,
            d_model=args.d_model,
            num_rules=args.num_rules,
            set_blocks=args.set_blocks,
            rel_dim=args.rel_dim,
            rel_hidden=args.rel_hidden,
            rel_proj_dim=args.rel_proj_dim,
            dropout=args.dropout,
            tau_softmin=args.tau_softmin,
            route_temp=args.route_temp,
        ).to(device)

        n_params = count_trainable_params(model)
        print(f"trainable params: {n_params:,}")
        if n_params > 25000:
            raise ValueError(
                f"Model has {n_params:,} trainable params, which violates the <25k requirement. "
                f"Reduce --cnn-dim, --meta-out, --d-model, --num-rules, --set-blocks, or --rel-dim."
            )

        best_state = None
        best_val_acc = -1.0
        best_val_loss = float("inf")
        wait = 0

        fold_model_path = fold_checkpoint_path(args.save_model, fold_id)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.stage1_epochs, 1))

        for epoch in range(1, args.stage1_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=stage1_loader,
                optimizer=optimizer,
                device=device,
                grad_clip=args.grad_clip,
                hard_mining_frac=1.0,
                route_entropy_weight=args.route_entropy_weight,
                route_balance_weight=args.route_balance_weight,
                gate_diversity_weight=args.gate_diversity_weight,
            )
            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                tta_perms=args.tta_perms_val,
            )
            scheduler.step()

            print(
                f"[stage1] epoch {epoch:03d} | "
                f"train {train_loss:.4f}/{train_acc:.4f} | "
                f"val {val_loss:.4f}/{val_acc:.4f}"
            )

            improved = (val_acc > best_val_acc + 1e-8) or (
                abs(val_acc - best_val_acc) <= 1e-8 and val_loss < best_val_loss
            )
            if improved:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, fold_model_path)
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    print("early stopping in stage 1")
                    break

        model.load_state_dict(best_state)

        remaining_epochs = args.epochs - args.stage1_epochs
        if remaining_epochs > 0:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.stage2_lr,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(remaining_epochs, 1))
            wait = 0
            stage2_loader = None

            for epoch in range(args.stage1_epochs + 1, args.epochs + 1):
                should_remine = (
                    stage2_loader is None
                    or (
                        args.stage2_remine_every > 0
                        and (epoch - (args.stage1_epochs + 1)) % args.stage2_remine_every == 0
                    )
                )
                if should_remine:
                    stage2_loader, is_hard, hard_threshold = build_stage2_loader(
                        model=model,
                        train_eval_loader=train_eval_loader,
                        train_ds=train_ds,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        device=device,
                        hard_fraction=args.stage2_hard_fraction,
                        hard_weight=args.stage2_hard_weight,
                        route_entropy_weight=args.route_entropy_weight,
                        route_balance_weight=args.route_balance_weight,
                        gate_diversity_weight=args.gate_diversity_weight,
                    )
                    print(
                        f"[stage2] re-mined {is_hard.sum()}/{len(is_hard)} hard samples "
                        f"({100.0 * is_hard.mean():.1f}%) | "
                        f"loss threshold >= {hard_threshold:.4f}"
                    )

                train_loss, train_acc = train_one_epoch(
                    model=model,
                    loader=stage2_loader,
                    optimizer=optimizer,
                    device=device,
                    grad_clip=args.grad_clip,
                    hard_mining_frac=args.stage2_ohem_frac,
                    route_entropy_weight=args.route_entropy_weight,
                    route_balance_weight=args.route_balance_weight,
                    gate_diversity_weight=args.gate_diversity_weight,
                )
                val_loss, val_acc = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    tta_perms=args.tta_perms_val,
                )
                scheduler.step()

                print(
                    f"[stage2] epoch {epoch:03d} | "
                    f"ohem {args.stage2_ohem_frac:.2f} | "
                    f"train {train_loss:.4f}/{train_acc:.4f} | "
                    f"val {val_loss:.4f}/{val_acc:.4f}"
                )

                improved = (val_acc > best_val_acc + 1e-8) or (
                    abs(val_acc - best_val_acc) <= 1e-8 and val_loss < best_val_loss
                )
                if improved:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    torch.save(best_state, fold_model_path)
                    wait = 0
                else:
                    wait += 1
                    if wait >= args.patience:
                        print("early stopping in stage 2")
                        break

        model.load_state_dict(best_state)
        all_fold_scores.append(best_val_acc)

        model.eval()
        fold_logits = []
        with torch.no_grad():
            for batch in test_loader:
                batch = move_batch(batch, device)
                logits = predict_logits_tta(model, batch, tta_perms=args.tta_perms_test)
                fold_logits.append(logits.cpu())

        fold_logits = torch.cat(fold_logits, dim=0)

        if y_public is not None:
            fold_public_acc = evaluate_public_logits(fold_logits, y_public)
            print(
                f"[fold {fold_id}] public test acc on first {n_public}/{len(x_test)} test samples: "
                f"{fold_public_acc:.4f}"
            )

        if test_logits_sum is None:
            test_logits_sum = fold_logits
        else:
            test_logits_sum += fold_logits

        if y_public is not None:
            running_mean_logits = test_logits_sum / float(len(all_fold_scores))
            running_public_acc = evaluate_public_logits(running_mean_logits, y_public)
            print(
                f"[fold {fold_id}] running ensemble public test acc on first {n_public}/{len(x_test)} test samples: "
                f"{running_public_acc:.4f}"
            )

    mean_test_logits = test_logits_sum / float(len(fold_ids))
    preds = mean_test_logits.argmax(dim=1).numpy()

    if y_public is not None:
        public_acc = evaluate_public_logits(mean_test_logits, y_public)
        print(f"final ensemble public test acc: {public_acc:.4f}")

    print(f"mean CV acc: {np.mean(all_fold_scores):.4f}")
    print(f"std CV acc: {np.std(all_fold_scores):.4f}")

    with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Label"])
        for i, p in enumerate(preds):
            writer.writerow([i, int(p)])
    print(f"wrote {args.save_csv}")


if __name__ == "__main__":
    main()