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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from augment_edge import CenteredAugConfig, CenteredOddOneOutAugment
from preprocess_edges import (
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
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: CenteredOddOneOutAugment):
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
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyCNNEncoder(nn.Module):
    def __init__(self, emb_dim: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(1, 12, stride=1),
            ConvBNAct(12, 16, stride=2),
            ConvBNAct(16, 24, stride=1),
            ConvBNAct(24, 24, stride=2),
            ConvBNAct(24, 24, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(24, emb_dim)

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


class RelationOddOneOutNet(nn.Module):
    def __init__(
        self,
        meta_dim: int,
        cnn_dim: int = 16,
        meta_out: int = 8,
        d_model: int = 16,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.cnn = TinyCNNEncoder(emb_dim=cnn_dim)
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, meta_out),
            nn.GELU(),
            nn.LayerNorm(meta_out),
        )
        self.fuse = nn.Sequential(
            nn.Linear(cnn_dim + meta_out, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.set_blocks = nn.Sequential(
            SetBlock(d_model=d_model, heads=4, dropout=dropout),
            SetBlock(d_model=d_model, heads=4, dropout=dropout),
        )
        self.scorer = nn.Sequential(
            nn.Linear(5 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def encode_items(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["centered_raw"]
        meta = batch["meta"]
        b, k, h, w = x.shape
        x = x.view(b * k, 1, h, w)
        img_feat = self.cnn(x).view(b, k, -1)
        meta_feat = self.meta_mlp(meta)
        h0 = self.fuse(torch.cat([img_feat, meta_feat], dim=-1))
        return self.set_blocks(h0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encode_items(batch)
        b, k, d = h.shape
        assert k == 5, f"Expected K=5 items, got {k}"

        sum_all = h.sum(dim=1, keepdim=True)
        mean_others = (sum_all - h) / float(k - 1)

        others_max = []
        for i in range(k):
            mask = [j for j in range(k) if j != i]
            others_max.append(h[:, mask, :].max(dim=1).values)
        max_others = torch.stack(others_max, dim=1)

        feats = torch.cat(
            [
                h,
                mean_others,
                max_others,
                torch.abs(h - mean_others),
                h * mean_others,
            ],
            dim=-1,
        )
        logits = self.scorer(feats).squeeze(-1)
        return logits


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
def predict(model: nn.Module, loader: DataLoader, device: torch.device, tta_perms: int = 8) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits = predict_logits_tta(model, batch, tta_perms=tta_perms)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


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
) -> Tuple[float, float]:
    model.train()
    ce = nn.CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        batch = move_batch(batch, device)
        y = batch["label"]

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        per_sample_loss = ce(logits, y)

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


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-train", type=str, default="datasets/x_train.npy")
    parser.add_argument("--x-test", type=str, default="datasets/x_test.npy")
    parser.add_argument("--y-train", type=str, default="datasets/y_train.npy")
    parser.add_argument("--y-test", type=str, default="datasets/y_test.npy", help="Optional labels for test/public subset evaluation")

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

    parser.add_argument("--cnn-dim", type=int, default=16)
    parser.add_argument("--meta-out", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--tta-perms-val", type=int, default=4)
    parser.add_argument("--tta-perms-test", type=int, default=8)

    parser.add_argument("--out-size", type=int, default=32)
    parser.add_argument("--inner-size", type=int, default=26)
    parser.add_argument("--crop-pad", type=int, default=2)

    parser.add_argument("--p-permute", type=float, default=1.0)
    parser.add_argument("--p-geom", type=float, default=0.9)
    parser.add_argument("--max-rotate-deg", type=float, default=180.0)
    parser.add_argument("--max-translate-px", type=float, default=0.0)
    parser.add_argument("--max-scale-frac", type=float, default=0.0)
    parser.add_argument("--p-noise", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--p-brightness", type=float, default=0.0)
    parser.add_argument("--brightness-frac", type=float, default=0.0)

    parser.add_argument("--stage2-hard-fraction", type=float, default=0.30)
    parser.add_argument("--stage2-hard-weight", type=float, default=3.0)
    parser.add_argument("--stage2-ohem-frac", type=float, default=1.0)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    x_train = np.load(args.x_train).astype(np.float32)
    x_test = np.load(args.x_test).astype(np.float32)
    y_train = np.load(args.y_train).astype(np.int64)

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
        max_translate_px=args.max_translate_px,
        max_scale_frac=args.max_scale_frac,
        p_noise=args.p_noise,
        noise_std=args.noise_std,
        p_brightness=args.p_brightness,
        brightness_frac=args.brightness_frac,
    )

    all_fold_scores = []
    test_logits_sum = None

    for fold_id in fold_ids:
        tr_idx, va_idx = folds[fold_id]

        x_tr = x_train[tr_idx]
        y_tr = y_train[tr_idx]
        x_va = x_train[va_idx]
        y_va = y_train[va_idx]

        processed_tr_for_stats, meta_names = preprocess_dataset(x_tr, preprocess_cfg)
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

        model = RelationOddOneOutNet(
            meta_dim=meta_dim,
            cnn_dim=args.cnn_dim,
            meta_out=args.meta_out,
            d_model=args.d_model,
            dropout=args.dropout,
        ).to(device)
        print(f"trainable params: {count_trainable_params(model):,}")

        best_state = None
        best_val_acc = -1.0
        best_val_loss = float("inf")
        wait = 0

        fold_model_path = args.save_model.replace(".pt", f"_fold{fold_id}.pt")

        # -----------------
        # Stage 1: standard training
        # -----------------
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

        # -----------------
        # Mine hard examples once using stage-1 model
        # -----------------
        train_losses = collect_per_sample_losses(model, train_eval_loader, device)
        stage2_sampler, is_hard, hard_threshold = make_stage2_sampler(
            losses=train_losses,
            hard_fraction=args.stage2_hard_fraction,
            hard_weight=args.stage2_hard_weight,
        )

        print(
            f"[stage2] mined {is_hard.sum()}/{len(is_hard)} hard samples "
            f"({100.0 * is_hard.mean():.1f}%) | "
            f"loss threshold >= {hard_threshold:.4f}"
        )

        stage2_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=stage2_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # -----------------
        # Stage 2: lower LR + hard-biased fine-tuning
        # -----------------
        remaining_epochs = args.epochs - args.stage1_epochs
        if remaining_epochs > 0:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.stage2_lr,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(remaining_epochs, 1))
            wait = 0

            for epoch in range(args.stage1_epochs + 1, args.epochs + 1):
                train_loss, train_acc = train_one_epoch(
                    model=model,
                    loader=stage2_loader,
                    optimizer=optimizer,
                    device=device,
                    grad_clip=args.grad_clip,
                    hard_mining_frac=args.stage2_ohem_frac,
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

        if test_logits_sum is None:
            test_logits_sum = fold_logits
        else:
            test_logits_sum += fold_logits

    mean_test_logits = test_logits_sum / float(len(fold_ids))
    preds = mean_test_logits.argmax(dim=1).numpy()

    if args.y_test and os.path.exists(args.y_test):
        y_test = np.load(args.y_test).astype(np.int64)
        n_public = len(y_test)

        if n_public <= len(preds):
            public_logits = mean_test_logits[:n_public]
            public_preds = public_logits.argmax(dim=1).numpy()
            public_acc = (public_preds == y_test).mean()
            print(f"public test acc: {public_acc:.4f}")
        else:
            print("Skipping y_test evaluation because y_test is longer than test set.")

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