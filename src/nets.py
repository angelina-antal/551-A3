import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import json

from src.augment import CenteredAugConfig, CenteredOddOneOutAugment, OddOneOutDataset
from src.preprocess import PreprocessConfig


# ---------------------------------------------------------------------------
# constants — change these in ONE place and everything stays consistent
# ---------------------------------------------------------------------------
CNN_EMB   = 20   # TinyCNN output dim
META_OUT  = 12   # meta MLP output dim
D_MODEL   = 24   # shared set-level embedding dim
K         = 6    # number of archetypes
REL_DIM   = 10   # pairwise relation dim

# derived — do NOT edit, computed from the above
_ITEM_IN      = CNN_EMB + META_OUT          # 32
_PAIR_IN      = 3 * D_MODEL + 2            # 74  (xi, xj, |ci-cj|, q_overlap, coord_dist)
_SCORER_IN    = 5 * D_MODEL + 2 * REL_DIM + K  # 146


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_csv_kaggle(yh_test: np.ndarray, out_path: str = "predicted_labels.csv"):
    import pandas as pd
    df = pd.DataFrame({"Id": np.arange(len(yh_test)), "Category": yh_test.astype(int)})
    df.to_csv(out_path, index=False)

# ---------------------------------------------------------------------------
# dataset  (centered_raw + meta only)
# ---------------------------------------------------------------------------

class DictDataset(Dataset):
    def __init__(self, arrays: dict, y=None, train: bool = False):
        self.arrays = arrays
        self.y      = None if y is None else y.astype(np.int64)

    def __len__(self):
        return len(self.arrays["centered_raw"])

    def __getitem__(self, idx):
        sample = {
            "centered_raw":       self.arrays["centered_raw"][idx].copy().astype(np.float32),
            "meta":               self.arrays["meta"][idx].copy().astype(np.float32),
        }

        sample["centered_raw"] = (sample["centered_raw"])
        sample["meta"]         = (sample["meta"])

        if self.y is None:
            return {k: torch.from_numpy(v) for k, v in sample.items()}

        label = int(self.y[idx])

        return (
            {k: torch.from_numpy(v) for k, v in sample.items()},
            torch.tensor(label, dtype=torch.long),
        )


def load_npz_arrays(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files if k != "meta_names"}


# ---------------------------------------------------------------------------
# model components
# ---------------------------------------------------------------------------

class TinyCNN(nn.Module):
    """Encodes a (2, H, W) image+mask pair → CNN_EMB-dim vector."""

    def __init__(self, in_ch: int = 1, emb_dim: int = CNN_EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # → 16
            nn.Conv2d(8, 12, 3, padding=1),    nn.ReLU(inplace=True), nn.MaxPool2d(2),   # → 8
            nn.Conv2d(12, 16, 3, padding=1),   nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, emb_dim)

    def forward(self, x):                       # x: [B*N, 2, H, W]
        return self.fc(self.net(x).flatten(1))  # → [B*N, emb_dim]


class ArchetypeModule(nn.Module):
    def __init__(self, d_model: int = D_MODEL, num_archetypes: int = K):
        super().__init__()
        self.centers   = nn.Parameter(torch.randn(num_archetypes, d_model) * 0.02)
        self.to_logits = nn.Linear(d_model, num_archetypes)

    def forward(self, x):                               # x: [B, N, D]
        q      = F.softmax(self.to_logits(x), dim=-1)  # [B, N, K]
        center = torch.matmul(q, self.centers)          # [B, N, D]
        coord  = x - center                             # family-relative residual
        return q, center, coord


class SetBlock(nn.Module):
    def __init__(self, d_model: int = D_MODEL, heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x    = self.norm1(x + a)
        return self.norm2(x + self.ff(x))


class PairwiseAggregator(nn.Module):
    """
    Computes per-item features by aggregating pairwise relations.
    Input dim = 3*D_MODEL + 2  (xi, xj, |ci-cj|, q_overlap, coord_dist)
    """

    def __init__(self, d_model: int = D_MODEL, rel_dim: int = REL_DIM):
        super().__init__()
        pair_in = 3 * d_model + 2          # 74 with defaults
        self.rel = nn.Sequential(
            nn.Linear(pair_in, rel_dim), nn.ReLU(inplace=True),
            nn.Linear(rel_dim, rel_dim),   nn.ReLU(inplace=True),
        )

    def forward(self, x, q, coord):
        # x, coord: [B, N, D]   q: [B, N, K]
        B, N, D = x.shape

        xi = x.unsqueeze(2).expand(B, N, N, D)    # [B, N, N, D]
        xj = x.unsqueeze(1).expand(B, N, N, D)
        ci = coord.unsqueeze(2).expand(B, N, N, D)
        cj = coord.unsqueeze(1).expand(B, N, N, D)

        q_overlap  = (q.unsqueeze(2) * q.unsqueeze(1)).sum(-1, keepdim=True)   # [B,N,N,1]
        coord_dist = torch.norm(ci - cj, dim=-1, keepdim=True)                 # [B,N,N,1]

        feat = torch.cat([xi, xj, torch.abs(ci - cj), q_overlap, coord_dist], dim=-1)
        # feat: [B, N, N, 3*D+2]  ← correct by construction
        rel = self.rel(feat)   # [B, N, N, rel_dim]

        eye      = torch.eye(N, device=x.device, dtype=torch.bool).view(1, N, N, 1)
        rel_mean = rel.masked_fill(eye, 0.0).sum(2) / float(N - 1)       # [B, N, rel_dim]
        rel_max  = rel.masked_fill(eye, -1e9).max(2).values              # [B, N, rel_dim]
        return rel_mean, rel_max


class OddOneOutNet(nn.Module):
    """
    Centered-only relational network for 5-way odd-one-out.

    Input keys (per batch):
        centered_raw        [B, 5, H, W]   (float32, normalised)
        meta                [B, 5, meta_dim]
    """

    def __init__(self, meta_dim: int):
        super().__init__()

        self.encoder  = TinyCNN(in_ch=1, emb_dim=CNN_EMB)

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, META_OUT), nn.ReLU(inplace=True),
            nn.Linear(META_OUT, META_OUT),
        )

        # _ITEM_IN = CNN_EMB + META_OUT = 32
        self.item_proj = nn.Sequential(
            nn.Linear(_ITEM_IN, D_MODEL), nn.ReLU(inplace=True),
            nn.Linear(D_MODEL, D_MODEL),
        )

        self.archetypes = ArchetypeModule(d_model=D_MODEL, num_archetypes=K)
        self.set_block  = SetBlock(d_model=D_MODEL, heads=2, dropout=0.1)
        self.pair       = PairwiseAggregator(d_model=D_MODEL, rel_dim=REL_DIM)

        # _SCORER_IN = 5*D_MODEL + 2*REL_DIM + K = 146
        self.scorer = nn.Sequential(
            nn.Linear(_SCORER_IN, 40), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(40, 1),
        )

    def _encode(self, gray):
        B, N, H, W = gray.shape
        x = gray.reshape(B * N, 1, H, W)
        return self.encoder(x).view(B, N, -1)

    def forward(self, batch):
        z_cen  = self._encode(batch["centered_raw"])

        B      = batch["meta"].shape[0]
        z_meta = self.meta_mlp(batch["meta"].view(-1, batch["meta"].shape[-1])).view(B, 5, -1)

        h0 = self.item_proj(torch.cat([z_cen, z_meta], dim=-1))   # [B, 5, D_MODEL]

        q, center, coord = self.archetypes(h0)
        h = self.set_block(h0 + center)

        # mean of the other 4 items for each position
        mean_other = torch.stack([
            torch.cat([h[:, :i], h[:, i+1:]], dim=1).mean(1)
            for i in range(5)
        ], dim=1)   # [B, 5, D_MODEL]

        rel_mean, rel_max = self.pair(h, q, coord)

        feat = torch.cat([
            h0,                      # per-item identity     [B,5,D]
            h,                       # contextualised        [B,5,D]
            center,                  # family centre         [B,5,D]
            coord,                   # family residual       [B,5,D]
            torch.abs(h - mean_other),  # deviation          [B,5,D]
            rel_mean,                # avg pairwise          [B,5,REL_DIM]
            rel_max,                 # max pairwise          [B,5,REL_DIM]
            q,                       # archetype weights     [B,5,K]
        ], dim=-1)
        # feat: [B, 5, 5*D_MODEL + 2*REL_DIM + K] = [B, 5, 146]

        logits = self.scorer(feat).squeeze(-1)   # [B, 5]
        return logits, q


# ---------------------------------------------------------------------------
# train / evaluate / predict
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    total_loss, total_correct, total = 0.0, 0, 0

    for batch, y in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y     = y.to(device)

        optimizer.zero_grad()
        logits, q = model(batch)

        loss_main = ce(logits, y)

        # entropy bonus: encourage diverse archetype usage
        mean_q  = q.mean(dim=(0, 1))
        entropy = -(mean_q * (mean_q + 1e-8).log()).sum()
        loss    = loss_main - 0.01 * entropy

        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total         += y.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0

    for batch, y in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y     = y.to(device)
        logits, _ = model(batch)
        total_loss    += ce(logits, y).item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total         += y.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits, _ = model(batch)
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    type=str,   default="datasets")
    parser.add_argument("--train-npz",   type=str,   default="datasets/processed_train_best.npz")
    parser.add_argument("--test-npz",    type=str,   default="datasets/processed_test_best.npz")
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=2e-3)
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    parser.add_argument("--val-size",    type=float, default=0.2)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--num-workers", type=int,   default=0)
    parser.add_argument("--save-path",   type=str,   default="best_model.pt")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    x_train_raw = np.load(os.path.join(args.data_dir, "x_train.npy"))

    arrays = load_npz_arrays(args.train_npz)   # normalized processed train, used for val split
    meta_dim = arrays["meta"].shape[-1]

    idx = np.arange(len(y_train))
    tr_idx, va_idx, y_tr, y_va = train_test_split(
        idx, y_train, test_size=args.val_size, random_state=args.seed, stratify=y_train,
    )

    def slice_arrays(arrs, inds):
        return {
            k: v[inds] if (isinstance(v, np.ndarray) and v.shape[0] == len(y_train)) else v
            for k, v in arrs.items()
        }

    # load preprocess config + normalization stats
    with open(os.path.join(args.data_dir, "preprocess_config_best.json"), "r", encoding="utf-8") as f:
        preprocess_cfg = PreprocessConfig(**json.load(f))

    stats_npz = np.load(os.path.join(args.data_dir, "preprocess_stats_best.npz"))
    norm_stats = {
        "centered_raw_mean": np.float32(stats_npz["centered_raw_mean"]),
        "centered_raw_std": np.float32(stats_npz["centered_raw_std"]),
        "meta_mean": stats_npz["meta_mean"].astype(np.float32),
        "meta_std": stats_npz["meta_std"].astype(np.float32),
    }

    train_aug = CenteredOddOneOutAugment(
        cfg=CenteredAugConfig(
            p_permute=1.0,
            p_geom=0.9,
            max_rotate_deg=180.0,
            max_translate_px=0.0,
            max_scale_frac=0.00,
            p_noise=0.0,
            noise_std=0.00,
            p_brightness=0.0,
            brightness_frac=0.0,
        ),
        preprocess_cfg=preprocess_cfg,
        norm_stats=norm_stats,
    )

    # train on raw images + online augmentation
    train_ds = OddOneOutDataset(x_train_raw[tr_idx], y=y_tr, augment=train_aug)

    # validate on frozen normalized processed arrays
    val_ds = DictDataset(slice_arrays(arrays, va_idx), y=y_va, train=False)

    test_arrays = load_npz_arrays(args.test_npz)
    test_ds = DictDataset(test_arrays, y=None, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = OddOneOutNet(meta_dim=meta_dim).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {total_params:,}")
    assert total_params <= 25_000, f"model too large: {total_params:,} > 25,000"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_state, bad = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")

        if va_acc > best_acc:
            best_acc   = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad        = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.save_path)
    print(f"best val acc: {best_acc:.4f}  →  saved to {args.save_path}")

    # optional public-half eval
    y_test_path = os.path.join(args.data_dir, "y_test.npy")
    if os.path.exists(y_test_path):
        from sklearn.metrics import accuracy_score
        y_test = np.load(y_test_path)
        n      = min(len(y_test), len(test_ds))
        pub_arrays = {k: v[:n] for k, v in test_arrays.items() if isinstance(v, np.ndarray)}
        pub_ds     = DictDataset(pub_arrays, y=y_test[:n], train=False)
        pub_loader = DataLoader(pub_ds, batch_size=args.batch_size, shuffle=False)
        pub_preds  = predict(model, pub_loader, device)
        print(f"public test acc (first {n}): {accuracy_score(y_test[:n], pub_preds):.4f}")

    yh_test = predict(model, test_loader, device)
    generate_csv_kaggle(yh_test)
    print("wrote predicted_labels.csv")


if __name__ == "__main__":
    main()
