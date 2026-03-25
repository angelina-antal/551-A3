"""
Thin API layer for notebooks and docs.

Implements `MultiRuleQuartetRankNet` in `net17.py`; this module exposes stable names
and helpers that match the assignment notebook.
"""

import csv
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.net17 import GroupDataset, MultiRuleQuartetRankNet, predict as predict_fn

# Defaults must match `argparse` defaults in `src.net17` `main()`, not the class __init__ defaults.


class OddOneOutNet(MultiRuleQuartetRankNet):
    """Same architecture as the training script; hyperparameters match CLI defaults."""

    def __init__(self, meta_dim: int, **kwargs: Any) -> None:
        defaults: Dict[str, Any] = {
            "cnn_dim": 10,
            "meta_out": 4,
            "d_model": 16,
            "num_rules": 3,
            "set_blocks": 2,
            "rel_dim": 8,
            "rel_hidden": 16,
            "rel_proj_dim": 6,
            "dropout": 0.10,
            "tau_softmin": 0.35,
            "route_temp": 0.70,
        }
        defaults.update(kwargs)
        super().__init__(meta_dim=meta_dim, **defaults)


DictDataset = GroupDataset


def load_npz_arrays(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files if k != "meta_names"}


def predict(model: torch.nn.Module, loader: Any, device: torch.device, tta_perms: int = 8) -> np.ndarray:
    return predict_fn(model, loader, device, tta_perms=tta_perms)


def generate_csv_kaggle(preds: np.ndarray, path: str = "predicted_labels.csv") -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Label"])
        for i, p in enumerate(preds):
            w.writerow([i, int(p)])


_CHECKPOINT_CANDIDATES = (
    "best_model.pt",
    *(f"best_model_fold{k}.pt" for k in range(5)),
)


def load_checkpoint_or_raise(model: torch.nn.Module, path: Optional[str] = None) -> str:
    """Load first existing checkpoint. Training writes `best_model_fold{k}.pt` per fold."""
    tried: list[str] = []
    if path:
        tried.append(path)
    tried.extend(p for p in _CHECKPOINT_CANDIDATES if p not in tried)

    for p in tried:
        if os.path.isfile(p):
            model.load_state_dict(torch.load(p, map_location="cpu"))
            return p

    raise FileNotFoundError(
        "No checkpoint found (tried: "
        + ", ".join(tried[:6])
        + ", …). Weights are gitignored (see .gitignore 'best*'). "
        "Train with:  python -m src.net17"
    )
