import random
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocess import preprocess_single, PreprocessConfig


@dataclass
class CenteredAugConfig:
    p_permute: float = 1.0

    p_geom: float = 0.9
    max_rotate_deg: float = 180.0

    clip_min: float = 0.0
    clip_max: float = 1.0


def _ensure_float01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _warp_centered_image(
    img: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    h, w = img.shape
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    out = cv2.warpAffine(
        img.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return out.astype(np.float32)


class CenteredOddOneOutAugment:
    def __init__(
        self,
        cfg: CenteredAugConfig,
        preprocess_cfg: PreprocessConfig,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.cfg = cfg
        self.preprocess_cfg = preprocess_cfg
        self.norm_stats = norm_stats

    def _normalize(self, imgs: np.ndarray, meta: np.ndarray):
        if self.norm_stats is None:
            return (
                imgs.astype(np.float32),
                meta.astype(np.float32),
            )

        imgs = (
            imgs - float(self.norm_stats["centered_raw_mean"])
        ) / float(self.norm_stats["centered_raw_std"])

        meta = (
            meta - self.norm_stats["meta_mean"][None, :]
        ) / self.norm_stats["meta_std"][None, :]

        return (
            imgs.astype(np.float32),
            meta.astype(np.float32),
        )

    def _sample_geom_params(self):
        if random.random() >= self.cfg.p_geom:
            return None

        return {
            "angle": random.uniform(-self.cfg.max_rotate_deg, self.cfg.max_rotate_deg)
        }

    def _augment_one_image(self, img: np.ndarray, geom_params=None) -> np.ndarray:
        img = _ensure_float01(img)

        if geom_params is not None:
            img = _warp_centered_image(
                img=img,
                angle_deg=geom_params["angle"]
            )

        return np.clip(img, self.cfg.clip_min, self.cfg.clip_max).astype(np.float32)

    def __call__(self, imgs: np.ndarray, label: int):
        imgs = np.asarray(imgs, dtype=np.float32)
        assert imgs.ndim == 3 and imgs.shape[0] == 5, f"Expected (5,H,W), got {imgs.shape}"

        out = imgs.copy()
        y = int(label)

        if random.random() < self.cfg.p_permute:
            perm = np.random.permutation(5)
            out = out[perm]
            y = int(np.where(perm == y)[0][0])

        geom_params = self._sample_geom_params()

        proc_imgs = []
        proc_meta = []

        for i in range(5):
            aug_img = self._augment_one_image(out[i], geom_params=geom_params)

            p = preprocess_single(
                aug_img,
                out_size=self.preprocess_cfg.out_size,
                inner_size=self.preprocess_cfg.inner_size,
                crop_pad=self.preprocess_cfg.crop_pad,
            )

            proc_imgs.append(p["centered_raw"])
            proc_meta.append(p["meta"])

        out = np.stack(proc_imgs, axis=0).astype(np.float32)
        meta = np.stack(proc_meta, axis=0).astype(np.float32)

        out, meta = self._normalize(out, meta)
        return out, meta, y


class CenteredOddOneOutEvalTransform:
    def __init__(
        self,
        preprocess_cfg: PreprocessConfig,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.preprocess_cfg = preprocess_cfg
        self.norm_stats = norm_stats

    def _normalize(self, imgs: np.ndarray, meta: np.ndarray):
        if self.norm_stats is None:
            return (
                imgs.astype(np.float32),
                meta.astype(np.float32),
            )

        imgs = (
            imgs - float(self.norm_stats["centered_raw_mean"])
        ) / float(self.norm_stats["centered_raw_std"])

        meta = (
            meta - self.norm_stats["meta_mean"][None, :]
        ) / self.norm_stats["meta_std"][None, :]

        return (
            imgs.astype(np.float32),
            meta.astype(np.float32),
        )

    def __call__(self, imgs: np.ndarray, label: int):
        imgs = np.asarray(imgs, dtype=np.float32)
        assert imgs.ndim == 3 and imgs.shape[0] == 5, f"Expected (5,H,W), got {imgs.shape}"

        proc_imgs = []
        proc_meta = []

        for i in range(5):
            img = _ensure_float01(imgs[i])

            p = preprocess_single(
                img,
                out_size=self.preprocess_cfg.out_size,
                inner_size=self.preprocess_cfg.inner_size,
                crop_pad=self.preprocess_cfg.crop_pad,
            )

            proc_imgs.append(p["centered_raw"])
            proc_meta.append(p["meta"])

        out = np.stack(proc_imgs, axis=0).astype(np.float32)
        meta = np.stack(proc_meta, axis=0).astype(np.float32)

        out, meta = self._normalize(out, meta)
        return out, meta, int(label)


class OddOneOutDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray = None, transform=None):
        self.x = x.astype(np.float32)
        self.y = None if y is None else y.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        imgs = self.x[idx]
        label = -1 if self.y is None else int(self.y[idx])

        if self.transform is None:
            raise RuntimeError("A transform must be provided for both train and validation.")

        imgs, meta, label = self.transform(imgs, label)

        sample = {
            "centered_raw": torch.from_numpy(imgs),
            "meta": torch.from_numpy(meta),
        }

        return sample, torch.tensor(label, dtype=torch.long)