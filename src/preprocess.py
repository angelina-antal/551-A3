import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


def apply_normalization(
    processed: Dict[str, np.ndarray],
    stats: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    centered_raw = processed["centered_raw"].astype(np.float32)
    meta = processed["meta"].astype(np.float32)

    centered_raw = (centered_raw - stats["centered_raw_mean"]) / stats["centered_raw_std"]
    meta = (meta - stats["meta_mean"][None, None, :]) / stats["meta_std"][None, None, :]

    return {
        "centered_raw": centered_raw.astype(np.float32),
        "meta": meta.astype(np.float32),
    }


def ensure_float01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def fixed_mask(img01: np.ndarray, threshold: float) -> np.ndarray:
    mask = (img01 > threshold).astype(np.uint8)
    return mask


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_with_pad(arr: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 2) -> np.ndarray:
    h, w = arr.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    return arr[y1:y2 + 1, x1:x2 + 1]


def paste_center_preserve_aspect(
    arr: np.ndarray,
    out_size: int = 32,
    inner_size: int = 26,
    interp: int = cv2.INTER_LINEAR
) -> np.ndarray:
    h, w = arr.shape[:2]
    canvas = np.zeros((out_size, out_size), dtype=np.float32)
    if h <= 0 or w <= 0:
        return canvas

    scale = min(inner_size / float(h), inner_size / float(w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(arr.astype(np.float32), (nw, nh), interpolation=interp)

    y0 = (out_size - nh) // 2
    x0 = (out_size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def compute_metadata(
    img01: np.ndarray,
) -> Tuple[np.ndarray, list]:
    h, w = img01.shape
    area = float(np.count_nonzero(img01 > 0))

    names = ["area_frac"]

    if area == 0:
        return np.zeros(len(names), dtype=np.float32), names

    area_frac = area / float(h * w)
    meta = np.array([area_frac], dtype=np.float32)
    return meta, names


def preprocess_single(
    img: np.ndarray,
    out_size: int = 32,
    inner_size: int = 26,
    crop_pad: int = 2,
):
    raw = ensure_float01(img)
    center_mask = fixed_mask(raw, threshold=0.0)
    meta, meta_names = compute_metadata(raw)

    if center_mask.sum() == 0:
        centered_raw = np.zeros((out_size, out_size), dtype=np.float32)
    else:
        x1, y1, x2, y2 = bbox_from_mask(center_mask)
        crop_raw = crop_with_pad(raw, x1, y1, x2, y2, pad=crop_pad)
        centered_raw = paste_center_preserve_aspect(
            crop_raw,
            out_size=out_size,
            inner_size=inner_size,
            interp=cv2.INTER_LINEAR,
        ).astype(np.float32)

    return {
        "centered_raw": centered_raw,
        "meta": meta.astype(np.float32),
        "meta_names": meta_names,
    }


@dataclass
class PreprocessConfig:
    out_size: int = 32
    inner_size: int = 26
    crop_pad: int = 2


def preprocess_dataset(x: np.ndarray, cfg: PreprocessConfig):
    n, k, h, w = x.shape

    first = preprocess_single(
        x[0, 0],
        out_size=cfg.out_size,
        inner_size=cfg.inner_size,
        crop_pad=cfg.crop_pad,
    )
    meta_dim = len(first["meta"])
    meta_names = first["meta_names"]

    out = {
        "centered_raw": np.zeros((n, k, cfg.out_size, cfg.out_size), dtype=np.float32),
        "meta": np.zeros((n, k, meta_dim), dtype=np.float32),
    }

    for i in range(n):
        if (i + 1) % 200 == 0 or i == 0 or i == n - 1:
            print(f"processing group {i+1}/{n}")
        for j in range(k):
            p = preprocess_single(
                x[i, j],
                out_size=cfg.out_size,
                inner_size=cfg.inner_size,
                crop_pad=cfg.crop_pad,
            )
            out["centered_raw"][i, j] = p["centered_raw"]
            out["meta"][i, j] = p["meta"]

    return out, meta_names


def compute_train_stats(processed_train: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    stats = {}

    arr = processed_train["centered_raw"]
    stats["centered_raw_mean"] = np.float32(arr.mean())
    stats["centered_raw_std"] = np.float32(arr.std() + 1e-6)

    meta = processed_train["meta"]
    stats["meta_mean"] = meta.mean(axis=(0, 1)).astype(np.float32)
    stats["meta_std"] = (meta.std(axis=(0, 1)) + 1e-6).astype(np.float32)

    return stats


def save_processed_npz(path: str, processed: Dict[str, np.ndarray], meta_names):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(
        path,
        **processed,
        meta_names=np.asarray(meta_names, dtype=str),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="datasets")
    parser.add_argument("--out-train", type=str, default="datasets/processed_train_area_only.npz")
    parser.add_argument("--out-test", type=str, default="datasets/processed_test_area_only.npz")
    parser.add_argument("--out-stats", type=str, default="datasets/preprocess_stats_area_only.npz")
    parser.add_argument("--out-config", type=str, default="datasets/preprocess_config_area_only.json")

    parser.add_argument("--out-size", type=int, default=32)
    parser.add_argument("--inner-size", type=int, default=26)
    parser.add_argument("--crop-pad", type=int, default=2)

    args = parser.parse_args()

    cfg = PreprocessConfig(
        out_size=args.out_size,
        inner_size=args.inner_size,
        crop_pad=args.crop_pad,
    )

    x_train_path = os.path.join(args.data_dir, "x_train.npy")
    x_test_path = os.path.join(args.data_dir, "x_test.npy")

    if not os.path.exists(x_train_path):
        raise FileNotFoundError(f"Missing {x_train_path}")
    if not os.path.exists(x_test_path):
        raise FileNotFoundError(f"Missing {x_test_path}")

    x_train = np.load(x_train_path)
    x_test = np.load(x_test_path)

    print("preprocessing train...")
    processed_train, meta_names = preprocess_dataset(x_train, cfg)

    print("preprocessing test...")
    processed_test, _ = preprocess_dataset(x_test, cfg)

    print("computing train-only normalization stats...")
    stats = compute_train_stats(processed_train)

    print("applying normalization...")
    processed_train = apply_normalization(processed_train, stats)
    processed_test = apply_normalization(processed_test, stats)

    print("saving outputs...")
    save_processed_npz(args.out_train, processed_train, meta_names)
    save_processed_npz(args.out_test, processed_test, meta_names)

    stats_parent = os.path.dirname(args.out_stats)
    if stats_parent:
        os.makedirs(stats_parent, exist_ok=True)
    np.savez_compressed(
        args.out_stats,
        **stats,
        meta_names=np.asarray(meta_names, dtype=str),
    )

    config_parent = os.path.dirname(args.out_config)
    if config_parent:
        os.makedirs(config_parent, exist_ok=True)
    with open(args.out_config, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\nDone.")
    print("Saved:")
    print(f"  {args.out_train}")
    print(f"  {args.out_test}")
    print(f"  {args.out_stats}")
    print(f"  {args.out_config}")
    print("\nMetadata names:")
    print(", ".join(meta_names))


if __name__ == "__main__":
    main()