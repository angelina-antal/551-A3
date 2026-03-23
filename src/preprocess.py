import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

#Helpers

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


# Geometry/ feature extraction

def pca_stats(mask: np.ndarray) -> Dict[str, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return {
            "major_axis": 0.0,
            "minor_axis": 0.0,
            "eccentricity": 0.0,
            "elongation": 1.0,
        }

    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    mean_xy = coords.mean(axis=0)
    centered = coords - mean_xy[None, :]
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)

    order = np.argsort(eigvals)[::-1]
    l1 = float(eigvals[order[0]])
    l2 = float(eigvals[order[1]])

    major_axis = float(4.0 * math.sqrt(l1))
    minor_axis = float(4.0 * math.sqrt(l2))
    eccentricity = float(math.sqrt(max(0.0, 1.0 - l2 / (l1 + 1e-8))))
    elongation = float(math.sqrt((l1 + 1e-8) / (l2 + 1e-8)))

    return {
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "eccentricity": eccentricity,
        "elongation": elongation,
    }


def contour_features(mask: np.ndarray, area_px: float) -> Dict[str, float]:
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "perimeter": 0.0,
            "solidity": 0.0,
            "compactness": 0.0,
            "convexity": 0.0,
        }

    cnt = max(contours, key=cv2.contourArea)
    cnt_area = max(float(cv2.contourArea(cnt)), 1e-6)
    perimeter = float(cv2.arcLength(cnt, True))

    hull = cv2.convexHull(cnt)
    hull_area = max(float(cv2.contourArea(hull)), 1e-6)
    hull_perimeter = max(float(cv2.arcLength(hull, True)), 1e-6)

    solidity = cnt_area / hull_area
    compactness = (perimeter ** 2) / (4.0 * math.pi * max(area_px, 1e-6))
    convexity = hull_perimeter / max(perimeter, 1e-6)

    return {
        "perimeter": perimeter,
        "solidity": solidity,
        "compactness": compactness,
        "convexity": convexity,
    }




def hu_features(mask: np.ndarray) -> Dict[str, float]:
    mask_u8 = (mask > 0).astype(np.uint8)
    hu = cv2.HuMoments(cv2.moments(mask_u8)).flatten().astype(np.float32)
    hu = np.sign(hu) * np.log1p(np.abs(hu))
    return {f"hu_{i+1}": float(hu[i]) for i in range(7)}


def compute_metadata(
    img01: np.ndarray,
    feature_mask: np.ndarray,
) -> Tuple[np.ndarray, list]:
    h, w = img01.shape
    ys, xs = np.where(feature_mask > 0)

    names = [
    "area_frac",
    "major_axis", "minor_axis", "elongation",
    "solidity", "compactness", "convexity",
    "hu_1", "hu_2", "hu_3", "hu_4",
]

    if len(xs) == 0:
        return np.zeros(len(names), dtype=np.float32), names

    area = float(len(xs))
    area_frac = area / float(h * w)


    pca = pca_stats(feature_mask)
    contour = contour_features(feature_mask, area)
    hu = hu_features(feature_mask)

    feat_dict = {
    "area_frac": area_frac,
    "major_axis": pca["major_axis"],
    "minor_axis": pca["minor_axis"],
    "elongation": pca["elongation"],
    "solidity": contour["solidity"],
    "compactness": contour["compactness"],
    "convexity": contour["convexity"],
    "hu_1": hu["hu_1"],
    "hu_2": hu["hu_2"],
    "hu_3": hu["hu_3"],
    "hu_4": hu["hu_4"],
}

    meta = np.array([feat_dict[n] for n in names], dtype=np.float32)
    return meta, names



# Single-image preprocessing


def preprocess_single(
    img: np.ndarray,
    out_size: int = 32,
    inner_size: int = 26,
    crop_pad: int = 2,
    hard_threshold: float = 0.0001,
):
    raw = ensure_float01(img)

    center_mask = fixed_mask(raw, threshold=0.0)
    hard_mask = fixed_mask(raw, threshold=hard_threshold)
    if hard_mask.sum() == 0:
        hard_mask = center_mask.copy()

    meta, meta_names = compute_metadata(raw, hard_mask)

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



# Dataset preprocessing


@dataclass
class PreprocessConfig:
    out_size: int = 32
    inner_size: int = 26
    crop_pad: int = 2
    hard_threshold: float = 0.0001

def preprocess_dataset(x: np.ndarray, cfg: PreprocessConfig):
    n, k, h, w = x.shape

    first = preprocess_single(
        x[0, 0],
        out_size=cfg.out_size,
        inner_size=cfg.inner_size,
        crop_pad=cfg.crop_pad,
        hard_threshold=cfg.hard_threshold,
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
                hard_threshold=cfg.hard_threshold,
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
    np.savez_compressed(path, **processed, meta_names=np.asarray(meta_names, dtype=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="datasets")
    parser.add_argument("--out-train", type=str, default="datasets/processed_train_best.npz")
    parser.add_argument("--out-test", type=str, default="datasets/processed_test_best.npz")
    parser.add_argument("--out-stats", type=str, default="datasets/preprocess_stats_best.npz")
    parser.add_argument("--out-config", type=str, default="datasets/preprocess_config_best.json")

    parser.add_argument("--out-size", type=int, default=32)
    parser.add_argument("--inner-size", type=int, default=26)
    parser.add_argument("--crop-pad", type=int, default=2)
    parser.add_argument("--hard-threshold", type=float, default=0.0001)

    args = parser.parse_args()

    cfg = PreprocessConfig(
        out_size=args.out_size,
        inner_size=args.inner_size,
        crop_pad=args.crop_pad,
        hard_threshold=args.hard_threshold,
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
    np.savez_compressed(args.out_stats, **stats, meta_names=np.asarray(meta_names, dtype=str))

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
