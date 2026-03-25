# COMP551 A3: Odd-One-Out Image Groups

5-class classification: given 5 grayscale images, identify which one is the outlier.

## Setup

1. **Download data** from [Kaggle](https://www.kaggle.com/competitions/mcgill-comp551-winter2026-a3/data):
   - `x_train.npy`, `y_train.npy`, `x_test.npy`, `y_test.npy` → place in `datasets/`

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline

### Option A: Run everything
```bash
./run_all.sh
```

### Option B: Step by step
```bash
# 1. Preprocess (crop, center, area_frac metadata) → processed_*_area_only.npz
python -m src.preprocess --data-dir datasets

# 2. Train (5-fold CV, augmentation, early stopping). Prints **public test accuracy**
#    when `datasets/y_test.npy` is present (same metric as the course / Kaggle public).
python -m src.net17

# Outputs: best_model_fold0.pt … best_model_fold4.pt, predicted_labels.csv  (checkpoints gitignored)
```

## Structure

| File | Role |
|------|------|
| `src/preprocess.py` | Crops, centers images; extracts shape metadata (Hu moments, contours, etc.) |
| `src/augment.py` | Online augmentation: rotation, permutation, etc. |
| `src/net17.py` | `MultiRuleQuartetRankNet`: CNN + meta MLP + attention + pairwise relations |
| `src/nets.py` | Notebook-friendly aliases (`OddOneOutNet`, loaders) around `net17` |
| `best_model_fold*.pt` / `best_model.pt` | Saved weights — **gitignored** (`best*`); train locally or use a teammate’s files |
| `predicted_labels.csv` | Kaggle submission format |

## Model

- **Architecture:** TinyCNN per image + metadata MLP → set-level attention + archetype clustering + pairwise aggregator → 5-way scorer
- **Params:** < 25,000 (assignment limit)
- **Input:** Preprocessed `centered_raw` (32×32) + `meta` (12-d shape features per image)

## Dataset layout

After preprocessing (current defaults):
- `datasets/processed_train_area_only.npz` — normalized train (`meta_dim=1`)
- `datasets/processed_test_area_only.npz` — normalized test
- Matching stats/config: `preprocess_stats_area_only.npz`, `preprocess_config_area_only.json`

Older `processed_*_best.npz` files in some clones use a different meta dimension and are **not** compatible with the current `net17` / notebook pipeline.

**Getting a % score:** run `python -m src.net17` and read `final ensemble public test acc` and `mean CV acc` in the log. The assignment notebook can also evaluate **if** you have a compatible `best_model.pt` and the `*_area_only.npz` files.
