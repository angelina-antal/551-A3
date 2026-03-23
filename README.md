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
# 1. Preprocess (crop, center, extract metadata)
python -m src.preprocess --data-dir datasets

# 2. Train (uses augmentation + early stopping)
python -m src.nets --data-dir datasets

# Outputs: best_model.pt, predicted_labels.csv
```

## Structure

| File | Role |
|------|------|
| `src/preprocess.py` | Crops, centers images; extracts shape metadata (Hu moments, contours, etc.) |
| `src/augment.py` | Online augmentation: rotation, permutation, etc. |
| `src/nets.py` | OddOneOutNet: CNN + meta MLP + attention + pairwise relations (~72% test) |
| `best_model.pt` | Saved weights (committed) |
| `predicted_labels.csv` | Kaggle submission format |

## Model

- **Architecture:** TinyCNN per image + metadata MLP → set-level attention + archetype clustering + pairwise aggregator → 5-way scorer
- **Params:** < 25,000 (assignment limit)
- **Input:** Preprocessed `centered_raw` (32×32) + `meta` (12-d shape features per image)

## Dataset layout

After preprocessing:
- `datasets/processed_train_best.npz` — normalized train
- `datasets/processed_test_best.npz` — normalized test  
- `datasets/preprocess_stats_best.npz` — normalization stats
- `datasets/preprocess_config_best.json` — config used
