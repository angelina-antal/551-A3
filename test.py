import numpy as np
import pandas as pd

# Load your Kaggle-style prediction file
pred_df = pd.read_csv("predicted_labels.csv")

# Basic checks
required_cols = {"Id", "Category"}
if not required_cols.issubset(pred_df.columns):
    raise ValueError(f"CSV must contain columns {required_cols}, got {list(pred_df.columns)}")

# Sort by Id so row order is guaranteed correct
pred_df = pred_df.sort_values("Id").reset_index(drop=True)

# Load true labels for the first 1000 test samples
y_true = np.load("datasets/y_test.npy").astype(int)   # shape should be (1000,)
n = len(y_true)

# Keep only predictions for Id 0..999
pred_public = pred_df[pred_df["Id"] < n].copy()

if len(pred_public) != n:
    raise ValueError(f"Expected {n} predictions for Id 0..{n-1}, but found {len(pred_public)}")

# Extra safety check: make sure IDs are exactly 0,1,2,...,999
expected_ids = np.arange(n)
actual_ids = pred_public["Id"].to_numpy()
if not np.array_equal(actual_ids, expected_ids):
    raise ValueError("Ids in predicted_labels.csv are not exactly 0..999 after sorting")

y_pred = pred_public["Category"].to_numpy().astype(int)

# Accuracy
acc = (y_pred == y_true).mean()

print(f"Public test accuracy: {acc:.6f}")
print(f"Public test accuracy: {acc*100:.2f}%")

# Optional: show a few mistakes
wrong = np.where(y_pred != y_true)[0]
print(f"Number wrong: {len(wrong)} / {n}")

if len(wrong) > 0:
    print("\nFirst 20 mismatches:")
    for i in wrong[:20]:
        print(f"Id={i}, pred={y_pred[i]}, true={y_true[i]}")