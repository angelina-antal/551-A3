#!/bin/bash
# Full pipeline: preprocess → train → predict
set -e

DATA_DIR="${DATA_DIR:-datasets}"

echo "=== 1. Preprocess ==="
python -m src.preprocess --data-dir "$DATA_DIR"

echo ""
echo "=== 2. Train ==="
python -m src.nets --data-dir "$DATA_DIR"

echo ""
echo "=== Done ==="
echo "Outputs: best_model.pt, predicted_labels.csv"
