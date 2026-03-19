#!/bin/bash
set -e

FOLDS=${1:-10}   # default 10, override with: ./run_folds.sh 5

for i in $(seq 1 $FOLDS); do
    echo "========================================="
    echo " Starting Fold $i / $FOLDS"
    echo "========================================="
    python train.py train.fold=$i
    echo "Fold $i done."
done

echo "All folds complete."
