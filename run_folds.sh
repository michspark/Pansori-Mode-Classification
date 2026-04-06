#!/bin/bash
set -e

#./run_folds.sh stratified        # stratified 3:1:1 split (single run)
#./run_folds.sh random            # 10 random folds (default)
#./run_folds.sh random 5          # 5 random folds
#./run_folds.sh stratified_half   # 10 half-stratified folds (2 per genre)

run_random_folds() {
    local FOLDS=${1:-10}   # default 10, override with: ./run_folds.sh random 5
    for i in $(seq 1 $FOLDS); do
        echo "========================================="
        echo " Starting Fold $i / $FOLDS"
        echo "========================================="
        python train.py data.split=random train.fold=$i
        echo "Fold $i done."
    done
    echo "All $FOLDS random folds complete."
}

run_stratified() {
    local FOLDS=${1:-5}
    for i in $(seq 1 $FOLDS); do
        echo "========================================="
        echo " Starting Stratified Fold $i / $FOLDS (3:1:1)"
        echo "========================================="
        python train.py data.split=stratified train.fold=$i
        echo "Fold $i done."
    done
    echo "All $FOLDS stratified folds complete."
}

run_stratified_half() {
    local FOLDS=${1:-10}
    for i in $(seq 1 $FOLDS); do
        echo "========================================="
        echo " Starting Half-Stratified Fold $i / $FOLDS"
        echo "========================================="
        python train.py data.split=stratified_half train.fold=$i
        echo "Fold $i done."
    done
    echo "All $FOLDS half-stratified folds complete."
}

MODE=${1:-random}

case "$MODE" in
    stratified)
        run_stratified "${2:-5}"
        ;;
    stratified_half)
        run_stratified_half "${2:-10}"
        ;;
    random)
        run_random_folds "${2:-10}"
        ;;
    *)
        echo "Usage: $0 [random [N_FOLDS] | stratified | stratified_half [N_FOLDS]]"
        echo "  random [N]          - run N random folds (default 10)"
        echo "  stratified [N]      - run N stratified folds with 3:1:1 ratio (default 5)"
        echo "  stratified_half [N] - run N half-stratified folds, 2 per genre (default 10)"
        exit 1
        ;;
esac
