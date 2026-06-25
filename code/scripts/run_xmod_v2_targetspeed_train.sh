#!/usr/bin/env bash
set -euo pipefail
BASE="/mnt/c/xmod_b2d/m1"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG="$BASE/logs/train_xmod_v2_targetspeed_c0_${RUN_TAG}.log"
CKPT="$BASE/checkpoints/xmod_v2_targetspeed_routev4_c0_8456.pt"
DATA_ROOT="$BASE/xmod_dataset_scale_c0_existing_broad_plus_dagger_refonly_20260620_0055"
PY="/home/lloitesa/miniconda3/envs/lead/bin/python"
mkdir -p "$BASE/logs" "$BASE/checkpoints"
echo "LOG=$LOG"
echo "CKPT=$CKPT"
# X-MoD v2: target-speed action head (fresh, no init-ckpt). --lambda-moving 0 is REQUIRED
# (the moving regularizer assumes raw-pedal columns; it corrupts target_speed semantics).
env CUDA_VISIBLE_DEVICES=0 "$PY" "$BASE/train_xmod_m1.py" \
  --data-root "$DATA_ROOT" \
  --out-ckpt "$CKPT" \
  --epochs 20 \
  --batch-size 48 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --sampler-mode route \
  --ego-mode route_v4 \
  --model-arch targetspeed \
  --lambda-moving 0 \
  --cache \
  2>&1 | tee "$LOG"
echo "TRAIN_DONE ckpt=$CKPT"
