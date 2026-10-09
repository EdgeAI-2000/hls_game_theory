#!/usr/bin/env bash
set -euo pipefail
cd /home-students/yu0666/majiang/majiang_ai/mahjong-ai-battle-main
mkdir -p DouZero_Sichuan_20260926/train_seed92601
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec py10/bin/python3 -u -m douzero_sichuan.train \
  --out DouZero_Sichuan_20260926/train_seed92601 \
  --device cuda:1 --workers 4 --frames 300000 \
  --games-per-update 32 --batch-size 3200 \
  --eval-every 50000 --eval-games 100 --seed 92601 \
  > DouZero_Sichuan_20260926/train_seed92601/console.log 2>&1
