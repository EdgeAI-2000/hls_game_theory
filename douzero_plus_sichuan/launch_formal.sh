#!/usr/bin/env bash
set -euo pipefail
cd /home-students/yu0666/majiang/majiang_ai/mahjong-ai-battle-main
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec py10/bin/python3 -u -m douzero_plus_sichuan.formal --device cuda:1 --workers 8 > DouZeroPlus_formal_20260927.log 2>&1
