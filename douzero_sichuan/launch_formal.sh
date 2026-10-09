#!/usr/bin/env bash
set -euo pipefail
cd /home-students/yu0666/majiang/majiang_ai/mahjong-ai-battle-main
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec py10/bin/python3 -u -m douzero_sichuan.formal --workers 8 --device cuda:1
