#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --config configs/mlp_twomoons_controls.yaml \
  --output-root outputs/mlp_twomoons_controls
