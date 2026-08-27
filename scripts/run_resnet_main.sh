#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --config configs/resnet_cifar100_main.yaml \
  --output-root outputs/resnet_cifar100_main
