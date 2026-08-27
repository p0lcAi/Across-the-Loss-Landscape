#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --config configs/resnet_cifar100_patience_trainloss_s10.yaml \
  --output-root outputs/resnet_cifar100_patience_trainloss_s10
