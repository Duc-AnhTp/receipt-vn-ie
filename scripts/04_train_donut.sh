#!/usr/bin/env bash
set -euo pipefail

python -m receipt_ie.training.train_donut --mode finetune "$@"
