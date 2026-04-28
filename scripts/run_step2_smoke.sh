#!/usr/bin/env bash
set -euo pipefail

# Smoke run：20 measured requests，1 req/s，验证链路
python -m src.benchmark.runner \
  --model-config configs/model.yaml \
  --workload-config configs/workload.yaml \
  --benchmark-config configs/benchmark.yaml \
  --override warmup_requests=5 \
  --override measured_requests=20 \
  --override "arrival.request_rate=1.0"