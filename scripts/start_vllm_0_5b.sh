#!/usr/bin/env bash
set -euo pipefail

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --enable-prefix-caching