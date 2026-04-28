#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  python -m src.benchmark.summarize --latest
else
  python -m src.benchmark.summarize "$@"
fi
