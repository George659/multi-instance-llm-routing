# Step 2 Benchmark Guide

## Overview

Step 2 builds a reusable single-instance benchmark harness for OpenAI-compatible
LLM serving. In this repository, the harness targets local vLLM first, but the
client and metrics path are intentionally decoupled from vLLM internals so the
same tooling can later benchmark a router endpoint.

Core capabilities:

- Generate `chat`, `rag`, and `agent` synthetic workloads.
- Support `constant`, `poisson`, and `burst` arrivals through a unified
  `effective_rate` interface.
- Send streaming chat-completions requests and record TTFT, e2e latency, TPOT,
  usage tokens, and structured failures.
- Persist per-request CSV, run-level summary JSON, and an environment snapshot
  keyed by `run_id`.

## Files

- `configs/model.yaml`: endpoint, model name, tokenizer name, timeout.
- `configs/workload.yaml`: workload mix and token-length controls.
- `configs/benchmark.yaml`: warmup, measured samples, arrival mode, timeout,
  and output path templates.
- `scripts/start_vllm_0_5b.sh`: local vLLM startup script.
- `scripts/run_step2_smoke.sh`: quick smoke benchmark with CLI overrides.
- `scripts/summarize_step2.sh`: recompute summary from the latest result CSV.
- `src/benchmark/runner.py`: main benchmark entrypoint.
- `src/benchmark/summarize.py`: offline summary CLI.

## Reproduce

### 1. Start vLLM

```bash
bash scripts/start_vllm_0_5b.sh
```

In a separate terminal, verify the endpoint is reachable:

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/models
```

### 2. Run a smoke benchmark

```bash
bash scripts/run_step2_smoke.sh
```

This smoke run uses:

- `warmup_requests=5`
- `measured_requests=20`
- `arrival.request_rate=1.0`

Expected behavior:

- warmup requests are executed but excluded from summary statistics
- `summary_json` shows `20 success / 0 failed` when the endpoint is healthy
- `p95` and `p99` stay `null` because the smoke sample size is below the
  configured reliability thresholds

### 3. Run the standard baseline

```bash
python -m src.benchmark.runner \
  --model-config configs/model.yaml \
  --workload-config configs/workload.yaml \
  --benchmark-config configs/benchmark.yaml
```

The default benchmark config uses:

- `warmup_requests=50`
- `measured_requests=1000`
- `arrival.mode=constant`
- `arrival.request_rate=2.0`

### 4. Summarize results

Recompute summary for the latest run:

```bash
python -m src.benchmark.summarize --latest
```

Or use the wrapper script:

```bash
bash scripts/summarize_step2.sh
```

To summarize a specific run, replace the placeholder with the real run id:

```bash
python -m src.benchmark.summarize \
  --input experiments/results/20260428_041804_8fb1d25-dirty_cfg298fdb96_results.csv \
  --output experiments/results/20260428_041804_8fb1d25-dirty_cfg298fdb96_summary.json
```

`<run_id>` in design docs is a placeholder only. Do not type angle brackets
literally in bash commands.

### 5. Run tests

```bash
pytest -q
```

## Output Layout

Each benchmark run generates three artifacts:

- `experiments/results/<run_id>_results.csv`
- `experiments/results/<run_id>_summary.json`
- `experiments/logs/<run_id>_env.json`

`run_id` format:

```text
YYYYMMDD_HHMMSS_<git_short_hash>_<config_hash>
```

Example:

```text
20260428_041804_8fb1d25-dirty_cfg298fdb96
```

## Metrics And Statistical Conventions

### Per-request CSV

Important CSV fields:

- `status`: `success` or `failed`
- `error_type`: populated for failed requests
- `queue_delay_sec`: actual send time minus scheduled send time
- `ttft_sec`: time from request send to first output token
- `e2e_latency_sec`: time from request send to stream completion
- `mean_tpot_sec` and `p95_tpot_sec`: derived from locally observed
  inter-token arrival gaps
- `prompt_tokens`, `completion_tokens`, `total_tokens`: from streaming usage
- `num_output_tokens_observed`: local fallback based on token chunk count
- `prefix_group_id`: RAG shared-prefix group, or empty for non-shared requests
- `effective_rate`: normalized request rate for cross-arrival comparisons

`token_arrival_times` are intentionally not written into CSV because they are
too large for the default output format.

### Summary JSON

The summary includes:

- sample counts
- latency distributions for TTFT and e2e latency
- TPOT distribution
- throughput
- grouped summaries by workload type and prefix group
- reliability flags for percentile interpretation

Reliability policy:

- `p95_reliable = success_count >= min_samples_for_p95`
- `p99_reliable = success_count >= min_samples_for_p99`
- if a percentile is not reliable, the corresponding field is emitted as `null`
  and a warning string is attached

This is why smoke runs produce valid means but `null` for `p95` and `p99`.

### Warmup Policy

Warmup requests are executed before the measured run and are excluded from:

- result CSV
- summary JSON
- reported sample counts

## Environment Snapshot

The runner writes `<run_id>_env.json` before sending benchmark traffic. This
snapshot is the run-level baseline for reproducibility and debugging.

It records:

- `run_id`
- UTC timestamp
- git branch, short commit hash, and dirty flag
- Python version
- package versions for `vllm`, `torch`, `transformers`, and `httpx`
- CUDA driver and visible GPU information
- hostname, platform, and current working directory
- full normalized copies of model/workload/benchmark configs

Use this file when comparing two runs to determine whether a metric change came
from code, config, or environment drift.

## Why Not `vllm bench` / `benchmark_serving.py`

We intentionally use a custom benchmark harness instead of depending directly
on vLLM's built-in benchmarking entrypoints.

Reasons:

- The long-term target is router evaluation, not only single-instance vLLM.
- We need one request schema shared by generator, client, metrics, and future
  router experiments.
- We need token-level shared-prefix construction and validation for RAG
  locality experiments.
- We need per-token arrival timestamps to compute TPOT locally.
- We need explicit reliability flags and stable output files for repeated
  experiments and interviews/demos.

vLLM's benchmark utilities are useful for quick serving checks, but they are
not the right ownership boundary for the Step 2 experiment substrate.

## Notes And Common Pitfalls

- If requests all fail with `connection_error`, verify vLLM is really listening
  on `127.0.0.1:8000` and re-run the `curl --noproxy '*'` check.
- If you use a shell proxy, `curl` may accidentally hit a local proxy port
  instead of the model endpoint; `--noproxy '*'` avoids that confusion.
- If a run hits `max_run_duration_sec`, unfinished requests are still written to
  the CSV as `failed` with `error_type=timeout`, so the measured result set
  remains complete.
