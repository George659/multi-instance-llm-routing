# PD Router Project

Prefix-Locality-Aware Routing and Workload-Adaptive Scheduling for Multi-Instance
LLM Serving.

## Goal

This project studies routing and scheduling policies for multi-instance vLLM
serving. The objective is not to reimplement vLLM itself, but to build an
experiment platform above the serving layer so we can evaluate:

- baseline routing policies
- prefix-locality-aware routing
- workload-adaptive scheduling
- reproducible benchmark workflows for future multi-instance experiments

## Current Status

- [x] Local environment setup
- [x] vLLM smoke test
- [x] Workload generator
- [x] Benchmark harness
- [ ] Router baseline policies
- [ ] Prefix-locality-aware routing
- [ ] Cloud multi-instance experiments

Step 2 is now in place: the repository includes a reusable single-instance
benchmark harness that can generate synthetic workloads, drive a local
OpenAI-compatible endpoint, and persist structured experiment outputs.

## Step 2 Highlights

- Generate `chat`, `rag`, and `agent` synthetic workloads.
- Support `constant`, `poisson`, and `burst` arrivals.
- Send streaming chat-completions requests to an OpenAI-compatible endpoint.
- Record TTFT, e2e latency, TPOT, token usage, and structured failures.
- Write per-request CSV, run-level summary JSON, and `<run_id>_env.json`.
- Mark unreliable percentiles explicitly when sample size is too small.

## Repository Layout

```text
configs/                 Step 2 benchmark configs
docs/                    setup notes, benchmark guide, risk log
scripts/                 vLLM startup, smoke run, summary helpers
src/workload/            workload schema, generators, arrival patterns
src/benchmark/           client, metrics, runner, env snapshot, summarize
tests/                   generator, arrival, prefix alignment, runner tests
```

## Quick Start

### 1. Start local vLLM

```bash
bash scripts/start_vllm_0_5b.sh
```

### 2. Verify the endpoint

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/models
```

### 3. Run the smoke benchmark

```bash
bash scripts/run_step2_smoke.sh
```

This runs a small benchmark with:

- `warmup_requests=5`
- `measured_requests=20`
- `arrival.request_rate=1.0`

### 4. Run the standard baseline

```bash
python -m src.benchmark.runner \
  --model-config configs/model.yaml \
  --workload-config configs/workload.yaml \
  --benchmark-config configs/benchmark.yaml
```

### 5. Recompute summary

```bash
bash scripts/summarize_step2.sh
```

## Outputs

Each run produces:

- `experiments/results/<run_id>_results.csv`
- `experiments/results/<run_id>_summary.json`
- `experiments/logs/<run_id>_env.json`

`run_id` is composed from:

```text
timestamp + git_short_hash + config_hash
```

This makes it easy to compare runs by code version, configuration, and time.

## Tests

```bash
pytest -q
```

Current test coverage includes:

- workload generation behavior
- arrival pattern logic
- token-level shared-prefix alignment
- runner timeout behavior

## Documentation

- [Step 2 Benchmark Guide](docs/step2_benchmark.md)
- [Environment Setup](docs/env_setup.md)
- [Risk Log](docs/risk_log.md)

## Hardware Assumption

- Local: WSL2 + Conda + RTX 5060
- Cloud: 2 x 24GB or 2 x 48GB GPU for final experiments
