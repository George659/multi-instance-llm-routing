# PD Router Project

Prefix-Locality-Aware Routing and Workload-Adaptive Scheduling for Multi-Instance LLM Serving.

## Goal

This project studies routing and scheduling policies for multi-instance vLLM serving.  
The main focus is not to reimplement vLLM, but to build a deployment-layer experiment platform on top of vLLM.

## Current Status

- [ ] Local environment setup
- [ ] vLLM smoke test
- [ ] Workload generator
- [ ] Benchmark harness
- [ ] Router baseline policies
- [ ] Prefix-locality-aware routing
- [ ] Cloud multi-instance experiments

## Hardware Assumption

- Local: WSL2 + Conda + RTX 5060
- Cloud: 2 x 24GB or 2 x 48GB GPU for final experiments