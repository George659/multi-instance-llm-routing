# Environment Setup Notes

## System

- OS: WSL2
- GPU: RTX 5060
- Python env manager: Anaconda
- CUDA: TBD
- Driver: TBD

## Conda Environment

Environment name:
pd-router

PyTorch install command:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

vLLM install command:
```bash
pip install vllm
```

start the server command:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.75
```