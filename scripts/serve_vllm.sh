#!/bin/bash
# Serve a VLM on a local vLLM OpenAI-compatible endpoint.
# usage: serve_vllm.sh <MODEL> <GPU> <PORT>
# Set VLLM_BIN if vllm isn't on your PATH (e.g. a conda env's bin/vllm).
MODEL="${1:?model}"; GPU="${2:-0}"; PORT="${3:-8000}"
VLLM_BIN="${VLLM_BIN:-vllm}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$VLLM_BIN" serve "$MODEL" \
    --tensor-parallel-size 1 --port "$PORT" --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 32}'
