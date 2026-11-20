#!/usr/bin/env bash
set -euo pipefail

image=ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x@sha256:19627342e1da2607f4db50745dca30e57d7dd0ebff06062f03fd69b43a252931
model=unsloth/Qwen3.6-35B-A3B-NVFP4
revision=739af1e7aac320af1682ed1e0cce369af4c5265d
cache="${SPARK_HF_CACHE:-$HOME/.cache/spark-qwen36-35b/huggingface}"
snapshot="hub/models--${model//\//--}/snapshots/$revision"
name=spark-recipe-qwen36-vllm

case "${1:-}" in
  download)
    mkdir -p "$cache"
    docker pull --platform linux/arm64 "$image"
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download "$model" --revision "$revision" --cache-dir "$cache/hub"
    ;;
  start)
    test -f "$cache/$snapshot/config.json"
    docker run -d --name "$name" --network host --ipc host --gpus all \
      --mount "type=bind,source=$cache,target=/models,readonly" \
      --env HF_HUB_OFFLINE=1 --env HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
      "$image" serve "/models/$snapshot" \
      --served-model-name Qwen3.6-35B-A3B --host 127.0.0.1 --port 18891 \
      --tensor-parallel-size 1 --moe-backend auto --gpu-memory-utilization 0.80 \
      --linear-backend flashinfer_b12x --attention-backend flashinfer \
      --max-model-len 262144 --max-num-seqs 24 --max-num-batched-tokens 32768 \
      --enable-chunked-prefill --async-scheduling --kv-cache-dtype fp8 \
      --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}' \
      --reasoning-parser qwen3
    ;;
  stop)
    docker stop "$name"
    docker rm "$name"
    ;;
  *)
    echo "Usage: $0 {download|start|stop}" >&2
    exit 2
    ;;
esac
