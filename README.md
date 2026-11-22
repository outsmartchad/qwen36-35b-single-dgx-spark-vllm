# Qwen3.6 35B-A3B on one DGX Spark

vLLM serves Unsloth’s Qwen3.6 35B-A3B NVFP4 checkpoint with MTP ×2. The model API binds to `127.0.0.1:18891`.

[How it works](docs/how-it-works.md)

[Benchmarks](#benchmarks) · [Runtime](#runtime) · [Quick start](#quick-start) · [Raw results](results/benchmark-2026-11-21.json)

## Benchmarks

Measured November 21, 2026 on one NVIDIA GB10 DGX Spark. The benchmark sent direct streaming requests to the localhost model API. Other model servers were paused.

### Decode, prose

SparkDash hash-map prompt with thinking on, temperature 0.7, top-p 1, top-k 20 and 400 generated tokens per request. Each row shows the fastest aggregate decode wave across three repetitions. Per-request decode and TTFT are medians from the same wave. TTFT counts the first streamed reasoning or answer token.

| Streams | Best aggregate tok/s | Per request tok/s | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 85.4 | 85.4 | 101 ms |
| 2 | 128.4 | 64.6 | 170 ms |
| 3 | 143.5 | 49.0 | 209 ms |
| 4 | 197.8 | 49.9 | 207 ms |
| 5 | 257.0 | 51.9 | 859 ms |

### Fresh prompt prefill

The input rate divides prompt tokens, minus any reported cache hits, by client TTFT. The benchmark uses an independent synthetic key for each context size and compares the response with that key.

| Prompt tokens | Input tok/s to first token | TTFT | Retrieval |
| ---: | ---: | ---: | --- |
| 32,740 | 4,349 | 7.53 s | Exact key |

The 200K prefill attempt was skipped because available host memory fell below 8 GiB after the 32K run. Cache-hit counts were unavailable in the streaming response.

## Runtime

- Pinned Mia vLLM arm64 image and pinned [Unsloth NVFP4 weights](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4/tree/739af1e7aac320af1682ed1e0cce369af4c5265d).
- FlashInfer attention, FP8 KV, MTP ×2 with Triton for the draft MoE backend, chunked prefill, and asynchronous scheduling.
- The launcher requests the FlashInfer b12x linear backend. vLLM selected FlashInfer CUTLASS for NVFP4 linear layers and NVFP4 MoE. The GDN prefill path selected Triton/FLA.
- Configured 262,144-token model window, 24 maximum scheduled sequences and a 32,768-token batch limit. The vLLM memory target is 0.80.
- Peak GPU memory was 104.6 GiB across sampled inference-container processes. Docker reported no fixed container memory limit.

## Quick start

Use a single DGX Spark with Docker, NVIDIA Container Toolkit, the Hugging Face CLI, Python 3 and curl. Allow around 25 GiB for model files plus room for the runtime image. Stop any existing model server before running this recipe.

```bash
./serve.sh download
./serve.sh start
docker logs -f spark-recipe-qwen36-vllm
```

When the localhost API is ready, run the direct benchmark on the Spark.

```bash
curl -fsS http://127.0.0.1:18891/v1/models
export SPARK_BENCHMARK_URL=http://127.0.0.1:18891
export SPARK_BENCHMARK_MODEL=Qwen3.6-35B-A3B
python3 benchmark.py > decode-local.json
python3 benchmark.py 32768 2026144868 > prefill-32768-local.json
# Optional 200K run when at least 8 GiB remains available
python3 benchmark.py 200000 2026312100 > prefill-200000-local.json
./serve.sh stop
```

This recipe builds on [Mia AI Lab's pinned deployment](https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark/tree/79c9e6f359f6101cdacd0dfd6fe9861ae2493a4d). Our scripts, documentation and synthetic fixtures use the [MIT license](LICENSE). Upstream models and runtime components retain their own terms.
