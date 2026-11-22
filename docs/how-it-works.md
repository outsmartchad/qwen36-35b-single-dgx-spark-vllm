# How Qwen3.6 35B-A3B works on one DGX Spark

## How the model works

Qwen3.6 35B-A3B is a hybrid sparse MoE with 40 decoder layers, 30 Gated DeltaNet layers and 10 full-attention layers. The publisher config describes 256 routed experts with eight active per token, hidden size 2,048, 16 query heads, two KV heads, and a 262,144-token position limit. Gated DeltaNet carries a recurrent-style fixed-size state through most layers, while full attention supplies periodic exact token-to-token access. That mix reduces long-context KV growth relative to all-full attention while retaining global attention checkpoints. See the [publisher config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json) and the [Gated DeltaNet paper](https://openreview.net/pdf/3b2c0abfb1a7e49a0aec0aacedcce4ef8fbd620d.pdf).

Qwen3.6 35B-A3B is selected because the Unsloth NVFP4 checkpoint combines this 35B mixture-of-experts model with roughly 3B active parameters per token, making a long-context single-Spark deployment practical. The pinned target is [Unsloth Qwen3.6-35B-A3B-NVFP4 at `739af1e7`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4/tree/739af1e7aac320af1682ed1e0cce369af4c5265d). The NVFP4 designation applies to stored weights. The recipe separately selects FP8 KV, so weight and cache precision are distinct choices.

vLLM uses MTP with two speculative tokens and Triton for the draft MoE backend. MTP adds future-token prediction heads so several next tokens can be proposed before one verification pass. See the [MTP paper](https://arxiv.org/abs/2404.19737) and [vLLM speculative decoding documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/). Accepted proposals can reduce decode steps, while the gain varies with prompt and sampling. The launcher also requests FlashInfer attention and the `flashinfer_b12x` linear backend, enables chunked prefill and asynchronous scheduling, and uses the Qwen3 reasoning parser. See the [FlashInfer project](https://github.com/flashinfer-ai/flashinfer), the [Qwen model card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4), and the [pinned upstream deployment](https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark/tree/79c9e6f359f6101cdacd0dfd6fe9861ae2493a4d6).

## How this recipe runs it on one DGX Spark

`serve.sh download` pulls the digest-pinned Mia vLLM ARM64 image and the immutable model snapshot. `serve.sh start` mounts the cache read-only and starts one vLLM worker on `127.0.0.1:18891`. The client posts to the OpenAI-compatible endpoint. vLLM applies the Qwen chat and reasoning parser, schedules chunked prefill and async decode, runs the NVFP4 MoE target with FlashInfer kernels, and verifies MTP proposals before streaming SSE chunks.

The profile requests a 262,144-token context, 24 maximum scheduled sequences, a 32,768-token batch limit, FP8 KV, and `0.80` GPU-memory utilization. A 32,740-token fresh prompt reached its exact key at 4,349 input tok/s with 7.53 s TTFT. Decode measured 85.4 tok/s for one stream and 257.0 tok/s aggregate for five. The planned 200K prefill was skipped after host memory fell below 8 GiB following the 32K run, so no 200K capacity claim is made. Peak sampled GPU-process memory was 104.6 GiB. See [`serve.sh`](../serve.sh), [`benchmark.py`](../benchmark.py), and the [raw results](../results/benchmark-2026-11-21.json).

The request path is

```text
client on Spark
  -> localhost:18891 OpenAI API
  -> vLLM parser, async scheduler, chunked prefill, and FP8 KV cache
  -> Qwen3.6 NVFP4 MoE target plus two-token MTP verification
  -> streamed reasoning or answer
```

The decode benchmark uses thinking on, temperature `0.7`, top-p `1`, top-k `20`, and forced 400-token waves. KV capacity, prompt length, MTP acceptance, and host-memory pressure all affect practical limits. The observed numbers are from the November 21, 2026 run and should be reproduced after changing image, model revision, or launch flags.
