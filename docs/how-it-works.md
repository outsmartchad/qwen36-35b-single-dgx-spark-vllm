# How Qwen3.6 35B-A3B runs on one DGX Spark

## How the model works

A document might introduce a code such as `AB-17` on its first page. A question much later asks for that code. Qwen3.6 gives different layers different ways to connect the question to its source. Thirty layers compress what they have read into a changing state. Ten can inspect individual positions across the preceding context. At every layer a router chooses a small set of feed-forward networks to transform the result. The model name summarizes the scale of the language backbone. Qwen reports **35 billion weights in total and about 3 billion active per token**. [Qwen model card][base] · [Exact served configuration][cfg]

### A forward pass starts with one vector per token

The client sends role-tagged messages. A chat template supplies the delimiters expected by the model, and a tokenizer splits that text into pieces called **tokens** and represents each piece as an integer ID. The model has 248,320 padded embedding rows. Looking up an ID returns 2,048 numbers, its initial embedding. A code, an emoji or a long word can take several IDs. A context window counts positions in the resulting sequence, including conversation markers, earlier turns and generated output. [Model dimensions and vocabulary][cfg] · [Model card][base]

An embedding-table entry is a **weight** learned in training. An **activation** is a value computed for the current request, such as the vector representing the second occurrence of a code after it has passed through a few layers. Pretraining taught the model to anticipate subsequent tokens. Post-training shaped how it responds to instructions. A training step can revise weights using prediction errors. An inference step on the Spark uses those weights to calculate new activations and request state. A document you supply can change a prediction immediately because it is in context. [Qwen training overview][base]

```text
text → IDs → embedding vectors of width 2,048
            │
            └→ 10 groups, each with four decoder layers
                 Gated DeltaNet → experts
                 Gated DeltaNet → experts
                 Gated DeltaNet → experts
                 full gated attention → experts
            → final hidden state → normalization and output head → next token
                      └──────────→ native MTP takes token input to propose following IDs
```

In each block, normalization rescales the current features. The sequence-mixing calculation produces an update that is added back to the input, a **residual connection**. Another normalization and a feed-forward network produce a second update. These additions carry earlier features through the depth of the network while allowing each layer to refine them. The checkpoint's class name begins `Qwen3_5Moe`. That is the Transformers implementation it declares for these Qwen3.6 weights, useful when locating the code for a specific operation. [Pinned config][cfg] · [Transformers model implementation][model]

### How Gated DeltaNet updates its memory

Think of one head's state as a small matrix that maps a key to an associated value. For a new token, learned projections produce a query, key and value and a causal convolution of width four mixes nearby features. Another projection determines how much of the previous state to keep. The layer reads from the decayed state with the new key and compares the predicted value with the incoming value. It writes a gated fraction of their difference back along that key. The query reads from the updated state, and a gated output projection returns information to the residual stream. [Model recurrence][model] · [Gated DeltaNet paper][gdn]

For intuition, suppose a feature associated with a job ID previously predicted “pending” and the next log entry says “complete.” A delta update can correct the association where its current prediction is wrong. Real keys and values are continuous learned vectors, so several associations can overlap and interfere. The layer keeps the resulting matrix and a short convolution history as each new token arrives. This representation compresses the history into the current state. Qwen configures 16 query/key heads and 32 value heads at width 128, with FP32 recurrent state specified in the checkpoint. The state shape stays fixed as a single sequence grows. [Linear attention dimensions and state type][cfg] · [Gated update design][gdn]

### How full attention looks up a specific earlier position

Layer four of each group gives every position sixteen queries of width 256 and two key/value heads. A query scores visible keys by similarity, normalizes those scores with softmax, and reads the corresponding weighted mixture of values. Eight query heads share each pair of key and value heads. This **grouped-query attention** lets different queries do their own matching while the cache stores fewer key and value vectors per token. The model normalizes queries and keys, applies rotary position embeddings to 64 out of 256 dimensions, and gates the attention output. The rotary transformation rotates pairs by position, making relative distance available to the comparison. [Checkpoint dimensions][cfg] · [Attention forward path][model]

At position 30,000 the causal mask exposes positions 1 through 30,000. Position 30,001 remains invisible even when training or prefill processes multiple rows together. If the code appears at position 100, a full-attention head can compare its key directly against the question's query. Gated DeltaNet can carry a signal about that code in a compressed state across the same span. These two kinds of history have different costs. Full attention retains token-indexed keys and values over the sequence. The other thirty layers keep their state matrices. A server continuing a request needs both the right attention prefix and the corresponding recurrent state. [Model layout][base] · [Hybrid cache and attention code][model]

### What the experts contribute and how the answer emerges

At each of 40 layers the router projects the current token vector into 256 expert scores, applies softmax, selects eight and normalizes their weights within that set. Each selected expert runs a gated feed-forward transformation with intermediate width 512. Its gate projection modulates features from an up projection, and a down projection maps the result back to width 2,048. The weighted sum of these outputs is joined by a shared expert whose output has its own sigmoid gate. Nearby tokens can follow different routes, and index 7 in one layer means different matrices from index 7 in the next. Each token thus triggers 320 routed selections across the main stack. The full bank still needs storage. [Router, gated experts and shared path][model] · [Checkpoint model overview][base]

After the final normalization, the output head produces a **logit** for every vocabulary row. These are raw numerical scores. Greedy decoding chooses the largest. For sampled prose in this recipe, temperature 0.7 sharpens the distribution and top-k 20 restricts it to the twenty highest candidates. Top-p 1 leaves that candidate set unrestricted by a further cumulative-probability cutoff. The selected ID is decoded into text as part of the growing output, and its embedding can enter the next step. An end-of-sequence token or an output limit can finish the response. [Generation parameters](../benchmark.py) · [Model output head][model]

The checkpoint also contains a vision encoder. It groups input pixels into patches with spatial size 16 and temporal size two, processes visual features through 27 layers of width 1,152, and merges neighboring two-by-two spatial features into representations of language width 2,048. Its architecture supplies position information for multimodal inputs. The benchmark sent text-only requests, so no published throughput or retrieval measurement here includes image preprocessing or visual encoding. [Vision configuration][cfg] · [Vision transformer and merger][model] · [Measured workloads](../benchmark.py)

## How this recipe runs it on one DGX Spark

### Load the mixed-precision checkpoint

```bash
./serve.sh download
./serve.sh start
docker logs -f spark-recipe-qwen36-vllm
# From a second terminal on the Spark
curl -fsS http://127.0.0.1:18891/v1/models
```

The download pulls the Mia vLLM ARM64 image by digest and a single immutable Unsloth checkpoint. The checkpoint already contains the multi-token prediction module. Allow around 25 GiB of disk for model files and additional room for the image. The launcher verifies the config exists, mounts the snapshot cache read-only, selects offline Hub mode and starts one tensor-parallel rank. The OpenAI-compatible model name is `Qwen3.6-35B-A3B`. The loopback address makes the endpoint available to clients running on the Spark. [Launcher](../serve.sh) · [Pinned upstream deployment][deploy] · [Unsloth checkpoint card][quant]

Unsloth calls the export NVFP4. Its actual quantization rules are mixed. Most routed and shared expert projections use packed four-bit floating-point values in groups of 16 with FP8 scales. The more specific group for the final eight decoder layers names their expert projections for FP8. The config also targets attention projections and the output head for FP8. The vision blocks and native MTP namespace appear among the exclusions. Calibration determines useful numerical ranges for the compressed operators. Compression reduces weight storage and bytes read at inference, while the separately requested FP8 KV cache reduces the size of attention history. [Exact targets, exclusions and formats][cfg] · [Calibration details][quant]

When a client posts to `/v1/chat/completions`, the engine formats and tokenizes the messages, runs the scheduler, and streams reasoning or answer chunks through the Qwen3 parser. A backend names the implementation of a mathematical operation. The launch requests FlashInfer attention and the `flashinfer_b12x` linear backend, with automatic target MoE selection. The recorded startup selected FlashInfer CUTLASS for NVFP4 linear and expert operations and Triton/FLA for **GDN prefill**. The Triton MoE setting inside the speculative configuration belongs to the drafter. Distinguishing these paths matters when reproducing this GB10 run. [Launch flags](../serve.sh) · [Backend observations](../README.md#runtime) · [FlashInfer project](https://github.com/flashinfer-ai/flashinfer)

### Prefill, two-token speculation and the memory budget

During **prefill**, many input rows are already known. vLLM can process them in parallel under the causal mask while writing the attention history and advancing recurrent states. The launch enables chunked prefill, allowing the scheduler to divide a large prompt across iterations. Asynchronous scheduling overlaps preparation of eligible work with execution. The 32,768-token batch limit measures scheduled tokens in an iteration. The 24-sequence limit caps the number of scheduled conversations, and the 262,144-token context caps each sequence. A 100,000-token prompt can require multiple prefill iterations even if it is the only request. [Scheduler settings](../serve.sh)

**Decode** starts with a target prediction and faces a sequential dependency between successive IDs. Qwen's one-layer multi-token prediction module provides a cheaper way to propose two successors. In vLLM's implementation it normalizes a recent hidden state and the latest token embedding, joins and projects them, then runs a prediction layer. Reusing that layer advances the draft another position. The full target verifies the proposed chain in a batch. If the first candidate passes and the second fails, the engine can keep the first, use a correction at the second and continue from the committed state. A longer accepted prefix amortizes an expensive target pass. The published end-to-end measurements combine MTP, routing and kernel choices. [Native MTP config][cfg] · [vLLM prediction layer][mtp] · [Multi-token training](https://arxiv.org/abs/2404.19737) · [vLLM speculative decoding][spec]

A 0.80 GPU-memory utilization target guides vLLM's planner. It must fit weights, caches, recurrent states, speculative work and temporary buffers within the available memory. For scale, ten full-attention layers, two KV heads, 256 features per head, K and V, and one byte per FP8 scalar imply **10,240 raw KV bytes per retained position**. One 262,144-position sequence would therefore require about 2.5 GiB for that payload alone. This calculation excludes recurrent state, scales, cache-block rounding and execution buffers. The Spark's Grace CPU and Blackwell GPU share 128 GB of physical memory, so available host memory also matters while a server is running. [Model sizes][cfg] · [Launch profile](../serve.sh) · [DGX Spark specifications][spark]

On November 21 the server returned the exact key from a fresh **32,740-token** document after **7.53 seconds**, or **4,349 input tok/s to first token**. The response did not expose the cache-hit count. Available host memory subsequently fell below 8 GiB, and the planned 200K test was skipped. The sampled GPU-process peak was **104.6 GiB**, with no fixed Docker memory cap. The configuration admits the 262,144 position limit as a setting. The completed long-prompt evidence here reaches 32,740. [Runtime and test scope](../README.md#benchmarks) · [Raw results][raw]

The separate short prose experiment requested 400 output tokens per stream with thinking enabled. The fastest of three waves reached **85.4 output tok/s** for one stream and **257.0 aggregate tok/s** for five. In the latter wave the median individual stream delivered 51.9 tok/s and its median first-token wait was 859 ms. Aggregate throughput describes completed output across users. Per-request rate describes the generation a user sees after the first chunk, while the fresh-prompt test measures the wait to reach that first chunk at a very different input length. [Protocol and per-stream results](../README.md#benchmarks) · [Raw decode waves][raw]

[base]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/345d467b117d1fd5eda9b879a5988de9ae5f47a7/README.md
[cfg]: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4/blob/739af1e7aac320af1682ed1e0cce369af4c5265d/config.json
[quant]: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4/blob/739af1e7aac320af1682ed1e0cce369af4c5265d/README.md
[model]: https://github.com/huggingface/transformers/blob/2f1bc5e321b9a736200a590d6e7743ccdeb60772/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
[gdn]: https://arxiv.org/abs/2412.06464
[mtp]: https://github.com/vllm-project/vllm/blob/e72fc9ae50c221180c4b8f4a94c5009714a4b199/vllm/model_executor/models/qwen3_5_mtp.py
[spec]: https://docs.vllm.ai/en/latest/features/speculative_decoding/
[deploy]: https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark/tree/79c9e6f359f6101cdacd0dfd6fe9861ae2493a4d
[spark]: https://www.nvidia.com/en-us/products/workstations/dgx-spark/#specifications
[raw]: ../results/benchmark-2026-11-21.json
