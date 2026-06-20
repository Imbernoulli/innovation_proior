## Research question

Serve a **fixed autoregressive LLM on a fixed set of GPUs at the highest possible throughput — generated
tokens per second — without raising per-request latency past a fixed budget.** Everything about the serving
problem that *defines* the task is frozen: the model weights and architecture are given, the hardware is a
given GPU (or fixed multi-GPU group), and the service-level target is a fixed tail latency that requests must
still meet. The single free variable is *the serving system itself* — how the engine batches requests, lays
out the per-request key/value cache in memory, schedules prefill and decode work onto the GPU, and stores the
cache numerically. A "better" rung is one that pushes the throughput-vs-latency frontier: it serves more
tokens per second at the same latency, or holds throughput while cutting latency, by spending the fixed GPU
more efficiently — chiefly by fitting **more concurrent requests into the same memory** so each forward pass
amortizes its fixed cost over a larger batch.

The reference point that anchors the exercise: a straightforward serving loop built on a standard
transformer-inference library, which runs each batch of requests through the model with the usual
preallocated cache and request-level batching. That loop leaves the GPU badly underutilized — the batch size
it can sustain is far below what the raw compute could feed — and the ladder built here is the sequence of
serving-system changes that close that gap, each one removing whatever is the current binding constraint on
how many requests fit and how densely the GPU's time is packed.

## Background

The workload is autoregressive transformer inference. Each request runs in two regimes: a **prefill** pass
that ingests the whole prompt at once (compute-bound, one big matmul over all prompt positions), then a
**decode** phase that emits one token per forward pass, each step attending over all previous tokens
(memory-bandwidth-bound, tiny per-step compute). To avoid recomputing attention keys and values for the
whole history every step, every serving system keeps a **KV cache**: for each request it stores the key and
value tensors for every past token, at every layer and head, and the decode step reads that cache and appends
one token's worth of new entries.

The load-bearing facts about this cache and this workload, all observable on existing serving systems:

- **The KV cache is enormous and grows token-by-token.** For a multi-billion-parameter model the per-token
  KV footprint is on the order of hundreds of kilobytes across all layers; a single request with a long
  context occupies a large slice of GPU memory, and that slice *grows* as the request generates. The cache,
  not the model weights, is what limits how many requests can be served at once.

- **Throughput at fixed latency is set by batch size, and batch size is set by memory.** Decode is
  bandwidth-bound: one decode forward pass over a batch of B requests costs almost the same wall-clock as
  over a batch of 1, because the dominant cost is streaming the weights and the cache, not the per-request
  arithmetic. So tokens-per-second scales nearly linearly with B — *until* memory runs out. The question
  "how do I serve faster at the same latency" is therefore largely the question "how do I fit more requests'
  KV caches into the same GPU memory."

- **Requests have wildly different and unpredictable lengths.** Prompt lengths and generation lengths vary by
  orders of magnitude across a real request stream (e.g. ShareGPT-like traffic), and a request's final length
  is not known when it arrives — generation stops at an end token or a length cap that is hit at runtime.

- **Standard serving reserves cache memory in one contiguous chunk per request, sized to the maximum.**
  Because the cache must grow and the system wants the per-request cache contiguous for fast kernel access,
  the common approach preallocates, per active request, a contiguous block of memory sized to the model's
  maximum sequence length. A request that ends up short still holds the whole reservation while it is alive;
  the unused tail is dead memory. Measurements on such systems show that only a small fraction — on the
  order of 20–40% — of KV-cache memory holds live tokens; the rest, roughly 60–80%, is wasted to internal
  over-reservation and to fragmentation between the variable-sized contiguous chunks. That waste directly
  caps the batch size, and through it, throughput.

- **Requests in a stream share content.** Many requests in real traffic begin with the *same* tokens — a
  shared system prompt, a few-shot preamble, a common instruction header, a branching sampler that forks one
  prompt into several continuations. Under per-request contiguous caches, each of these stores its own private
  copy of the identical prefix's keys and values, duplicating memory that is bit-for-bit the same.

- **Prefill and decode interfere.** A long prompt's prefill is one large compute-bound burst; the decodes of
  other in-flight requests are many small bandwidth-bound steps. Run naively, a long prefill monopolizes a
  forward pass and stalls everyone else's decoding (a latency spike), while pure-decode passes leave the GPU's
  compute units idle.

- **The cache is stored at full activation precision.** KV entries are kept in the model's activation dtype
  (FP16/BF16, two bytes per element). The number of bytes per cached token is fixed by that choice, and the
  cache's memory footprint — hence the achievable batch size — scales directly with it.

## Baselines

The systems and techniques a faster server is measured against, with what each does and where it stalls:

- **HuggingFace Transformers generation loop.** The reference implementation of autoregressive generation:
  load the model, batch a set of prompts, call `model.generate`, keep a per-request KV cache preallocated for
  the batch. It is correct and ubiquitous but throughput-naive — it batches at the granularity of a whole
  `generate` call (the batch is fixed for the call's duration), and its cache management is the
  contiguous-per-request style above. **Gap:** the sustainable batch size is small because so much KV memory
  is reserved-but-unused, and a batch cannot release a finished request or admit a new one mid-generation, so
  short requests sit idle waiting for the longest one in their batch to finish.

- **FasterTransformer.** A heavily hand-optimized inference engine with fast fused transformer kernels,
  the performance reference for raw per-pass speed. Its kernels are excellent, but its request and memory
  management inherit the contiguous-reservation model. **Gap:** the kernels are fast but the system still
  reserves cache per request up to a max length and batches coarsely, so under a stream of variable-length
  requests the GPU runs at low effective batch occupancy — fast kernels feeding a small batch.

- **Orca.** A serving system that introduced **iteration-level** request handling: rather than fixing a batch
  for the duration of a whole generation, it makes scheduling decisions at the granularity of model
  iterations, so requests can in principle join and leave the running set between steps. This is the key
  advance over call-level batching. **Gap:** Orca attacks the *scheduling* axis but still manages the KV
  cache as per-request memory in the contiguous style, so even with finer-grained scheduling the memory waste
  caps how many requests can actually be co-resident, and the iteration-level scheduler is bottlenecked by
  the same fragmentation it sits on top of.

- **The single lever everyone reaches for: just batch more requests.** The obvious throughput knob is to
  raise the batch size. But under contiguous per-request reservation, raising the configured batch size means
  reserving max-length cache for that many requests, which overflows memory long before the GPU's compute is
  saturated — or, if the reservation is sized down, risks running out of cache mid-generation and having to
  evict or crash a request. **Gap:** batch size is the right target, but it is gated by a memory-management
  scheme that ties up far more memory than the live tokens need; you cannot turn the batch-size knob up
  without first changing *how the cache is stored*.

## Evaluation settings

The workload is a stream of generation requests with prompt and output lengths drawn from a realistic
distribution (e.g. requests sampled from ShareGPT-style conversation logs, mixing short and long prompts and
generations). The fixed model is a standard decoder-only LLM (the LLaMA family is the canonical test case:
LLaMA-7B on a single mid-range datacenter GPU such as an A10G, LLaMA-13B on a single A100-40GB), served on a
fixed GPU or fixed multi-GPU tensor-parallel group. The primary metric is **serving throughput — generated
(or total) tokens per second**, reported **at a fixed latency / SLO** so that throughput is not bought by
letting latency blow up; secondary views are requests-per-second at fixed latency and the
latency-vs-throughput curve. The natural in-repo yardsticks are an offline **throughput benchmark** (feed a
fixed set of prompts, measure tokens/sec at saturation) and a **prefix-sharing benchmark** (a workload where
many requests share a common prefix). Each serving change is judged on whether it moves the
throughput-at-fixed-latency frontier for this fixed model on this fixed hardware.

## Code framework

The serving engine is a Python loop around the fixed model and a GPU KV-cache region. The pieces that already
exist before any of the ladder's methods: the model forward, an attention kernel, a tensor of cache memory,
and a loop that pulls requests and runs steps. The slots the methods will fill are left empty.

```python
import torch

# ---- given, fixed ----
model = load_decoder_only_llm(model_id)          # frozen weights + architecture
device = "cuda"                                  # fixed GPU(s)

# ---- the GPU KV-cache region (how it is laid out is a design choice) ----
class KVCacheStore:
    """Holds keys/values for all in-flight requests on the GPU."""
    def __init__(self, num_layers, num_heads, head_size, dtype, total_memory):
        # TODO: how cache memory is organized and addressed per request.
        pass

    def reserve_for(self, request):
        # TODO: make room for this request's keys/values.
        pass

    def free(self, request):
        # TODO: release this request's cache.
        pass


def attention(query, key_cache, value_cache, request_layout):
    # Reads each request's cached keys/values and attends over them.
    # TODO: how the kernel locates a request's KV entries in the store.
    pass


# ---- the engine loop ----
class LLMEngine:
    def __init__(self, model, kv_store):
        self.model = model
        self.kv_store = kv_store
        self.waiting = []   # requests that have arrived
        self.running = []   # requests currently being served

    def add_request(self, request):
        self.waiting.append(request)

    def step(self):
        # One unit of serving work.
        # TODO: decide which requests run this step (batching / scheduling),
        #       run prefill and/or decode, append new KV entries, retire finished
        #       requests, admit waiting ones.
        pass

    def run(self):
        while self.waiting or self.running:
            self.step()


# How the KV cache is stored numerically is also a free choice:
#   cache_dtype = torch.float16   # TODO: the precision used for stored KV entries.
