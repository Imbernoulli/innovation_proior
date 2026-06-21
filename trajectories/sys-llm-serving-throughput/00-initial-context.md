## Research question

Serve a fixed autoregressive LLM on a fixed GPU (or fixed multi-GPU group) at the highest possible throughput—generated tokens per second—while keeping per-request latency inside a fixed budget. The model weights, architecture, and hardware are frozen; the free variables are in the serving system itself: how requests are batched, how the per-request KV cache is laid out in memory, how prefill and decode work are scheduled, and how the cache is stored numerically. A better serving system pushes the throughput-vs-latency frontier, mainly by fitting more concurrent requests into the same GPU memory so each forward pass amortizes its fixed cost over a larger batch.

## Prior art / Background / Baselines

Autoregressive transformer inference has two regimes. **Prefill** ingests an entire prompt at once (compute-bound), then **decode** emits one token per forward pass, attending over all previous tokens (memory-bandwidth-bound). To avoid recomputing past keys and values, every serving system keeps a **KV cache**: per-request key and value tensors for every past token, at every layer and head.

The load-bearing observations on current systems:

- The KV cache is enormous and grows token-by-token. For multi-billion-parameter models the per-token footprint is hundreds of kilobytes across layers, so a long request occupies a large slice of GPU memory that only grows during generation. Cache capacity, not model weights, usually limits concurrency.
- Throughput at fixed latency is set by batch size, and batch size is set by memory. Decode is bandwidth-bound: one decode pass over B requests costs nearly the same wall-clock as over one request, so tokens-per-second scales near-linearly with B until memory runs out.
- Requests have wildly different and unpredictable lengths. Prompt and generation lengths vary by orders of magnitude, and a request's final length is not known when it arrives.
- Standard serving reserves cache memory as one contiguous chunk per request, sized to the maximum sequence length. A short request holds the whole reservation until it finishes; the unused tail is dead memory. Measurements on such systems show that only 20–40% of KV-cache memory holds live tokens; the rest is wasted by over-reservation and fragmentation.
- Requests in a stream share content. Many requests begin with the same system prompt, few-shot preamble, or instruction header, yet each stores its own private copy of the identical prefix keys and values.
- Prefill and decode interfere. A long-prompt prefill is one large compute-bound burst; other in-flight decodes are many small bandwidth-bound steps. Run naively, a long prefill stalls everyone else's decoding, while pure-decode passes leave compute units idle.
- The cache is stored at full activation precision. KV entries are kept in FP16/BF16 (two bytes per element), so the cache footprint scales directly with that choice.

Baselines:

- **HuggingFace Transformers generation loop.** Reference autoregressive generation that loads the model, batches prompts, and calls `model.generate` with a per-request KV cache preallocated for the batch. **Gap:** batching lasts the whole `generate` call, so short requests wait for the longest request in their batch to finish, and the contiguous-per-request cache leaves much memory reserved-but-unused, keeping the sustainable batch size small.
- **FasterTransformer.** Heavily hand-optimized inference engine with fast fused transformer kernels; the raw per-pass performance reference. **Gap:** kernels are fast, but request and memory management still reserve per-request cache up to a max length and batch coarsely, so under variable-length streams the GPU feeds fast kernels with small effective batches.
- **Orca.** Introduces iteration-level request handling: scheduling decisions happen at model-iteration granularity, so requests can join and leave the running set between steps. **Gap:** it opens up the scheduling axis but still keeps the KV cache as per-request contiguous memory, so the same memory waste caps how many requests can actually be co-resident, and the scheduler cannot fill the GPU beyond that cap.
- **Just batch more requests.** The obvious throughput knob is to raise the configured batch size. **Gap:** under contiguous per-request reservation, raising the batch size forces a proportional max-length reservation that overflows memory long before compute saturates; shrinking the reservation risks running out of cache mid-generation.

## Fixed substrate / Code framework

The serving engine is a Python loop around the fixed model and a GPU KV-cache region. The pieces that already exist: the model forward, an attention kernel, a tensor of cache memory, and a loop that pulls requests and runs steps. The slots the methods will fill are left empty.

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
```

## Editable interface

The design choices the method can change are:

- **KV-cache organization.** How cache memory is allocated, addressed, and released per request, and how the attention kernel locates a request's keys and values.
- **Scheduling and batching.** Which requests run on each step, how prefill and decode are interleaved, and when finished requests retire and waiting requests are admitted.
- **Cache precision.** The numerical dtype used to store KV entries.
- **Shared prefixes.** Whether and how the system detects and stores identical token prefixes only once.

## Evaluation settings

The workload is a stream of generation requests with prompt and output lengths drawn from a realistic distribution (e.g., ShareGPT-style conversations mixing short and long prompts and generations). The fixed model is a standard decoder-only LLM such as LLaMA-7B on an A10G or LLaMA-13B on an A100-40GB, served on a fixed GPU or fixed tensor-parallel group. The primary metric is **serving throughput—generated tokens per second—at a fixed latency/SLO**; secondary metrics include requests-per-second at fixed latency and the latency-vs-throughput curve. In-repo yardsticks are an offline throughput benchmark and a prefix-sharing benchmark. Each serving change is judged by whether it moves the throughput-at-fixed-latency frontier for the fixed model on the fixed hardware.
