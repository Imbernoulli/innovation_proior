Continuous batching (iteration-level scheduling) — re-form the batch every forward pass; retire finished
requests and admit waiting ones per decode step, holding batch occupancy at the memory ceiling. This rung has
a hard published number.

The isolating comparison is **vLLM vs HuggingFace Text Generation Inference (TGI)**. TGI *already* does
iteration-level (continuous) batching itself — it is not a naive call-level baseline — so the vLLM-over-TGI
margin isolates the contribution of the block-paged cache *plus* the scheduler working together, above and
beyond continuous batching alone (which TGI has). On LLaMA-7B (A10G) and LLaMA-13B (A100-40GB), ShareGPT-
sampled request lengths:

- **2.2×–2.5× higher throughput than TGI for single-completion serving, and 3.3×–3.5× for three parallel
  completions.** Source: the vLLM launch blog, 2023-06-20 ("vLLM: Easy, Fast, and Cheap LLM Serving with
  PagedAttention").

Reading the number: because TGI already batches continuously, the 2.2×–3.5× over TGI is the part of the
throughput gain that continuous batching alone (which TGI has) does *not* explain — it isolates the
paging+scheduler combination. The headline up-to-24× figure is against HF Transformers, which does *neither*
paging *nor* continuous batching, so that gap conflates both rungs; the TGI gap is the cleaner read on what
this rung and rung 1 buy jointly over an already-continuously-batched server.

Role on the ladder: continuous batching converts paging's spatial headroom into sustained high occupancy. It
also rephrases the per-step work as "advance each request by some tokens toward its target within a token
budget," which is the hook the next rungs plug into — splitting long prefills (chunked prefill), skipping
already-computed prefix tokens (prefix caching), and verifying many draft tokens per step (speculative
decoding) all live inside this per-iteration, token-budgeted loop.

(Provenance: 2023-06-20 vLLM launch blog for the 2.2×–3.5× vs TGI figures; LLaMA-7B@A10G / LLaMA-13B@A100-40GB,
ShareGPT request lengths. Code: vllm/v1/core/sched/scheduler.py — Scheduler.schedule(), the per-iteration
admit/retire loop, preemption on block exhaustion.)
