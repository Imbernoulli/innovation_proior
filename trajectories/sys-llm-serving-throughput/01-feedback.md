PagedAttention — block-paged KV cache → near-zero fragmentation → larger sustainable batch → higher
serving throughput at fixed latency. This rung has hard published numbers.

Versus the state-of-the-art serving systems, at equal latency:
- **2–4× higher throughput than FasterTransformer and Orca at the same level of latency.** Source: the
  SOSP 2023 PagedAttention paper abstract ("Efficient Memory Management for Large Language Model Serving with
  PagedAttention", SOSP '23 / arXiv:2309.06180): "vLLM improves the throughput of popular LLMs by 2-4× with
  the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca."
  The gain is larger with longer sequences, larger models, and more complex decoding.

Versus the naive reference stacks (vLLM launch blog, 2023-06-20, "Easy, Fast, and Cheap LLM Serving with
PagedAttention", measured on LLaMA-7B on an NVIDIA A10G and LLaMA-13B on an NVIDIA A100-40GB, request lengths
sampled from ShareGPT):
- **Up to 24× higher throughput than HuggingFace Transformers (HF).** Single-completion: 14×–24× over HF;
  three parallel completions: 8.5×–15× over HF.
- **Up to 3.5× higher throughput than HuggingFace Text Generation Inference (TGI).** Single-completion:
  2.2×–2.5× over TGI; three parallel completions: 3.3×–3.5× over TGI.

Mechanism behind the numbers, in the paper's own framing: existing systems waste 60–80% of KV-cache memory to
fragmentation and over-reservation; PagedAttention brings KV waste to near zero, which raises the sustainable
batch size, and because decode is memory-bandwidth-bound, throughput at fixed latency rises with batch size.

Role on the ladder: the foundation. Every later rung — continuous batching, chunked prefill, prefix caching,
speculative decoding, FP8 KV cache — is built on the block-paged cache this rung introduces (block tables,
the free-list block pool, reference-counted blocks).

(Provenance: SOSP'23 abstract / arXiv:2309.06180 for the 2–4× vs FasterTransformer/Orca; the 2023-06-20 vLLM
launch blog for the up-to-24× vs HF and up-to-3.5× vs TGI numbers. Code: csrc attention kernels
(paged_attention_v1.cu / paged_attention_v2.cu and attention_kernels.cuh); vllm/v1/core/kv_cache_manager.py
and block_pool.py for the block-pool / block-table management.)
