# Mistral 7B synthesis (Phase 1.5)

## Verified
- arXiv 2310.06825, "Mistral 7B". Title verified from source.
- Canonical impl: original mistral-src repo (the reference implementation cited in the paper), pinned commit 147c4e6 — code/orig_model.py (Attention with GQA + sliding window, SwiGLU, RMSNorm, RoPE) and code/orig_cache.py (RotatingBufferCache, CacheView, unrotate, get_input_metadata). This is the cleanest grounding.

## Pain point
- Scaling for performance inflates inference cost + latency → barrier to real deployment. Want high performance AT a fixed, efficient inference cost. Two distinct inference bottlenecks attacked:
  (a) KV-cache memory bandwidth during autoregressive decode (every step streams all cached K,V) — limits batch size/throughput.
  (b) attention is O(n^2) compute and O(n) memory in sequence length n — long sequences are expensive in both latency and cache size.
- Base architecture is a LLaMA-style decoder: pre-norm RMSNorm, SwiGLU FFN, RoPE. (These are established; reasoning treats them as the substrate and focuses on the inference innovations.)
- Config (Table): dim=4096, n_layers=32, head_dim=128, hidden_dim=14336, n_heads=32, n_kv_heads=8, window_size W=4096, context_len=8192, vocab=32000.

## Mechanism 1: Grouped-Query Attention (GQA) — attack (a)
- n_heads=32 query heads, n_kv_heads=8 key/value heads → repeats = 32/8 = 4 query heads per kv head. KV cache is 32/8 = 4× smaller. (Full derivation of GQA is the cache-bandwidth argument: cache memory term ∝ number of kv heads; cutting kv heads by 4 cuts the dominant decode-time memory traffic by 4, with capacity spread over 8 subspaces instead of collapsed to 1.) — already fully grounded in methods/gqa.
- Code: wq → n_heads*head_dim; wk,wv → n_kv_heads*head_dim; repeat_kv via torch.repeat_interleave(repeats=4, dim of heads) after loading from cache, before the score matmul. So 8 heads live in the cache, expanded to 32 only for the matmul (no extra cache cost).

## Mechanism 2: Sliding Window Attention (SWA) — attack (b)
- Vanilla attention: token i attends to all j ≤ i → O(n^2) ops, O(n) cache per token, growing unboundedly.
- SWA: token i (layer k) attends only to positions in [i-W, i] of the previous layer. Each layer's attention is local, window W.
- Key insight exploiting STACKED layers: although one layer only reaches back W, information PROPAGATES across layers. h_i at layer k can access input-layer tokens up to W*k back. After k layers, receptive field ≈ k*W. (Like a CNN's growing receptive field with depth.)
- With W=4096, n_layers=32 → theoretical span ≈ 32*4096 ≈ 131K tokens at the last layer. So tokens outside the window still influence the prediction, indirectly, through the layer stack.
- Cost: attention is now O(n*W) not O(n^2); each query attends to ≤ W keys.
- Practical: with FlashAttention/xFormers changes, ~2× speedup over vanilla for seq 16K, W=4096.
- Mask: each query attends to the W keys at or before it (local causal mask).

## Mechanism 3: Rolling Buffer Cache — consequence of fixed W
- SWA gives a FIXED attention span W → you never need more than W cached K,V per sequence (anything older than W back is never attended to). So cap the cache at size W.
- Rolling/ring buffer: K,V for timestep i stored at slot (i mod W). When i ≥ W, slot (i mod W) overwrites the entry from i-W (which has just fallen out of every query's window). Cache size stops growing at W.
- Memory: for seq len 32k with W=4096 → 32768/4096 = 8× reduction in cache memory, no quality impact. (Generally min(n, W)/n... the reduction factor is n/W when n>W.)
- Mechanics (orig_cache.py): cache_positions = positions % W (+ batch offset). to_cache_mask keeps only the LAST W tokens of each input chunk (x >= seqlen - W). On retrieval for attention, the ring is "unrotated" back into chronological order: if seqlen >= W and position = seqlen % W != 0, cat([cache[position:], cache[:position]]).

## Mechanism 4: Pre-fill and Chunking — efficient prompt processing
- Generation is one-token-at-a-time, but the PROMPT is known fully in advance → pre-fill the (k,v) cache with the whole prompt in one parallel pass (no need to decode it token by token).
- If prompt is long, chunk it (chunk size = W is natural). For each chunk the attention is computed over (a) the chunk itself with a CAUSAL mask (rightmost block), (b) the cache from previous chunks with a SLIDING-WINDOW mask (center block), and (c) NOT over tokens older than W (left block, masked out). This keeps memory bounded during prefill of long prompts.

## Derivations to work in reasoning.md
- SWA receptive field: induction over layers. Layer 1: token i sees [i-W, i] of input. Layer 2: token i sees [i-W,i] of layer-1 outputs, each of which saw back another W → [i-2W, i] of input. By induction after k layers: [i-kW, i]. So effective span = k*W. Plug k=32, W=4096 → ~131K.
- Cache size bound: query at position i attends only [i-W+1 ... i] (W keys). Any K,V at position ≤ i-W is dead. So at any time only the most recent W entries can ever be needed → buffer of size W suffices; slot = i mod W; entry at slot i mod W when writing position i held position i-W (if i≥W), which is exactly the one just expiring.
- GQA cache factor = n_heads/n_kv_heads = 4 (memory + bandwidth).
- Combined: cache memory ∝ W * n_kv_heads (per layer per token-dim) instead of n * n_heads → reductions multiply.

## Scaffold ↔ final code correspondence
- repeat_kv ← context stub: helper to match kv heads to query heads
- Attention (wq,wk,wv,wo with n_kv_heads<n_heads, sliding-window mask, cache) ← context stub: self-attention module
- RotatingBufferCache/CacheView (ring buffer size W, i mod W, unrotate) ← context stub: a cache object
- FeedForward (SwiGLU), RMSNorm, TransformerBlock, Transformer ← context stubs (LLaMA-style substrate)

## OUT of scope (eval)
- All benchmark tables vs Llama 2 / Llama 34B / Code-Llama, instruction-finetuning (Mistral-Instruct) results, guardrails/system-prompt content moderation eval. Stop at architecture + the four inference mechanisms.
