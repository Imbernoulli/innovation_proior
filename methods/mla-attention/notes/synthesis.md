# MLA synthesis

## Pain point
Autoregressive decoding is memory-bandwidth bound. Per generated token, every layer reloads the cached K and V for all past tokens. MHA caches 2·n_h·d_h·l elements/token. This KV cache caps batch size and context length and dominates decode latency (Shazeer 2019; arithmetic intensity is low — one new query against the whole cache).

## Tools on the table
- MHA (Vaswani 2017): q,k,v = W^Q,W^K,W^V h; per-head dot products; cache K,V. Strong but heavy cache.
- MQA (Shazeer 2019): all heads share ONE k and ONE v head → cache 2·d_h·l. Big cut, "minor" quality loss but loss nonetheless; on hard benchmarks degrades.
- GQA (Ainslie 2023): G groups, each shares a k,v head; MHA=G=n_h, MQA=G=1. Cache 2·n_g·d_h·l. Interpolates; uptrain via mean-pooling KV heads with ~5% compute. Still below MHA.
- Diagnostic finding (motivating, about EXISTING systems): 7B dense, 1.33T tokens, params aligned ~7B. MHA > GQA(8) > MQA on BBH/MMLU/C-Eval/CMMLU (e.g. MMLU 45.2 vs 41.2 vs 37.9). → sharing K,V across heads costs real quality.

## RoPE (Su 2021)
Rotation R_{θ,m} applied to q,k before dot product; (R_m q)^T(R_n k)=q^T R_{n-m} k → score depends only on relative pos m−n. Block-diagonal over coord pairs, θ_i=base^{-2i/d}. Crucial: rotation sits BETWEEN q and k; does not commute with arbitrary left-multiplication.

## The MLA idea (re-derive)
Don't share K,V → instead COMPRESS. Down-project h_t to a small latent c_t^KV = W^DKV h_t (d_c ≪ n_h d_h). Cache only c (d_c·l/token). Reconstruct per-head k_t^C=W^UK c, v_t^C=W^UV c. Joint latent shared by K and V (one cache, two up-projs).

### Absorption (so we never materialize K,V at inference)
Content score head i: q_{t,i}^C·k_{j,i}^C = (W^{UQ}_i c_t^Q)·(W^{UK}_i c_j^{KV}) = c_t^{Q⊤} (W^{UQ}_i)^⊤ W^{UK}_i c_j^{KV}. Precompute W̃^Q_i=(W^{UQ}_i)^⊤W^{UK}_i (or absorb W^UK into W^UQ / W^Q). Then score = (absorbed query)·c_j^{KV} — only the latent is touched. Output side: u_t=W^O[...o_{t,i}...], o_{t,i}=Σ a v_{j,i}^C=Σ a W^{UV}_i c_j^{KV} → absorb W^UV into W^O, so V also never materialized. Cache size effectively GQA-like, quality near/above MHA.

### Decoupled RoPE (the wall + fix)
If k_{t,i}^C carries RoPE: k = R_t W^UK c. Score = c_t^{Q⊤}(W^UQ)^⊤ R_t^⊤ R_j W^UK c_j = c_t^{Q⊤}(W^UQ)^⊤ R_{j−t} W^UK c_j. The relative-rotation R_{j−t} depends on token pair → sits between the two up-projs → can't fold into one precomputed matrix → must recompute k for every prefix token → absorption dead.
Fix: decouple position from content. Add a SMALL extra key k_t^R=RoPE(W^KR h_t) ∈ R^{d_h^R}, SHARED across heads (MQA-style, one per token), and matching per-head q_{t,i}^R=RoPE(W^QR c_t^Q). Concatenate: k_{t,i}=[k_{t,i}^C; k_t^R], q_{t,i}=[q_{t,i}^C; q_{t,i}^R]. Score = q^C·k^C (absorbable, no RoPE) + q^R·k^R (carries position, small dim). Softmax scale 1/√(d_h+d_h^R). Cache c_t^KV (d_c) + k_t^R (d_h^R) → (d_c+d_h^R)·l/token.

### Hyperparams (V2/V3, the big model)
n_h=128, d_h=128, d_c=512 (=4 d_h), d_c'=1536, d_h^R=64 (=d_h/2). KV cache=(d_c+d_h^R)l≈(512+64)l ≈ 4.5 d_h l = GQA with 2.25 groups. RMSNorm after c^KV and after c^Q (q_a_layernorm, kv_a_layernorm). V3: recompute up-projections in backprop to save activation memory; YaRN applied only to the decoupled key k^R.

## Code mapping (HF modeling_deepseek.py, DeepseekV2Attention)
- W^DQ→q_a_proj; RMSNorm→q_a_layernorm; W^UQ & W^QR (concatenated, q_head_dim=qk_nope_head_dim+qk_rope_head_dim)→q_b_proj; split q into q_nope,q_pe.
- W^DKV & W^KR fused→kv_a_proj_with_mqa (out = kv_lora_rank + qk_rope_head_dim); split→compressed_kv, k_pe. RMSNorm→kv_a_layernorm.
- W^UK & W^UV fused→kv_b_proj (out = n_h·(qk_nope_head_dim+v_head_dim)); split→k_nope, value_states.
- RoPE applied to q_pe,k_pe only; k_pe shared across heads (head dim 1, broadcast).
- assemble query_states=[q_nope;q_pe], key_states=[k_nope;k_pe (broadcast)]; softmax_scale=q_head_dim^{-0.5}; o_proj=W^O.
Note: this eager forward materializes K,V (training/prefill); absorption is the inference-time optimization (fold kv_b_proj into q_b_proj / o_proj).

## Config (DeepSeek-V2): kv_lora_rank=512, q_lora_rank=1536, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, num_attention_heads=128.

## Sources
1. Primary: DeepSeek-V2 (arXiv:2405.04434) §"Multi-Head Latent Attention" + App full formulas/ablation; DeepSeek-V3 report (arXiv:2412.19437) MLA refinements.
2. Background: Shazeer 2019 MQA (1911.02150); Ainslie 2023 GQA (2305.13245); Su 2021 RoPE (2104.09864); Vaswani 2017 MHA.
3. Explainer: Lior Sinai, "DeepSeek's Multi-Head Latent Attention" (liorsinai.github.io 2025).
