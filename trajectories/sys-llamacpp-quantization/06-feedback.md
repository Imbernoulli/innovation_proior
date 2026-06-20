Measured results for **mixed k-quant allocation** (Q4_K_M / Q4_K_S — most tensors Q4_K, leverage-heavy tensors
promoted to Q5_K/Q6_K by the per-tensor/per-layer policy). Perplexity on Wikitext-2; lower is better.

## Llama-2 70B (fp16 PPL = 3.4313)

| format | bits/weight | model size (GiB) | perplexity | Δ to fp16 |
|---|---|---|---|---|
| Q4_K_S (uniform) | ~4.5 | 36.39 | 3.4852 | +1.57% |
| Q4_K_M (mixed)   | ~4.8 | 38.54 | 3.4725 | +1.20% |
| Q5_K_M (mixed)   | ~5.7 | 45.41 | 3.4451 | +0.40% |
| Q6_K (uniform)   | 6.56 | 52.70 | 3.4367 | +0.16% |

The localized-sensitivity hypothesis holds: Q4_K_M cuts the gap from Q4_K_S's +1.57% to **+1.20%** for only a
~2 GiB increase (36.39 → 38.54), by sprinkling Q5_K/Q6_K onto `attn_v`, `ffn_down`, and the boundary layers.
A few targeted promotions recover a meaningful fraction of the gap toward the Q6_K anchor at far below Q6_K's
cost — confirming that bits should be spent *non-uniformly* across tensors.

## Llama-3 8B (fp16 PPL = 6.233160)

| format | bits/weight | model size (GiB) | perplexity | ΔPPL | KLD vs fp16 |
|---|---|---|---|---|---|
| q4_K_S (no imatrix) | 4.6672 | 4.37 | 6.500529 | +0.268895 | 0.043136 |
| q4_K_M (no imatrix) | 4.8944 | 4.58 | 6.407115 | +0.175482 | 0.031273 |

On 8B the mixed Q4_K_M (no imatrix) improves ΔPPL from q4_K_S's 0.269 to **0.175** for ~0.2 GiB. Allocation
alone, however, cannot touch the deeper limit: every k-quant still optimizes *weight* error using a
magnitude-derived proxy, not by how much each weight moves the *output* — the lever for the next rung.
