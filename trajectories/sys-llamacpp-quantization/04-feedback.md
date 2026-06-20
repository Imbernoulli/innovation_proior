Measured results for **Q4_K** (k-quants: 256-weight super-block, eight 6-bit affine sub-scales, weighted
scale+min search). Perplexity on Wikitext-2; lower is better. Q4_K_S is the "small" Q4_K variant (all tensors
Q4_K); Q4_K_M adds mixed higher-precision tensors (next rung).

## Llama-2 70B (fp16 PPL = 3.4313)

| format | bits/weight | model size (GiB) | perplexity | Δ to fp16 |
|---|---|---|---|---|
| Q4_0 | 4.5 | 36.20 | 3.5550 | +3.61% |
| Q4_1 | 5.0 | 40.20 | 3.5125 | +2.37% |
| Q5_0 | 5.5 | 44.20 | 3.4744 | +1.26% |
| Q4_K_S | ~4.5 | 36.39 | 3.4852 | +1.57% |

The frontier finally moves down-and-left. Q4_K_S at ~4.5 bpw (36.39 GiB) reaches +1.57% — better than Q4_0 at
the *same* bpw (+3.61% → +1.57%, less than half the gap) and nearly matching Q5_0's +1.26% while costing a full
bit less (4.5 vs 5.5 bpw, 36.39 vs 44.20 GiB). The super-block hierarchy plus the weighted scale+min search,
not added bits, was the unlock: the legacy formats were being strangled by per-32 fp16 metadata.

## Llama-3 8B (fp16 PPL = 6.233160)

| format | bits/weight | model size (GiB) | perplexity | ΔPPL | KLD vs fp16 |
|---|---|---|---|---|---|
| q4_K_S (no imatrix) | 4.6672 | 4.37 | 6.500529 | +0.268895 | 0.043136 |

On 8B the plain (no-imatrix) Q4_K_S lands at ΔPPL 0.269 — far below the legacy q4_0 (0.469) and q4_1 (0.451) at
comparable size. The same super-block idea instantiates a whole graded family (Q5_K, Q6_K, …), opening the next
two questions: a high-precision rung, and whether the bits can be *mixed* across tensors.
