Measured results for **Q6_K** (6-bit symmetric k-quant: 16 sub-blocks of 16, int8 sub-scales, one fp16
super-scale). Perplexity on Wikitext-2; lower is better. Q6_K is the near-lossless anchor of the k-quant family.

## Llama-2 70B (fp16 PPL = 3.4313)

| format | bits/weight | model size (GiB) | perplexity | Δ to fp16 |
|---|---|---|---|---|
| Q4_K_S | ~4.5 | 36.39 | 3.4852 | +1.57% |
| Q5_K_M | ~5.7 | 45.41 | 3.4451 | +0.40% |
| Q6_K | 6.5625 | 52.70 | 3.4367 | +0.16% |
| fp16 | 16 | 128.5 | 3.4313 | — |

Q6_K reaches **+0.16%** — essentially fp16 quality (3.4367 vs 3.4313). The graded family is now pinned end to
end: Q4_K_S +1.57% @ 4.5 bpw → Q5_K_M +0.40% @ ~5.7 bpw → Q6_K +0.16% @ 6.56 bpw. At 6.56 bpw and 52.70 GiB
Q6_K is the high-cost corner, not a frontier move — its job is to be the near-lossless reference, and to show
that within one model the precision *most* tensors need (Q4_K) is far below what a *few* sensitive ones need
(Q6_K).

## Llama-3 8B (fp16 PPL = 6.233160)

| format | bits/weight | model size (GiB) | perplexity | ΔPPL | KLD vs fp16 |
|---|---|---|---|---|---|
| q6_K | 6.5633 | 6.14 | 6.253382 | +0.021748 | 0.005452 |
| q8_0 | 8.5008 | 7.96 | 6.234284 | +0.002650 | 0.001355 |
| f16  | 16.0005 | 14.97 | 6.233160 | — | — |

On 8B, q6_K's ΔPPL of 0.022 and KLD of 0.0055 confirm near-losslessness at 6.56 bpw. The observation it surfaces
— uneven per-tensor sensitivity — is the lever for the next rung's *mixed* bit allocation.
