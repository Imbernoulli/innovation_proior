Measured results for **Q5_0** (symmetric 5-bit code, 32 levels, one fp16 scale + 32-bit `qh` per block).
Perplexity on Wikitext-2; lower is better.

## Llama-2 70B (fp16 PPL = 3.4313)

| format | bits/weight | model size (GiB) | perplexity | Δ to fp16 |
|---|---|---|---|---|
| Q4_0 | 4.5 | 36.20 | 3.5550 | +3.61% |
| Q4_1 | 5.0 | 40.20 | 3.5125 | +2.37% |
| Q5_0 | 5.5 | 44.20 | 3.4744 | +1.26% |

Resolution wins decisively over offset: Q5_0 cuts the gap to **+1.26%**, nearly halving Q4_1's +2.37% — for the
same half-bit step on the cost axis (5.0 → 5.5 bpw). Doubling the levels halves the round-off floor, and on the
near-symmetric blocks that dominate, that floor was the binding residual. But the cost-axis story is now stark:
three legacy formats marching *up* in bpw (4.5 → 5.0 → 5.5), each better than the last, none cheaper. The fixed
per-block fp16 metadata that all three pay is the next target.

## Llama-3 8B (fp16 PPL = 6.233160)

| format | bits/weight | model size (GiB) | perplexity | ΔPPL | KLD vs fp16 |
|---|---|---|---|---|---|
| q5_0 | 5.21 | 5.21 | 6.363224 | +0.131591 | 0.022239 |

On 8B, Q5_0's ΔPPL of 0.132 is far below the 4-bit legacy formats (q4_0 0.469, q4_1 0.451) — the resolution gain
holds. The frontier it leaves open: every legacy format here spends a fixed fp16 per 32-weight block, so the only
way it has found to gain quality is to keep adding bits.
