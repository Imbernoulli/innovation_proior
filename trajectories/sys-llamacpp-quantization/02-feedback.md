Measured results for **Q4_1** (affine code, per-block scale *and* min). Perplexity on Wikitext-2; lower is
better. Δ is the gap to the fp16 reference on the same model/engine.

## Llama-2 70B (fp16 PPL = 3.4313)

| format | bits/weight | model size (GiB) | perplexity | Δ to fp16 |
|---|---|---|---|---|
| Q4_0 | 4.5 | 36.20 | 3.5550 | +3.61% |
| Q4_1 | 5.0 | 40.20 | 3.5125 | +2.37% |

The affine offset cut the perplexity gap from +3.61% to **+2.37%** — a clear win on quality. The cost is the
extra fp16 per block: 5.0 bpw vs 4.5, and 40.20 GiB vs 36.20. So the trade is real but paid for in bits: Q4_1
is better *and* bigger. The asymmetry fix worked, but it bought quality with storage rather than for free —
leaving the question of whether the same half-bit could do more if spent on resolution or on cheaper shared
metadata.

## Llama-3 8B (fp16 PPL = 6.233160)

| format | bits/weight | model size (GiB) | perplexity | ΔPPL | KLD vs fp16 |
|---|---|---|---|---|---|
| q4_0 | 4.34 | 4.34 | 6.700147 | +0.468514 | 0.071940 |
| q4_1 | ~4.78 | 4.78 | 6.682737 | +0.451103 | 0.071683 |

On 8B the ordering holds (q4_1 0.451 ΔPPL < q4_0 0.469) but the margin is thin and again bought with bits
(4.78 vs 4.34 GiB). Both legacy 4-bit formats sit well above the k-quant frontier reached next.
