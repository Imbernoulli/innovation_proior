Measured results for **Q4_0** (symmetric round-to-nearest, one fp16 scale per 32-weight block). Perplexity
on Wikitext-2; lower is better. Δ is the perplexity gap to the fp16 reference on the same model/engine.

## Llama-2 70B (fp16 PPL = 3.4313)

| format | bits/weight | model size (GiB) | perplexity | Δ to fp16 |
|---|---|---|---|---|
| Q4_0 | 4.5 | 36.20 | 3.5550 | +3.61% |

## Llama-3 8B (fp16 PPL = 6.233160)

| format | bits/weight | model size (GiB) | perplexity | ΔPPL | KLD vs fp16 |
|---|---|---|---|---|---|
| q4_0 | 4.34* | 4.34 | 6.700147 | +0.468514 | 0.071940 |

\* bits/weight on Llama-3 8B as reported in-repo (the metadata amortizes slightly differently across this
model's tensor shapes). Q4_0 is the most-degraded 4-bit format in both tables: on 70B it loses 3.61% PPL,
the largest gap among the 4-bit-and-up legacy/k-quant formats; on 8B its ΔPPL of 0.47 is the worst of any
~4.3 bpw scheme. This is the baseline every later rung is measured against.
