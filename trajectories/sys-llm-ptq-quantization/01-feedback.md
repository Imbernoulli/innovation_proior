Measured result — round-to-nearest (RTN), the quantization floor. Metric: WikiText perplexity
(**lower is better**); exact bit-width and grouping stated per row because they are not comparable.

| model | setting | RTN perplexity | FP16 reference | source |
|---|---|---|---|---|
| LLaMA-7B | 3-bit weight-only, **per-channel** | **25.54** | 5.68 | AWQ Table 4 (arXiv:2306.00978), RTN row |
| Llama-2-7B | W4-**g128** weight-only | **5.73** | ~5.47 | AWQ Table 4, RTN-g128 |
| Llama-2-7B | INT3-**g128** weight-only | 6.66 | ~5.47 | AWQ Table 4, RTN-g128 |
| Llama-2-7B | **W4A4KV4** (weight+activation) | ~2×10³ | 5.47 | SpinQuant README W4A4KV4 table, RTN row |

RTN is the right floor: nearly free and tolerable at 4-bit g128 (5.73 ppl, within ~0.3 of FP16), but
it falls off a cliff at **3-bit per-channel** — LLaMA-7B perplexity **25.54** against an FP16 of 5.68,
a >4× blow-up no one would deploy. And the moment activations are pushed to 4 bits (W4A4), RTN
produces perplexity on the order of **2,000** — a broken model — because a few activation channels
carry ~100× outliers a single scale cannot represent. Per-channel 3-bit weight rounding is the gap the
next rung must close; the W4A4 catastrophe is the gap rungs 4–6 must close.
