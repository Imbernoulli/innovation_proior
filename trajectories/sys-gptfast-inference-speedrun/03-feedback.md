Measured result — int8 weight-only per-channel quantization, Llama-2-7B, batch 1, A100-80GB
(power-limited 330 W). Source: gpt-fast README benchmark table, row `Llama-2-7B / 8-bit`; the companion
PyTorch blog reports 157.4 for the same step. Metric: decoding tokens/second, **higher is better**.

| configuration | tokens/second | bandwidth achieved (GB/s) |
|---|---|---|
| compile + static KV-cache ("Base") | 104.9 | 1397.31 |
| + int8 weight-only per-channel | **155.58** | 1069.20 |

A ~1.48× throughput gain. Note the achieved bandwidth *drops* to 1069.20 GB/s while tokens/s rises — the
signature of a bandwidth-bound win: each token now streams ~half the weight bytes, so fewer GB cross HBM
per token even though the absolute GB/s is lower. Accuracy on the EleutherAI harness (hellaswag /
winogrande) shows **no observable quality loss**. The companion blog quotes 157.4 tok/s for the same
int8 step.
