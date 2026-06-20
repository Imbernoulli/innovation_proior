Measured result — int4 weight-only groupwise (G=32) + GPTQ, Llama-2-7B, batch 1, A100-80GB
(power-limited 330 W). Source: gpt-fast README benchmark table, row `Llama-2-7B / 4-bit (G=32)`; the
companion PyTorch blog reports 202.1 for the int4+GPTQ step. Metric: decoding tokens/second, **higher is
better**.

| configuration | tokens/second | bandwidth achieved (GB/s) |
|---|---|---|
| int8 weight-only per-channel | 155.58 | 1069.20 |
| + int4 weight-only G=32 + GPTQ | **196.80** | 862.69 |

A ~1.27× gain over int8. Achieved bandwidth drops again to 862.69 GB/s as tokens/s rises — the same
bandwidth-bound signature (fewer weight bytes streamed per token). Quality, measured on the EleutherAI
harness (hellaswag / winogrande, via `eval.py`), is held to **minimal** loss with G=32 grouping plus GPTQ
error-feedback calibration on wikitext — distinct from int8's no-observable-loss. The companion blog
quotes 202.1 tok/s for the same int4+GPTQ step.
