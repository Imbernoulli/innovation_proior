Measured result — torch.compile (reduce-overhead / CUDA-graph) + static KV-cache, Llama-2-7B, batch 1,
A100-80GB (power-limited 330 W). Source: gpt-fast README benchmark table, row `Llama-2-7B / Base` (the
README "Base" already includes compile + static KV-cache); the accompanying PyTorch blog reports 107.0
for the same step. Metric: decoding tokens/second, **higher is better**.

| configuration | tokens/second | bandwidth achieved (GB/s) |
|---|---|---|
| eager baseline | 25.5 | ~340 |
| + torch.compile + static KV-cache (README "Base") | **104.9** | 1397.31 |

A ~4.1× jump. The achieved memory bandwidth rises to 1397.31 GB/s — about 70% of the ~2 TB/s A100 HBM
peak — confirming the run is now genuinely memory-bandwidth bound rather than idle on host-side launch
overhead. The output distribution is unchanged (scheduling-only). The companion blog quotes 107.0 tok/s
for the same compile + static-KV-cache step.
