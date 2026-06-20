Measured result — eager PyTorch baseline decoding, Llama-2-7B, batch 1, A100-80GB (power-limited 330 W).
Source: the PyTorch blog "Accelerating Generative AI with PyTorch II: GPT, Fast" (the eager starting
point, before any optimization). Metric: decoding tokens/second, **higher is better**.

| configuration | tokens/second |
|---|---|
| eager PyTorch baseline (no compile, no static cache) | **25.5** |

At 25.5 tok/s the run streams the model's ~13.5 GB of weights about 25.5 times per second, i.e. it
achieves on the order of ~340 GB/s of effective bandwidth against an A100 HBM peak near 2 TB/s — the
device is running at well under a fifth of its memory ceiling, idle much of the time waiting on host-side
kernel dispatch.
