Measured result — QuaRot (Ashkboos et al. 2024), computational-invariance Hadamard rotations +
GPTQ for true W4A4KV4. Metric: WikiText-2 perplexity (**lower is better**); bit-width **W4A4** (4-bit
weights + 4-bit activations; KV4 in the full pipeline). Numbers from **QuaRot Table 1** (arXiv:2404.00456),
the multi-method 4-bit anchor.

| model | SmoothQuant W4A4 | QuaRot W4A4 | FP16 | source |
|---|---|---|---|---|
| Llama-2-7B | 83.12 | **6.10** | 5.47 | QuaRot Table 1 |
| Llama-2-13B | — | **5.40** | 4.88 | QuaRot Table 1 |
| Llama-2-70B | — | **3.79** | 3.32 | QuaRot Table 1 |

QuaRot is the first method on the ladder to make true **4-bit activations** work: rotating the residual
stream by a randomized Hadamard matrix (a free transform under computational invariance) dissolves the
axis-aligned activation outliers that broke SmoothQuant at W4A4, and GPTQ on the rotated weights handles
the 4-bit weight rounding. On Llama-2-7B it goes from SmoothQuant-W4A4's broken **83.12** down to
**6.10** against FP16 5.47 — a gap of ~0.63. As predicted, the rotation argument gets cleaner with
width: the FP16 gap shrinks to **5.40** vs 4.88 at 13B (~0.52) and **3.79** vs 3.32 at 70B (~0.47), so
the largest model is within roughly half a perplexity point of full precision at W4A4 — with every
matmul genuinely in 4 bits and no higher-precision channels. The remaining loose thread: the rotation Q
is *random*, not chosen for this model — the opening the finale exploits by learning it.
