Measured result — SmoothQuant (Xiao et al. 2022), offline migration of activation outliers into the
weights for W8A8 INT8. Metric here is **zero-shot accuracy averaged over the reasoning suite (higher is
better)**, since the breakthrough is making INT8 *activation* quantization work at scale; bit-width is
W8A8 (INT8 weights + INT8 activations).

| model | setting | naive W8A8 | SmoothQuant W8A8 | FP16 | source |
|---|---|---|---|---|---|
| OPT-175B | W8A8, zero-shot avg | 35.5% | **66.8%** | 66.9% | SmoothQuant Table 3 (arXiv:2211.10438) |

Naive per-tensor W8A8 is chance-level on OPT-175B (**35.5%**, a destroyed model) because activation
channel outliers blow up the per-tensor step. SmoothQuant migrates those outliers into the weights
offline — folding a per-channel factor into the preceding LayerNorm and the next linear's weights — so
both operands quantize on hardware-friendly scales and the layers run as dense INT8 GEMMs. The zero-shot
average comes back to **66.8%**, just **0.1 point** below FP16's 66.9%. This is the first true
weight-and-activation quantization on the ladder, and it works at 8-bit. But the migrated outliers are
*smaller*, not *gone*: the same per-tensor scheme will not survive 4-bit activations, where the residual
outliers reappear — the wall the next rung must break to reach true W4A4.
