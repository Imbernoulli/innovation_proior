Measured results for **importance-matrix (imatrix) quantization** — same k-quant format, but the per-sub-block
scale+min search is weighted by calibration activation statistics instead of weight magnitude. Perplexity on
Wikitext-2; lower is better. "imatrix WT" = importance matrix built from Wikitext tokens.

## Llama-3 8B (fp16 PPL = 6.233160)

| format | imatrix | bits/weight | model size (GiB) | perplexity | ΔPPL | KLD vs fp16 |
|---|---|---|---|---|---|---|
| q4_K_M | None    | 4.8944 | 4.58 | 6.407115 | +0.175482 | 0.031273 |
| q4_K_M | WT 10m  | 4.8944 | 4.58 | 6.382937 | +0.151303 | 0.028152 |
| q4_K_S | None    | 4.6672 | 4.37 | 6.500529 | +0.268895 | 0.043136 |
| q4_K_S | WT 10m  | 4.6672 | 4.37 | 6.409697 | +0.178064 | 0.031951 |

The imatrix is a pure-quality, **zero in-model bpw cost** gain after the calibration pass — the format and
bits/weight are byte-identical, only the stored integers differ. q4_K_M: ΔPPL 0.175 → **0.151** at the same 4.58 GiB. q4_K_S: ΔPPL 0.269 → **0.178** at
the same 4.37 GiB — the larger gain, exactly where the uniform 4-bit code had the most activation-blind error.
KLD drops in lockstep (q4_K_S 0.0431 → 0.0320). The activation-aware weighting recovers most of what
bit-allocation alone could not, without spending a bit.

Where it matters most — making low bit-widths viable (all imatrix-quantized):

| format | bits/weight | size (GiB) | perplexity | ΔPPL |
|---|---|---|---|---|
| iq2_M   | 2.9294 | 2.74 |  8.600799 | +2.369166 |
| iq2_XS  | 2.5882 | 2.42 | 10.761424 | +4.529791 |
| iq2_XXS | 2.3824 | 2.24 | 14.091782 | +7.860148 |

These sub-3-bit results — coherent models below 3 bpw — are only reachable because the importance signal tells
the quantizer which coordinates to keep and which to sacrifice. Without it, 2-bit quantization is unusable. That
is the door the finale walks through: a codebook quantizer that, given imatrix importance, packs ~2 bits/weight.
