Measured results — two-token architecture (class token + distillation token), hard-label objective and
teacher (RegNetY-16GF, 82.9% top-1) fixed from rung 2, rung-1 recipe otherwise unchanged. ImageNet-1k,
top-1 accuracy.

## Readout comparison, 224² training

| model | class-token only | distillation-token only | fused (class+distillation) |
|---|---|---|---|
| Tiny | 73.9 | 74.6 | 74.5 |
| Small | 80.9 | 81.1 | 81.2 |
| Base | 83.0 | 83.1 | 83.4 |

## Same readouts, Base size fine-tuned at 384²

| readout | top1_imagenet |
|---|---|
| class-token only | 84.2 |
| distillation-token only | 84.4 |
| fused (class+distillation) | 84.5 |

## Reference: rung-2 single-token hard distillation (same teacher, same recipe, no second token)

| model | 224² | 384² |
|---|---|---|
| Base, class token only, hard-label loss | 83.0 | 84.0 |

The fused two-token readout (83.4% / 84.5%) beats the rung-2 single-token hard-distillation baseline
(83.0% / 84.0%) at every resolution, and beats either individual token's own classifier read alone. The
distillation-token classifier alone already edges out the class-token classifier alone at every size and
resolution (e.g. Base size: 83.1 vs. 83.0 at 224², 84.4 vs. 84.2 at 384²).

## Token-similarity diagnostic (Base size)

| token pair | cosine similarity, near input | cosine similarity, final layer |
|---|---|---|
| class token vs. distillation token (real design) | 0.06 | 0.93 |
| class token vs. duplicate-class-token (control) | not separately reported | 0.999 |

The two tokens in the real design start almost orthogonal and only partially converge by the last layer,
staying measurably below 1. The duplicate-class-token control converges to a near-identical vector
(cosine 0.999) and its output embeddings are reported as quasi-identical to the class token's; this
control token does not bring any additional classification performance over a single class token.
