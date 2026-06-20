Block-wise Stochastic Depth — each residual block has a probability of dropping its transformation (using only
the skip connection), reducing computation and regularizing. Only applicable to architectures with skip
connections (ResNet-50/101/152). Suggested ResNet-50 config: `drop_rate=0.1` (smallest-time-with-minimal-
accuracy per the Suggested Hyperparameters), `drop_distribution='linear'`, `drop_warmup=0.0dur`.

Effect (from the card): "For ResNet-50 on ImageNet, we used `drop_rate=0.2` and `drop_distribution=linear`. We
measured a 5% decrease in training wall-clock time while maintaining a similar accuracy to the baseline." For
ResNet-101 the card reports `drop_rate=0.4` giving "a 10% decrease in training wall-clock time while
maintaining an accuracy within 0.1% of the baseline." The Suggested Hyperparameters section notes that for
ResNet-50 a drop rate of 0.1 with a linear distribution "has the largest reduction in training time with a
minimal reduction in accuracy."

| metric | effect (ResNet-50 / ImageNet) | direction |
|---|---|---|
| wall-clock training time | −5% (at `drop_rate=0.2`, linear) | lower is better |
| top-1 accuracy | ~maintained / similar to baseline (at `drop_rate=0.2`) | higher is better |
| net tradeoff | improved time-vs-quality frontier per the card | — |

The card frames it as a worthwhile tradeoff and notes that the tolerable drop rate scales with model depth
(ResNet-101 tolerates almost double ResNet-50's rate). Caveats: it can shift the bottleneck to data loading by
speeding up the forward/backward passes; and composing regularization methods may give diminishing quality
returns.

(Provenance: `docs/source/method_cards/stochastic_depth.md`. The 5% wall-clock reduction at similar accuracy
for ResNet-50 with `drop_rate=0.2` linear is the card's own measured number.)
