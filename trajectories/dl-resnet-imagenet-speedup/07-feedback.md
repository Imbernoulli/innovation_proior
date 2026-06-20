EMA (Exponential Moving Average) — model-averaging method; maintains an exponentially weighted moving average
of the model parameters (and buffers) during training, used for evaluation. Suggested config: `half_life=
'1000ba'`, `update_interval` defaulting to `1ba` (or larger, e.g. `10ba`, to save runtime).

Effect (from the card, stated qualitatively): "EMA typically results in less noisy validation metrics over
the course of training, and sometimes increased generalization." And: "In our experiments, EMA improves the
attainable tradeoffs between training speed and the final quality of the trained model. We recommend EMA for
training convolutional networks." On the noise-reduction point: "If evaluation metrics are computed over the
course of training, EMA should result in these metrics being smoother and less noisy due to averaging."

| metric | effect (ResNet-50 / ImageNet) | direction |
|---|---|---|
| validation-metric noise | reduced (smoother metrics over training) | lower is better |
| top-1 accuracy | sometimes increased generalization | higher is better |
| throughput | small slowdown (extra moving-average compute; mitigated by larger `update_interval`) | higher is better |
| memory | extra copy of params + buffers (small relative to activations/optimizer state) | — |

No single percentage number is given for ResNet-50 in the card; the documented effects are "less noisy
validation metrics" and "sometimes increased generalization." Caveats from the card: model-averaging methods
do not compose well — use EMA or SWA, not both; and evaluation must be done with the `ema_model` (the averaged
weights), not the training model.

(Provenance: `docs/source/method_cards/ema.md`.)
