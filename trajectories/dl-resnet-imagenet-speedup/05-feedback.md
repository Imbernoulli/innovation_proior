Progressive Image Resizing — trains on downsampled images early and grows them back to full size, tagged
`Computer Vision`. Suggested ResNet-50 / ImageNet config: `initial_scale=0.5`, `finetune_fraction=0.2`,
`delay_fraction=0.5`, `size_increment=4`, `mode='resize'` (interpolation, which the card says works better
than crop for ResNet-50 on ImageNet).

Effect (from the card, stated as a wall-clock / quality tradeoff rather than a single number): "In our
experience with ResNets on ImageNet, Progressive Resizing improves training speed (as measured by wall clock
time) with limited effects on classification accuracy." And: "In our experiments, Progressive Resizing
improves the attainable tradeoffs between training speed and the final quality of the trained model. In some
cases, it leads to slightly lower quality than the original model for the same number of training steps.
However, Progressive Resizing increases training speed so much (via improved throughput during the early part
of training) that it is possible to train for more steps, recover accuracy, and still complete training in
less time."

| metric | effect (ResNet-50 / ImageNet) | direction |
|---|---|---|
| wall-clock training time | substantially reduced (cheaper early steps; convs scale with image area) | lower is better |
| top-1 accuracy | limited effect; possibly slightly lower per-step, recoverable by training longer | higher is better |
| net tradeoff | improved time-vs-quality frontier | — |

The card gives no single percentage throughput number for ResNet-50; the documented effect is a large
early-phase throughput increase with limited accuracy impact. Flagged caveats from the card: it may shift
the bottleneck onto data loading / CPU during the small-image phase, and it interacts with other
input-size-changing methods (ColOut, Selective Backprop with downsampling, CutOut) with diminishing returns.

(Provenance: `docs/source/method_cards/progressive_resizing.md`,
`composer/algorithms/progressive_resizing/README.md`.)
