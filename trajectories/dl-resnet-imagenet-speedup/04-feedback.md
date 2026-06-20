Label Smoothing — softens the target distribution toward uniform; a regularizer, tagged `Computer Vision`.
Formula: `smoothed_labels = (targets * (1. - smoothing)) + (smoothing / n_classes)`, with `smoothing=0.1` the
standard starting point.

Accuracy effect (from the card, stated qualitatively across the image-classification benchmarks rather than
as a single ResNet-50 number): "Label smoothing is intended to act as a regularizer, and a possible effect
is a change (ideally improvement) in generalization performance. We find this to be the case on all of our
image classification benchmarks, which see improved accuracy under label smoothing." And: "In our
experiments, Label Smoothing improves the attainable tradeoffs between training speed and the final quality
of the trained model. We recommend Label Smoothing for image classification tasks."

| metric | effect (ResNet-50 / ImageNet) | direction |
|---|---|---|
| top-1 accuracy | improved (regularization; "all of our image classification benchmarks") | higher is better |
| throughput | unaffected ("We did not observe label smoothing to affect throughput in any way") | higher is better |
| memory/compute | small extra to densify label indices | — |

No separate logged percentage is given for ResNet-50 in the card; the documented effect is "improved
accuracy" with no throughput cost. Caveat from the card: label smoothing produces a full distribution (the
loss must accept it), and it may interact with other target-modifying methods such as MixUp.

(Provenance: `docs/source/method_cards/label_smoothing.md`.)
