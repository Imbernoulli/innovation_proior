ColOut — drops a random fraction of rows and columns of each input image, shrinking it (speed) while adding
mild augmentation variability, tagged `Computer Vision`. Suggested config: `p_row=0.15`, `p_col=0.15`,
`batch=True` (GPU, batch-wise).

Throughput (from the card): running ColOut batch-wise on the GPU yields "a large increase in throughput
(~11% for ResNet-50 on ImageNet) because ColOut is only called once per batch and its operations are offloaded
onto the GPU" in CPU-bottlenecked contexts.

Accuracy (from the card): "The variability induced by randomly dropping rows and columns can negatively affect
generalization performance. In our testing, we saw a decrease in accuracy of ~0.2% in some models on ImageNet
and a decrease in accuracy of ~1% on CIFAR-10."

| metric | effect (ResNet-50 / ImageNet) | direction |
|---|---|---|
| throughput | ~+11% (batch-wise on GPU, CPU-bottlenecked context) | higher is better |
| top-1 accuracy | ~−0.2% (some models on ImageNet) | higher is better |
| net tradeoff | pareto improvement on ResNet-50/ImageNet per the card | — |

The card frames it as a worthwhile tradeoff: "On ResNet-50 applied to ImageNet ... we found this tradeoff to
be worthwhile: it is a pareto improvement over the standard versions of those benchmarks. We also found it to
be worthwhile in composition with other methods." Caveats from the card: `batch=True` yields slightly lower
accuracy than per-sample CPU ColOut; and ColOut "will show diminishing returns when composed with other
methods that change the size of images, such as Progressive Resizing," and combining regularization methods
gives sublinear accuracy improvements.

(Provenance: `docs/source/method_cards/colout.md`.)
