BlurPool — anti-aliasing low-pass filter before pooling/strided convolutions, tagged `Computer Vision`.

Accuracy effect (from the card): "Zhang (2019) showed that BlurPool improves accuracy by 0.5-1% on ImageNet
for various networks." Follow-up work (Zou et al., Lee et al.) reproduced ImageNet accuracy improvements.
For ResNet-50 specifically, the card characterizes it as a pareto improvement: "On ResNet-50 on ImageNet,
we found this tradeoff to be worthwhile: it is a pareto improvement over the standard versions of those
benchmarks. We also found it to be worthwhile in composition with other methods."

| metric | effect (ResNet-50 / ImageNet) | direction |
|---|---|---|
| top-1 accuracy | +0.5–1% (Zhang 2019, "for various networks" on ImageNet) | higher is better |
| throughput | slight decrease (added blur ops; ~doubles maxpool data movement) | higher is better |
| net tradeoff | pareto improvement over the standard ResNet-50/ImageNet benchmark | — |

On the blur-first vs blur-after choice (`blur_first`): the card notes blur-*after* "yielding a roughly 0.1%
larger accuracy gain on ResNet-50 + ImageNet in exchange for a ~10% slowdown" — so the recipe uses
`blur_first=True` to keep the conv multiply-adds constant and add only the filtering overhead. Suggested
config: `blur_first=True`, `blur_convs=True`, `blur_maxpools=True`. The card adds that BlurPool "tends to
compose well with other methods" with no significant change in its effect from other methods being present.

(Provenance: `docs/source/method_cards/blurpool.md`, `composer/algorithms/blurpool/README.md`. The
0.5–1% figure is the card's quoted result from Zhang (2019) on ImageNet across networks; the card frames
the ResNet-50 effect as a pareto improvement rather than quoting one absolute ResNet-50 number.)
