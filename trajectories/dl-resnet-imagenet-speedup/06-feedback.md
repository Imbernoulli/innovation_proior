Squeeze-and-Excitation — channel-wise attention inserted after Conv2d modules, tagged `Computer Vision`.
Suggested ResNet-50 / ImageNet config: `min_channels=512`, `latent_channels=64`.

Effect (from the card): "This method tends to consistently improve the accuracy of CNNs both in absolute
terms and when controlling for training and inference time. This may come at the cost of a roughly 20%
increase in inference latency, depending on the architecture and inference hardware." And: "Squeeze-Excite
slows down training, but it leads to quality improvements that make this a worthwhile tradeoff."

| metric | effect (ResNet-50 / ImageNet) | direction |
|---|---|---|
| top-1 accuracy | consistently improved (absolute and time-controlled) | higher is better |
| inference latency | ~+20% (architecture/hardware dependent) | lower is better |
| training throughput | decreased (extra pool + MLP + scale per block) | higher is better |
| net tradeoff | improved time-vs-quality frontier; worthwhile per the card | — |

No single percentage accuracy number for ResNet-50 is quoted in the card; the documented effect is a
"consistent" accuracy improvement at a ~20% inference-latency cost. Useful composition noted by the card:
because SE decreases GPU throughput, "it can reduce relative load on the CPU and data loading pipeline,
potentially allowing more CPU-intensive speedup methods (e.g., RandAugment) to run without bottlenecking
training on the CPU" — i.e. it helps balance the data-loader bottleneck that progressive resizing introduces.

(Provenance: `docs/source/method_cards/squeeze_excite.md`.)
