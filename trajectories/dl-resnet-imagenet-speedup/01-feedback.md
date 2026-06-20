Decoupled Weight Decay (DecoupledSGDW) — optimizer correctness fix on the floor recipe. This is not
implemented as a quality "algorithm" with its own benchmark delta; it ships as a drop-in optimizer
replacement for `torch.optim.SGD`. There is no separately logged accuracy/throughput number for this record
in the method card.

What it changes, in the card's own words: the standard SGD optimizer couples the weight-decay term with the
gradient calculation, which ties the optimal value of `weight_decay` to `lr`. Decoupling them lets weight
decay be tuned independently of the learning rate. The card notes that the optimal `weight_decay` for the
decoupled optimizer "will typically be smaller than for their non-decoupled counterparts, because decoupled
weight decay is not scaled by the learning rate," and that there are "no known negative side effects to
using decoupled weight decay once it is properly tuned." Suggested ResNet-50 setting: `lr=0.05`,
`momentum=0.9`, `weight_decay=2.0e-3`.

Role on the ladder: the clean foundation. It does not by itself move the time-accuracy frontier; it makes
the LR/schedule/regularization sweeps in every later rung honest. As a regularizer it is expected to yield
diminishing returns when composed with the later regularization methods (label smoothing, stochastic depth,
augmentations).

(Provenance: `docs/source/method_cards/decoupled_weight_decay.md`; the standard-schedule reference accuracy
of 76.6% top-1 for ResNet-50 on ImageNet is from `docs/source/method_cards/scale_schedule.md`.)
