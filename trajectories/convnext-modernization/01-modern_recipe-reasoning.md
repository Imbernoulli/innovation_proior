Before touching a single convolution, I want to settle a confound that sits
underneath the whole exploration. ResNet-50's 76.13% and Swin-T's 81.30% are
not a clean architecture-vs-architecture comparison. They were produced under
two different training procedures: ResNet-50's number comes from the
original recipe — 90 epochs, plain SGD, light or no modern augmentation —
while Swin-T was trained with the newer Transformer-family recipe: AdamW,
hundreds of epochs, and a heavy bundle of Mixup, CutMix, RandAugment, Random
Erasing, Stochastic Depth, and Label Smoothing. A 5.2-point gap measured
across that double change (different network *and* different optimization
procedure) tells me nothing about which factor is responsible. If I start
editing the network now, every later number will still carry this same
confound, and I won't be able to attribute movement to architecture at all.

So the very first move has to isolate the two factors, and the cheapest way
to do that is to hold architecture at exactly zero and vary only the recipe.
Concretely: take the unmodified ResNet-50 — same bottleneck blocks, same stem
(7x7 stride-2 conv + maxpool), same stage depths (3, 4, 6, 3), same 64-then-256
channel bottleneck widths — and retrain it from scratch with a recipe that
matches what the Transformer-family papers actually use, rather than
ResNet's original one. This isn't a new idea I have to invent; it is
already documented that modern training techniques by themselves can lift a
plain ResNet-50 substantially, so the correct first experiment is exactly
this: measure how much of that lift is available for free, before any
architectural credit is claimed.

What does "match the Transformer-family recipe" mean concretely, and how do
I choose among the pieces? I don't want to cherry-pick individual
augmentation tricks and tune each one separately — that would turn rung 1
into its own multi-step search and burn budget on a question (which
knob matters how much) that isn't the one this rung is trying to answer.
The question rung 1 asks is coarser: is the training-procedure gap large or
small? So I take the recipe as a bundle, matching what DeiT and Swin
already use as a package, rather than assembling it piece by piece:

- **Optimizer and schedule.** Swap SGD for AdamW, and extend training from
  the original 90 epochs to 300. Transformers are known to need longer
  schedules and to respond better to adaptive optimizers than to SGD; there
  is no reason a ConvNet trained for the same length under the same
  optimizer wouldn't at least partially benefit, since the underlying
  optimization problem (fitting a deep net to 1.28M images) doesn't change
  because the architecture is convolutional rather than attention-based.
- **Data augmentation: Mixup, CutMix, RandAugment, Random Erasing.** All
  four are architecture-agnostic — they operate on the input image and
  label, never touch a single network weight — so there's no reason to
  expect them to only work for Transformers. Their whole purpose is
  regularizing a large-capacity model trained for many epochs against
  overfitting a fixed 1.28M-image set, which is exactly the regime a
  300-epoch run enters that a 90-epoch run does not.
- **Regularization: Stochastic Depth, Label Smoothing.** Same logic —
  neither depends on the block internals being convolutional or
  attention-based. Stochastic Depth randomly drops residual branches during
  training, which only needs a residual structure to exist (ResNet has
  one); Label Smoothing only touches the loss.
- **LayerScale.** This one is more architecture-adjacent — it's a learned
  per-channel gate multiplying a residual branch's output before the
  addition, initialized near zero so training starts close to identity and
  the scale grows only as far as the optimization wants it to. But it's a
  strict generalization of "add nothing extra" (it can converge to any
  scale, including something close to 1 if that's optimal), so wrapping
  every residual branch in ResNet-50 with this gate is a training-technique
  addition, not an architectural redesign — it doesn't change what
  operations exist in the block, only how their contribution is weighted
  during optimization. I'll fold it in.
- **What I will *not* do**: use Exponential Moving Average (EMA) of weights.
  EMA smooths the trained weights over the tail of training and is common
  in the Transformer recipes, but averaging weights interacts with
  BatchNorm's running statistics in a way plain SGD-style ResNet training
  was never designed around — the EMA weights and the BN running mean/var
  computed for the instantaneous weights can drift out of sync. Since
  ResNet-50 is staying BatchNorm-based in this rung (I'm not touching
  normalization yet — that's a much later, architecture-adjacent decision),
  the safer choice is to leave EMA out here and revisit it only once/if the
  network's normalization layer itself changes.

One design decision cuts across everything that follows, not just this rung:
once I fix this recipe, I commit to reusing it, unchanged, for every future
step. The entire point of isolating training procedure from architecture at
rung 1 is wasted if a later rung silently changes the optimizer or
augmentation strength too — then I'd be right back to confounding two
factors at once, just at a different point in the ladder. So whatever number
this rung produces becomes the fixed "training procedure" term, and every
subsequent rung's delta is attributable to architecture alone by
construction, as long as I hold the recipe still.

What do I expect? I genuinely don't know the split in advance, and that's
the point of running this as its own measured rung rather than assuming an
answer. Two extremes bound my uncertainty. If the training-procedure gap
turns out to be most of the 5.2 points between 76.1 and 81.3, that would
mean the "architecture" story is smaller than it looks, and the network
redesign should be a light touch pursued mainly for other reasons (FLOPs
efficiency, simplicity) rather than as the primary route to matching Swin-T.
If instead the gap barely moves, that tells me the accuracy difference is
overwhelmingly architectural, and the coming redesign steps (macro layout,
grouped/depthwise convolution, inverted bottlenecks, kernel size, per-layer
micro design) are carrying essentially the whole burden, which raises the
bar for how aggressively I should be willing to change the block. Given that
this recipe bundle is specifically the one credited elsewhere with
substantial gains on plain ResNet-50 — otherwise it wouldn't have become the
default for training modern vision backbones — my prior leans toward a real,
non-trivial improvement, plausibly several points, but I want the number
before committing to any story about how much of the remaining gap is
"really" architectural.

One more thing this rung has to produce besides the accuracy number: a
FLOPs sanity check. Since I am not touching the network at all — same
convolutions, same channel counts, same everything — FLOPs must come out
byte-for-byte identical to the un-retrained baseline's 4.09 GFLOPs. If it
doesn't, that's a sign I accidentally changed something architectural while
assembling the recipe, and I'd need to go find the bug before trusting the
accuracy number at all. So the test this rung commits to is narrow and
exact: retrain the unmodified ResNet-50 architecture end-to-end under this
fixed AdamW/augmentation/regularization/LayerScale recipe for 300 epochs, no
EMA, and read off top-1 accuracy and GFLOPs, three seeds, mean and std. The
result becomes the frozen baseline recipe for every step that follows, and
the frozen departure point for asking whether the network itself needs to
change at all.
