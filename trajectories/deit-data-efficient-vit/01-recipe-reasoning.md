The architecture is not what I want to touch first. A ViT-B trained on ImageNet-1k alone, with the
recipe it was introduced with, sits at 77.91% top-1 — and an off-the-shelf training pipeline, without
changing a single weight of the network, has already pushed that same architecture to 79.35% just by
adjusting how it is trained. That two-point jump from procedure alone, with zero architectural change, is
the whole thesis I need to test before I let myself touch a single Transformer block: the gap between
this convolution-free classifier and a data-efficient convnet might not be an architecture gap at all. It
might be a training-procedure gap that convnets solved for themselves over a decade of tuning, dressed up
to look architectural because nobody has yet run the Transformer through that same decade of tuning
compressed into one pass.

Why would training procedure matter this much specifically for this architecture? Convolutions bake in
locality and translation equivariance directly into the weight-sharing structure of the operator — a
convnet does not have to be shown that a cat shifted three pixels to the left is still a cat, the
convolution's translation equivariance guarantees it for free. Self-attention has no such built-in
constraint: every patch token can attend to every other patch token with a fully learned weighting, so
whatever spatial structure the network ends up respecting has to be induced from the training
distribution itself. On a genuinely huge dataset that induction happens anyway, because the data density
is high enough to make every local pattern the convolution would have hard-coded also empirically true
often enough to learn. On 1.28M images it may not be. That reframes the problem: I do not need to add
locality to the architecture, I need the training procedure to manufacture, through the data the network
actually sees, some of the invariance the convolution would have supplied structurally. Augmentation and
regularization are exactly the tools that can do that job on the data side without touching a single
weight-sharing pattern in the model.

So the design question for this rung is not "which single trick helps" but "which coherent bundle
substitutes, cheaply, for the missing architectural prior, without either starving the network of
capacity to fit at all or leaving it so unconstrained that it just memorizes." I have a toolbox: AdamW,
cosine schedule with warmup, AutoAugment, RandAugment, Mixup, CutMix, random erasing, label smoothing,
stochastic depth, dropout, repeated augmentation, EMA, and a documented safe initialization
(truncated normal, since several untested initializations are known not to converge for this
architecture at all — that is a prerequisite for running anything, not a result). I have to commit to a
specific subset and specific values, on the reasoning available now, and let the numbers arbitrate.

Optimizer and weight decay first, because they gate whether anything else even has a chance to matter.
The as-published ViT recipe already uses AdamW, so I keep it — there is no motivation here to regress to
SGD, an optimizer with no adaptive per-parameter scaling, on a network with no batch normalization to
smooth its loss landscape the way it smooths a convnet's. But the as-published weight decay, 0.3, was
tuned against a training set two orders of magnitude larger than the one available here; a decay
coefficient that strong is a bet that overfitting is the dominant risk, which is exactly backwards for a
model with almost no structural prior being asked to fit a comparatively small, fixed set of images
directly. I am dropping it to 0.05 on the hypothesis that at this data scale the failure mode is closer
to fitting badly than fitting too well, and that the augmentation stack below is a better place to put
anti-overfitting pressure than a blunt global decay term that also throttles useful capacity. Batch size
comes down too, from 4096 to 1024, scaling the learning rate proportionally (base 0.0005 at batch 512) —
not a regularization choice but a compute one, since the lack of batch normalization means nothing in the
architecture depends on a large batch for stable statistics, so I am free to fit the schedule to a single
8-GPU node rather than the many-node setup the original recipe assumed. Cosine decay with a five-epoch
warmup and 300 epochs total round out the schedule: enough epochs to let a heavily augmented, weakly
data-primed model converge, since every epoch under strong augmentation is effectively a harder, more
varied training signal than an epoch of unaugmented images.

For the augmentation bundle, the guiding principle is the one above: manufacture invariances the
convolution does not supply. RandAugment composes a sequence of randomly sampled photometric and
geometric transforms per image, directly diversifying what a "cat" looks like beyond what any single
crop of the raw dataset shows — I am choosing it over AutoAugment because AutoAugment's policy was
searched against a specific convnet's inductive biases already in place, and there is no reason to
inherit a search target tuned for an architecture with a completely different prior; RandAugment's
randomized, untargeted composition seems the safer default for an architecture with none of that prior to
begin with, though I will test rather than assume. Mixup and CutMix both convexly or spatially combine
two training images and their labels; beyond their usual smoothing effect on the decision boundary, they
are also a cheap way to force the class token's attention to spread across more of the patch sequence,
since a CutMix-composited image literally contains two objects and the correct label is a mixture — the
network cannot get away with keying on one dominant local cue the way it might on an unmixed image, which
directly counters the risk of an unconstrained attention pattern collapsing onto a spurious shortcut.
Random erasing knocks out rectangular patches outright, which is a blunter version of the same
occlusion-robustness a convolution's local pooling gives away for free. Stochastic depth randomly drops
whole residual sub-blocks during training, which the augmentation/regularization background specifically
documents as easing convergence for deep transformer stacks in particular, as distinct from ordinary
dropout's per-unit masking — twelve residual blocks stacked with no batch-norm to keep gradients
well-scaled is exactly the setting that guidance targets, so I am including it and, on that same
reasoning, leaving plain dropout out rather than stacking every regularizer the toolbox offers: dropout
masks activations within a block, stochastic depth masks whole blocks, and layering both together on top
of an already-heavy augmentation stack risks under-fitting a network that has no architectural head start
to spare. Label smoothing at the standard 0.1 softens the hard-label cross-entropy target, consistent
with training under transformations (Mixup, CutMix, erasing) that already make the "true" label somewhat
approximate for any given augmented view. Repeated augmentation generates multiple independently
augmented views of the same source image within a batch rather than drawing every batch element from a
distinct source image; combined with heavy per-sample augmentation, this raises the effective diversity
of gradient signal per batch without requiring more distinct images than the dataset has to offer, which
is precisely the kind of data-multiplication a data-limited, prior-free architecture should want.

That is the full bundle: AdamW at wd 0.05, truncated-normal init, RandAugment, Mixup, CutMix, random
erasing, stochastic depth, label smoothing, repeated augmentation, no dropout, 300 epochs, batch 1024. I
am applying it unchanged across all three sizes — Ti, S, and B — since nothing in the reasoning above is
size-specific; if the diagnosis is right, the same procedure fix should help a 5M-parameter model and an
86M-parameter model for the same underlying reason. I also want to know what a completed 224²-trained
model gains from a resolution bump at test/fine-tune time, independent of anything else, so alongside the
headline 224² numbers I am fine-tuning the resulting Base-size model at higher input resolutions with bicubic
interpolation of the positional embeddings (chosen over the more familiar bilinear interpolation on the
stated reasoning that bilinear systematically shrinks the norm of an interpolated position vector
relative to its un-interpolated neighbors, and a vector far outside the norm range the pre-trained
transformer was fit to is not a safe input to give it without a chance to adapt — bicubic approximately
preserves that norm and should be the gentler starting point for fine-tuning). I do not know yet whether
this training-procedure-only change closes the gap to convnets, partially closes it, or falls short in
some size- or resolution-dependent way that would tell me the architecture does need to change after all
— that is exactly what running it will decide.
