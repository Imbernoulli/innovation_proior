The training-procedure fix closed most of the distance to convnets on its own: the Base size reached 81.8% (224²),
83.1% after a 384² fine-tune, both up sharply from the 77.91–79.35% starting bar, with the same
architecture and the same dataset throughout. That confirms the diagnosis — this was substantially a
training-procedure gap — but it also tells me the procedure lever is close to exhausted for what it can
do alone. The recipe already stacks RandAugment, Mixup, CutMix, random erasing, stochastic depth, label
smoothing, and repeated augmentation; piling on more of the same category of fix has diminishing room
left to work with. The next lever available in the toolbox is qualitatively different: instead of only
manufacturing invariances from the raw data, bring in an external signal — a teacher classifier that
already encodes visual structure this architecture had to learn from scratch. If a convnet's convolution
supplies locality and translation equivariance structurally, then a convnet *classifier's output
distribution*, in principle, carries some trace of having been shaped by that same structural prior even
after it is reduced to a soft or hard label. Distillation is a way to let a Transformer, which has no
such structural prior of its own, absorb some of that trace indirectly through supervision rather than
through architecture. That is the hypothesis for this rung, and the default teacher is fixed for it and
every distillation rung after it: a RegNetY-16GF convnet, 84M parameters, 82.9% top-1 on its own,
trained on the same data with the same augmentation as the student, so any effect I measure is
attributable to the distillation signal itself and not to the teacher having seen images the student
never saw.

The toolbox gives two concrete ways to define the distillation signal, and I do not yet know which suits
this recipe better, so I want to test both under otherwise identical conditions: same architecture (class
token only — I am not touching the token layout yet, only the loss), same recipe from the previous rung,
same teacher, same three sizes. Soft distillation keeps the teacher's full output distribution and
minimizes a temperature-softened KL divergence between student and teacher softmax outputs, blended with
the ground-truth cross-entropy via a mixing coefficient λ and rescaled by τ² to keep the softened
gradient magnitude comparable across temperature choices. It requires committing to two extra
hyperparameters, τ and λ, and it is designed to transfer the teacher's *relative confidence across
classes* — the fact that a photo of a wolf gets some non-trivial probability mass on "husky" as well as
"wolf" is itself informative, and soft distillation is built to pass that structure through.

Hard-label distillation instead takes only the teacher's argmax prediction on the same input the student
sees, and treats it exactly like a second ground-truth label: half the loss is ordinary cross-entropy
against the true label, half is cross-entropy against the teacher's hard decision. It is parameter-free —
no τ, no λ to select — which is a real simplicity advantage on its own, since every additional
hyperparameter is another dimension I would otherwise have to sweep. It also has a property the soft
variant does not automatically have: because the teacher is evaluated on the *same* augmented crop the
student trains on, its hard decision can track what is actually visible in that crop even when Mixup,
CutMix, or an aggressive random erasing has changed the image enough that the original dataset label is
no longer a fully accurate description of the content shown. A hard pseudo-label recomputed per crop is,
in that sense, targeting the same thing the true label targets — "what class is in this image" — just
with a second, independently-derived estimate of it, rather than adding a differently-shaped signal on
top.

That last point is what makes me suspect, without yet knowing, that hard distillation might have more
room to help here specifically because of what the recipe from the previous rung already contains. Label
smoothing already softens the hard cross-entropy target by construction; Mixup and CutMix already inject
continuous, non-one-hot targets into a large fraction of training batches. Soft distillation's contribution
— a smoothed, multi-class target distribution — heavily overlaps in kind with signal the recipe is already
supplying from other sources, so its marginal information beyond what training already provides may be
smaller than it would be in a recipe without heavy label mixing. Hard distillation's contribution is
different in kind: it is a second, independently-produced *decision* about the image's class, not another
softening of the same one-hot target the student already has. Whether that difference in kind translates
into a measurable difference in accuracy is exactly what I cannot know without running both, and it is
possible the two end up close, or that soft distillation still wins on some size where the recipe's own
smoothing signal is weaker relative to model capacity. I am deliberately not committing to a
prediction beyond that qualitative asymmetry, since the whole point of running both under matched
conditions is to let the comparison, not my priors, decide which loss the next rung inherits.

Concretely: freeze the rung-1 recipe and rung-1 architecture (class token only, plain linear classifier
on its output) for both variants, add nothing else, and run each of the two objectives — soft KD at the
standard temperature-3, λ=0.1 setting recommended for this kind of setup, and hard-label distillation
with no extra hyperparameters — against the fixed RegNetY-16GF teacher, on all three sizes, reporting the
same 224² and 384²-fine-tuned top-1 numbers as before so the comparison to rung 1's undistilled baseline
is direct. Whichever objective wins becomes the fixed loss for every later rung that touches the
distillation architecture itself, the same way the rung-1 recipe is now fixed underneath both of these
variants.
