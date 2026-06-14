The regularizer is the whole point, but it sits on top of an invariance objective, and before I can
judge any anti-collapse term I have to see what the invariance objective does *without* one. So the
floor I start from is the barest possible fill of the contract: pull the two views together and add
nothing to stop them from imploding. That is the lower bound by construction, and running it is not a
throwaway — it is the diagnostic that tells me, in measured numbers, exactly how bad collapse is on
this harness, which is the failure every later rung has to beat.

Let me write down what the loop actually hands me, because the contract is narrow and I want to design
to it, not to a paper. `forward(z1, z2)` receives two `[B, D]` projected embeddings — view 1 and view 2
of the same batch of images, each pushed through the same backbone and the same projector — and must
return a dict with a scalar `"loss"`. That is the entire surface. There is no place to put negative
samples beyond what is already in the batch, no second network, no predictor, no stop-gradient hook,
no teacher branch — the harness gives me two embedding tensors and asks for a number. So the simplest
honest thing to optimize is "make the two views agree," and the cleanest scalar that says that is the
mean squared distance between the paired embeddings, `F.mse_loss(z1, z2)`. No coefficient, no margin,
no second term. Whatever survives the augmentations — crop, color jitter, grayscale, flip — is what
this loss declares to be the content, and it asks the encoder to map both views of an image to the same
embedding.

I should be precise about why this is the floor and not merely a weak choice, because the precision is
the whole reason to run it first. The invariance term is, in the metric-learning lineage, exactly the
*attract-only* half of a contrastive pair loss: take the similar-pair branch `½‖G(x₁) − G(x₂)‖²`, treat
two views as the similar pair, and delete the margin repulsion that would push dissimilar pairs apart.
Picture the embeddings as point masses joined by springs whose rest length is zero — every spring pulls
its two endpoints toward coincidence, nothing ever pushes anything apart. Ask gradient descent to relax
that network and it has a configuration that makes *every* spring perfectly happy at once, with exactly
zero energy: put all the points at the same location. If the encoder learns to ignore its input and
emit one constant vector `c`, then for every image both views map to `c`, every paired distance is
zero, the MSE is zero — a global minimum of the objective I wrote. It is not a bad local optimum I might
dodge with luck; the constant map is *the* global minimizer, and it carries no information about the
image at all. There is no internal force opposing the collapse because the loss only ever asks for
energy to be lowered and never asks for it to be raised anywhere, so the trivial way to make it low
everywhere I look is to make the encoder flat. That missing counter-pressure — the repulsion, the
negatives, the decorrelation, whatever a real method supplies — is precisely the thing this baseline
omits, and naming the omission is what makes the rest of the ladder legible.

There is one detail of *this* harness that makes the collapse louder than the pure argument suggests,
and it is worth pinning down because it shapes what the numbers will look like. The metric is not the
regularizer's own loss; it is a **linear probe trained online on the frozen, detached backbone
features**. The probe never sees the projector output `z`; it reads the 512- or 2048-dim representation
straight off the backbone and fits a 10-way classifier to it each step. So the question the metric
actually asks is: *after the encoder has been driven by this loss, do the backbone features linearly
separate the ten CIFAR classes?* If the encoder collapses — features with little variance, or all
information crammed into one direction — a linear classifier on top has nothing to separate and lands
near chance. CIFAR-10 has ten roughly balanced classes, so chance is 10%. I therefore expect the
naive run to sit not far above 10% on every backbone: a collapsed or near-collapsed representation,
with whatever residual signal the BatchNorm in the projector and the joint probe gradient happen to
leak through. It will not be exactly 10%, because the projector's batch-norm layers inject a little
per-feature variance and the probe is trained on whatever the backbone does retain early before the
invariance term finishes pulling everything together, but it should be unambiguously broken — a
two-digit accuracy that any real regularizer beats by a wide margin.

There is a subtle wrinkle in this harness that I should think through, because it could in principle
*soften* the collapse and I want to know whether to count on it. The training step does not minimize the
regularizer alone: it minimizes the regularizer's `"loss"` *plus* a cross-entropy term from the online
linear probe, both backpropagated in the same `backward`. But the probe reads the backbone features
through a `detach()`, so the probe's gradient flows only into the probe's own weights, never back into
the encoder. That means the encoder is shaped purely by my MSE invariance term; the probe is a passive
reader fit to whatever the encoder produces. So no, the joint probe does not push back against collapse
— it cannot, by the detach. The encoder is free to march to the constant, and the probe simply reports
how unseparable the resulting features are. Good: that confirms the floor is a clean test of the
objective and not contaminated by the metric's own training signal.

A note on what I deliberately do *not* do here. I do not touch `CONFIG_OVERRIDES`; the default projector
`2048 → 2048` is fine, because reshaping the projector cannot rescue a loss that has no anti-collapse
term — collapse is a property of the objective, not the width. I do not add an epsilon, a temperature,
a normalization of the embeddings, or any stabilizer, because every one of those would be a partial
anti-collapse mechanism smuggled in, and the point of the floor is to have *none*. I do not even
l2-normalize the embeddings before the MSE, which some invariance losses do — normalization onto the
sphere is itself a weak variance constraint (it forbids the all-zero vector), and I want the floor to
carry no such crutch. The loss is one line. I return the scalar under `"loss"` plus an
`"invariance_loss"` echo so the log shows the two are identical — there is nothing else in the
objective.

Now reason about what running this must produce across the three backbones, because the cross-backbone
spread is itself information for the next rung. All three see the identical loss; the only difference is
the backbone capacity and the feature width feeding the probe (512 for ResNet-18 and ResNet-34, 2048
for ResNet-50). Collapse is not a capacity problem — a bigger network collapses just as completely, and
arguably *more* cleanly, because it has more freedom to find the constant solution. So I do not expect
the larger backbones to rescue accuracy; if anything the three should all sit in the same broken band,
with only noise separating them, and possibly ResNet-50's wider features giving the probe a marginally
larger chance-level cushion. What I am watching for is exactly that: a tight, low band across all three,
confirming the failure is the *objective's* collapse and not anything backbone-specific. If instead one
backbone scored well, that would mean the invariance term alone was somehow not collapsing on that
architecture, and I would have to rethink — but the spring argument says that should not happen.

The diagnosis the floor sets up for step 2 is already written into the construction. The failure is not
a learning-rate problem, not a schedule problem, not a backbone problem; it is that the objective has
no term that raises the energy of the collapsed configuration. So the fix is forced in *kind* even
before I pick the exact one: I must add an anti-collapse term that makes the constant map — and the
softer dimensional collapse where all variance crowds into a low-rank subspace — a high-loss state
rather than a minimizer. The metric-learning ancestor would restore the margin repulsion against
negatives; the information-maximization line would instead demand that the embedding dimensions stay
spread and decorrelated, which is the branch I expect to climb on this harness because it needs no
negatives and no second network — exactly the machinery this contract does not give me room for. So
step 1's job is done the moment it produces a broken number: it converts "collapse is a theoretical
worry" into "collapse costs roughly N points of val_acc on this exact setup," and that gap is the
budget every later regularizer is trying to recover. I expect the naive baseline to be unambiguously
the weakest thing on the board — a two-digit accuracy near the ten-class chance floor on all three
backbones — and the size of that hole is what the next rung's decorrelation-or-variance term has to
fill. (The distilled module and the one-line scaffold edit are in the answer.)
