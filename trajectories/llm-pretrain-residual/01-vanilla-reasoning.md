The residual-flow rule is the whole object I get to design, but a rule bolts onto an otherwise fixed
GPT, and with the rule left at its plainest the model *is* the floor — so the thing to start from is
just training this 24-layer GPT-2 Medium on FineWeb with the residual stream untouched. The scaffold
already hands me that: the default `Block.forward` is the standard Pre-LN block, `x = x + attn(ln_1(x))`
then `x = x + mlp(ln_2(x))`, and the default `GPT.forward` runs a bare loop `for block in h: x =
block(x)`. My first edit is the trivial one — leave the loop exactly as the template writes it, adding
nothing. The edit surface requires at least one edit op, so this is an identity replacement of the block
loop with itself; the model that runs is plain additive Pre-LN.

I want to be precise about *why* this is the right floor and not an arbitrary one, because everything I
build later is going to be measured against it. The Pre-LN choice is itself the resolution of the
residual lineage in the context, and each ancestor's gap is a clue to where the floor itself is still
weak. Start with plain stacked layers, no skip: compose `L` width-preserving maps directly. In principle
this is enough — a deep stack can represent anything a shallow one can by idling the extra layers — but
train it and the optimizer stalls past a few dozen layers, and a deeper plain net reaches *higher*
training error than a shallower one. That is not a generalization gap; it is an optimization failure, and
the mechanism is the per-layer gain. Perturb the input by a little and ask how big the perturbation is at
the output: each layer multiplies it by its Jacobian, so across the depth the perturbation is multiplied
by a product of Jacobians, on average a factor `r` per layer. Make that concrete with this network's own
depth. It has 24 layers, two sublayers each, so 48 residual writes; put `r = 1.05`, a mere five-percent
per-layer gain, and the end-to-end amplification is `1.05^48 ≈ 10.4` — an order of magnitude blow-up of
any input perturbation, and the transposed product does the same to the gradient. Nudge `r` the other way
to `0.95` and I get `0.95^48 ≈ 0.085`, an eleven-fold attenuation: whatever signal the deepest layer
produces has shrunk to under a tenth by the time it reaches the loss, and the shallow layers' gradients
are annihilated on the way back. The only survivable regime is `r ≈ 1` held at *every* layer, because
either side of one compounds exponentially, and a plain stack has no mechanism to hold it there — nothing
in `σ(W x)` pins the product of Jacobians to one.

The residual connection (`x_{l+1} = σ(x_l + F(x_l))`) was the escape: add an identity skip so the block
only has to learn a nudge to identity, and — this is the load-bearing part — the additive `1` in the
unrolled backward product gives the top gradient a route to every shallow layer that is *not* multiplied
by all the Jacobians above it. Unroll one step of the backward pass: `∂x_{l+1}/∂x_l = I + ∂F/∂x_l`, so
the gradient reaching layer `l` is a product of terms each shaped `(I + J_k)`. Expand that product and
the leading term is the bare `I·I·…·I = I` — a clean, unattenuated path from the loss straight to layer
`l` — while all the `r^L`-style compounding lives in the *other* terms of the expansion, the ones that
carry at least one Jacobian factor. That single additive `1` is what made hundreds of layers trainable.
But it does not pin `r = 1`: the branch `F` fires at full strength at init, so the block is *not* the
identity at step zero, and the output variance of a residual block still compounds with depth. The skip
tames the *worst* of `r^L`; it does not set `r = 1`. Then the Transformer added normalization, and
*where* it goes turns out to matter enormously. Post-LN normalizes after the addition, `x ← LN(x +
sublayer(x))`, which puts a LayerNorm Jacobian *on* the residual highway — so the backward path becomes a
product of normalization Jacobians, exactly the multiplicative `∏ J_k` structure the skip was meant to
avoid. The `I` I just isolated is gone, buried inside the norm. Concretely, at init this gives large,
depth-imbalanced gradients near the output layer, which is why a full learning rate immediately
destabilizes a deep Post-LN Transformer and why the field bolts on learning-rate warm-up: ramp the rate
up slowly so the first updates are tiny enough to survive. Warm-up is a band-aid over bad init-time
signal propagation, and the tell that it is the *architecture* and not the optimizer is that the warm-up
dependence persists even when you swap Adam for plain SGD — a purely optimizer-side fix would not care
which optimizer it is patching.

Pre-LN is the fix the floor uses: move the norm *inside* the branch, `x ← x + sublayer(LN(x))`, so each
sublayer reads a normalized input but writes its raw output into an unnormalized stream whose
through-path is identity-plus-addition again. The leading `1` is restored, the backward highway is clean,
and the last-layer gradient now shrinks like `1/√L` instead of staying large and depth-independent. Put a
number on `1/√L`: with `L = 24` that is `1/√24 ≈ 0.20`, and counted over the 48 sublayer writes it is
`1/√48 ≈ 0.14` — a gradient at the deep end that is a fifth to a seventh of the shallow-end scale, small
but *bounded and depth-graded*, not the depth-independent spike Post-LN hands the top layers. The price of
moving the norm inside is that the stream is no longer reset to a fixed scale between layers, so its
expected squared norm grows with depth — which is precisely why a single final LayerNorm before the head
is needed, to soak up that accumulated scale before the logits. So Pre-LN is not a strawman: it is the
strongest *plain* residual rule, the one the field actually trains with, and that is exactly what makes it
the honest floor. Any depth-flow redesign here has to beat the best simple thing, not a deliberately
crippled one.

One thing the context hands me is load-bearing and worth pinning down: the `c_proj.weight` init
rescaling, `std = 0.02/√(2·n_layer)`, one factor per branch write. With `n_layer = 24` that is
`0.02/√48 ≈ 0.00289`, a factor `1/√48` smaller than the default `0.02`. Each branch's output variance
scales as the square of its projection std, so relative to an un-rescaled branch the write into the stream
carries variance `∝ (1/√48)² = 1/48`. There are exactly 48 such writes, so the total variance they add is
`48 · (1/48) = 1` — order one. So at initialization the floor is *deliberately conditioned*: the embedding
contributes an O(1) variance, the 48 branch writes contribute another O(1) between them, and the stream
norm at the top is O(1) rather than blown up. That sharpens the whole diagnosis, because it says the
variance problem is *not* an init artifact — the init trick has already flattened it. The
variance-climbs-with-depth story is a *training-time* drift: as the branch weights leave their suppressed
initialization and grow, the `1/48` suppression is just a starting condition, no longer enforced, and the
accumulator's natural depth-growth reasserts itself. The init rescaling delays deep-layer death; it does
not prevent it.

Now name the one property of this floor that every later move will attack. The Pre-LN residual stream is a
*fixed unit-weight accumulator*. Read the unrolled recurrence: with `x_{l+1} = x_l + F_l(LN(x_l))`, the
state entering layer `l` is `x_l = x_1 + Σ_{i<l} F_i(LN(x_i))` — the embedding plus every earlier branch
output, each added with coefficient exactly one. Three things follow, and they are the gaps to exploit.
First, the depth-mixing rule is *rigid*: the sequence axis gets full self-attention, the feed-forward gets
a learned nonlinearity, but the depth axis gets one unweighted running sum with no knob — no way to say
"layer 9 should lean on layer 3's output more than layer 8's," and the same mixing for every token.
Second, the stream variance *climbs with depth* once training has moved the weights off that suppressed
init: each branch reads a normalized input of fixed scale and writes its raw output into an unnormalized
stream, so the accumulated norm grows, and because the branch's LayerNorm divides by that growing scale,
the deep blocks' Jacobians shrink toward the identity. Explicitly, the block Jacobian is `I +
(∂F/∂LN)·(∂LN/∂x)`, and the normalization factor `∂LN/∂x` carries a `1/σ_{x_l}` from dividing by the
stream's own scale, so a large `σ_{x_l}` at the deep end drives the whole second term toward zero and
leaves `≈ I` — the deepest layers become near-identity maps that contribute little, the redundancy that
depth-pruning probes keep finding. Third, the branch is at full strength from step zero: the block is
*not* the identity at init, so the early, chaotic updates of a freshly-random deep stack get written into
the stream at full force, which is precisely the instability warm-up is a band-aid over.

These three gaps are not equally important, and the only thing I actually get to *choose* at this step is
what to attack first, since the edit itself is a no-op. The rigidity gap is the most fundamental — the
depth axis is the only axis of this network with *no* learned mixing, a striking asymmetry once I notice
it — but it is also the most *expensive*, needing new parameters and compute that echo attention, and I do
not want to spend that before I have confirmed the cheaper mechanisms matter. The variance gap is the most
*mechanical*: a measurable, monotone drift I can counteract with even a crude per-layer reweighting, and
it is the proximate cause of the deep-layer death. The init-strength gap is the most *temporal*: it is
about *when* the branches are allowed to write at full force. These are coupled — a fix that holds the
deep branches back early also slows the variance climb, because a branch that writes less early adds less
to the accumulator that the `1/σ` term is dividing by. That coupling is the lever: the cheapest first move
is one that touches *when and how much* each branch writes, since that single knob bears on two of the
three gaps at once without any new learned machinery, and I can read its effect as a clean single-variable
delta against the floor. The full learned depth-mixing is the biggest prize but I will earn my way up to
it after the cheap fix tells me how much of the deep-layer deficit is conditioning versus genuine missing
capacity.

The variance gap deserves an estimate rather than a gesture, because "the deep Jacobians shrink toward
`I`" is the whole mechanistic bet. Take the benign case the init rescaling nudges the network toward:
linear variance growth, `σ²_{x_l} ≈ Θ(l)`, so `σ_{x_l} ≈ √l`. From the first block to the last that is a
growth of `√24 ≈ 4.9×` in the running stream's scale. Feed that into the block Jacobian's `1/σ_{x_l}`
factor: the deep layers' non-identity contribution is divided by a `σ` roughly five times larger than the
shallow layers', so their marginal transformation per unit input change is about `1/4.9 ≈ 0.20` of what a
shallow layer delivers — and that is the *friendly* growth law. This is also exactly what the single final
LayerNorm quietly does: it absorbs a ~5× scale spread between the embedding-dominated bottom and the
accumulator-dominated top so the logits see a normalized representation, which is why the floor cannot
drop it. So deep-layer death is not a rounding error — under the friendliest assumption the top third of
the stack contributes a fifth as much marginal transformation as the bottom, and a real chunk of the
355M-parameter capacity is being spent on near-identity maps. That is the capacity to reclaim, and it
tells me the win, if there is one, should be largest on whatever downstream behavior actually needs the
deep layers to *do* something rather than pass their input through.

So the constraint I hand forward, before I have any freedom to violate it: whatever I do to the depth flow
— weight it, schedule it, attend over it, split it into parallel copies — the forward map from embedding
to head must keep the leading `I` in its unrolled Jacobian, or something close to it, so `r ≈ 1` survives
across the 48 writes and the `1.05^48`/`0.95^48` cliffs stay off. Two corollaries fall out. First,
whatever coefficient I put in front of the *skip* had better sit near one at initialization, because that
coefficient is what the identity route is made of; scale the skip down and I reintroduce the
multiplicative decay the residual connection was invented to kill. So the safe place for any new knob is
on the *branch*, not the skip — dial the branch from nothing up to full and I interpolate cleanly between
the exact identity (`r = 1`, perfectly conditioned, does nothing) and vanilla (the working floor), never
leaving the survivable corridor. Second, any new routing coefficient that depends on the stream's
*content* must not depend on the stream's *scale*, because the scale is the diseased quantity that grows
with depth; a router that keys on magnitude would re-import the depth-norm drift straight back through the
routing path. I do not need either corollary yet — the floor has no knobs — but they are the guardrails
every richer move will have to respect.

Because the edit is a no-op, the only way this fails is if the fixed loop itself misbehaves, and the
floor's own conditioning tells me what to expect: the init rescaling keeps the stream O(1), the Pre-LN
highway keeps the deep gradient at the `1/√L`-graded scale, and the cosine schedule with 4%-of-steps
warm-up lets even a Post-LN-style stack survive its first steps — so a Pre-LN stack, which needs the
band-aid *less*, should train monotonically down with no loss spikes. The validation cross-entropy should
land where a 355M model on ~7B FineWeb tokens sits: roughly `2.28`, a token perplexity of `e^2.28 ≈ 9.8`;
low-2.4s would mean the highway is leaking gradient or the schedule is mistuned, and high-2.1s at this
token budget would make me suspect a validation leak rather than a triumph. WikiText-2 and LAMBADA are
out-of-distribution perplexities and will read much higher than the in-domain 9.8 — tens, not single
digits — because they are different corpora; that gap is domain shift, not a defect. The downstream
zero-shot tasks are the noisiest read: at 355M the model is barely competent, so ARC-Easy in the mid-50s
and HellaSwag in the low-30s (its four-way adversarial completion is close to its 25% floor at this scale)
is the honest expectation, and I will not over-read a percentage point in either. Fixing these now matters
because the next move's win, if it comes, will be *small* — a conditioning fix on a floor that already
trains cleanly cannot produce a large jump — so I need to know which columns can carry a real signal
(validation loss and the deep-layer-sensitive LAMBADA) and which are noise (the downstream accuracies).
There is even a weak falsifiable reading for the whole climb: if the deep-layer-death mechanism is real,
then LAMBADA — the long-range last-word completion that leans hardest on the top of the stack — is the
column that should move most when a later move wakes those layers up, more than a broad-average metric
like WikiText-2. I cannot test that yet, but it tells me which column to watch once I have a second number.

So the vanilla baseline is the weakest starting point *by construction*: the cleanest backward highway the
plain residual can offer, and no mechanism at all for weighting, scheduling, or re-routing the depth flow.
That is the floor the next step has to clear: keep the clean Pre-LN highway, but stop treating the depth
axis as a fixed unit-weight sum — give the residual flow a knob. The first move I reach for is the
cheapest possible one, changing *when and how much* each branch writes to the stream without adding real
parameters, so that whatever the vanilla numbers turn out to be, I have a controlled, single-variable
delta to read against them. The distilled module and the literal scaffold fill are in the answer.
