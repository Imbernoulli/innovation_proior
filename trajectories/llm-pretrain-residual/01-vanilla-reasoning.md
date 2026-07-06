The residual-flow rule is the whole object I get to design, but a rule bolts onto an otherwise fixed
GPT, and with the rule left at its plainest the model *is* the floor — so the thing to start from is
just training this 24-layer GPT-2 Medium on FineWeb with the residual stream untouched. The scaffold
already hands me that: the default `Block.forward` is the standard Pre-LN block, `x = x + attn(ln_1(x))`
then `x = x + mlp(ln_2(x))`, and the default `GPT.forward` runs a bare loop `for block in h: x =
block(x)`. My step-1 edit is the trivial one — leave the loop exactly as the vanilla template writes it,
adding nothing. The `vanilla` baseline exists only so the rigorous-codebase harness has at least one
edit op per baseline, so the edit is an identity replacement of the block loop with itself; the model
that runs is plain additive Pre-LN.

I want to be precise about *why* this is the right floor and not an arbitrary one, because the entire
ladder is going to be measured against it. The Pre-LN choice is itself the resolution of the residual
lineage in the context, and I should walk that lineage carefully because each ancestor's gap is a clue
to where the floor itself is still weak. Start with plain stacked layers, no skip: compose `L`
width-preserving maps directly. In principle this is enough — a deep stack can represent anything a
shallow one can by idling the extra layers — but train it and the optimizer stalls past a few dozen
layers, and a deeper plain net reaches *higher* training error than a shallower one. That is not a
generalization gap; it is an optimization failure, and the mechanism is the per-layer gain. Perturb the
input by a little and ask how big the perturbation is at the output: each layer multiplies it by its
Jacobian, so across the depth the perturbation is multiplied by a product of Jacobians, on average a
factor `r` per layer. Let me make that concrete with this network's own depth. It has 24 layers, two
sublayers each, so 48 residual writes; put `r = 1.05`, a mere five-percent per-layer gain, and the
end-to-end amplification is `1.05^48 ≈ 10.4` — an order of magnitude blow-up of any input perturbation,
and the transposed product does the same to the gradient. Nudge `r` the other way to `0.95` and I get
`0.95^48 ≈ 0.085`, an eleven-fold attenuation: whatever signal the deepest layer produces has shrunk to
under a tenth by the time it reaches the loss, and the shallow layers' gradients are annihilated on the
way back. The only survivable regime is `r ≈ 1` held at *every* layer, because either side of one
compounds exponentially, and a plain stack has no mechanism to hold it there — nothing in `σ(W x)`
pins the product of Jacobians to one.

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

Pre-LN is the fix that the floor uses: move the norm *inside* the branch, `x ← x + sublayer(LN(x))`, so
each sublayer reads a normalized input but writes its raw output into an unnormalized stream whose
through-path is identity-plus-addition again. The leading `1` is restored, the backward highway is clean,
and the last-layer gradient now shrinks like `1/√L` instead of staying large and depth-independent. Let
me put a number on `1/√L` too, because "shrinks like `1/√L`" should not be a slogan: with `L = 24` that
is `1/√24 ≈ 0.20`, and counted over the 48 sublayer writes it is `1/√48 ≈ 0.14` — a gradient at the deep
end that is a fifth to a seventh of the shallow-end scale, small but *bounded and depth-graded*, not the
depth-independent spike Post-LN hands the top layers. The price of moving the norm inside is that the
stream is no longer reset to a fixed scale between layers, so its expected squared norm grows with
depth — which is precisely why a single final LayerNorm before the head is needed, to soak up that
accumulated scale before the logits. So Pre-LN is not a strawman: it is the strongest *plain* residual
rule, the one the field actually trains with, and that is exactly what makes it the honest floor. Any
depth-flow redesign on this ladder has to beat the best simple thing, not a deliberately crippled one.

Before I name the gaps, I want to check something that the context hands me and that I suspect is
load-bearing: the `c_proj.weight` init rescaling, `std = 0.02/√(2·n_layer)`, one factor per branch
write. With `n_layer = 24` that is `0.02/√48 ≈ 0.00289`, a factor `1/√48` smaller than the default
`0.02`. Why is that there, and what does it tell me about the floor's variance? Each branch's output
variance scales as the square of its projection std, so relative to an un-rescaled branch the write into
the stream carries variance `∝ (1/√48)² = 1/48`. There are exactly 48 such writes, so the total variance
they add to the stream is `48 · (1/48) = 1` — order one. So at initialization the floor is *deliberately
conditioned*: the embedding contributes an O(1) variance, the 48 branch writes contribute another O(1)
between them, and the stream norm at the top is O(1) rather than blown up. That is a real, checkable fact
and it sharpens the whole diagnosis, because it says the variance problem is *not* an init artifact — the
init trick has already flattened it. The variance-climbs-with-depth story is a *training-time* drift: as
the branch weights leave their suppressed initialization and grow, the `1/48` suppression is just a
starting condition, no longer enforced, and the accumulator's natural depth-growth reasserts itself.
The init rescaling delays deep-layer death; it does not prevent it. Good — that is the honest version of
the mechanism, and I will carry it into the next rung.

Now I should name the one property of this floor that every later rung will attack, because it is
load-bearing for the whole climb. The Pre-LN residual stream is a *fixed unit-weight accumulator*. Read
the unrolled recurrence: with `x_{l+1} = x_l + F_l(LN(x_l))`, the state entering layer `l` is
`x_l = x_1 + Σ_{i<l} F_i(LN(x_i))` — the embedding plus every earlier branch output, each added with
coefficient exactly one. Three things follow, and they are the gaps the ladder will exploit. First, the
depth-mixing rule is *rigid*: the sequence axis gets full self-attention, the feed-forward gets a
learned nonlinearity, but the depth axis gets one unweighted running sum with no knob — no way to say
"layer 9 should lean on layer 3's output more than layer 8's," and the same mixing for every token.
Second, the stream variance *climbs with depth* once training has moved the weights off that suppressed
init: each branch reads a normalized input of fixed scale and writes its raw output into an
unnormalized stream, so the accumulated norm grows (linearly in the benign case, faster in the bad one),
and because the branch's LayerNorm divides by that growing scale, the deep blocks' Jacobians shrink
toward the identity. Make that last step explicit: the block Jacobian is `I + (∂F/∂LN)·(∂LN/∂x)`, and
the normalization factor `∂LN/∂x` carries a `1/σ_{x_l}` from dividing by the stream's own scale, so a
large `σ_{x_l}` at the deep end drives the whole second term toward zero and leaves `≈ I` — the deepest
layers become near-identity maps that contribute little, the redundancy that depth-pruning probes keep
finding. Third, the branch is at full strength from step zero: the block is *not* the identity at init,
so the early, chaotic updates of a freshly-random deep stack get written into the stream at full force,
which is precisely the instability that learning-rate warm-up is a band-aid over.

Let me press on each of those three gaps a little, because they are not equally important and I want to
know which one to attack first — this is the real decision at this rung, since the edit itself is a
no-op and the only thing I get to *choose* is what to prioritize next. The rigidity gap is the most
fundamental: the depth axis is the only axis of this network with *no* learned mixing. The sequence axis
has self-attention, which lets every position read a learned, content-dependent combination of every
other position. The feature axis has the MLP, a learned nonlinearity. The depth axis has a single
unweighted running sum — no coefficients, no content-dependence, the same accumulation for every token
and every layer. That is a striking asymmetry once I notice it: I have lavished learned, dynamic mixing
on two of three axes and left the third frozen. But it is also the most *expensive* to attack — genuine
learned mixing over the depth axis means new parameters and new compute that echo attention, and I do not
want to spend that on the first move before I have even confirmed the cheaper mechanisms matter. The
variance gap is the most *mechanical*: it is a measurable, monotone drift that I can in principle
counteract with even a crude per-layer reweighting, and it is the proximate cause of the deep-layer
death — the `1/σ_{x_l}` collapse I just wrote out. The init-strength gap is the most *temporal*: it is
about *when* in training the branches are allowed to write at full force, and it is the reason warm-up
exists at all. These three are not independent — a fix that holds the deep branches back early
(temporal) also slows the variance climb (mechanical), because a branch that writes less early adds less
to the accumulator that the `1/σ` term is dividing by. That coupling is the lever: the cheapest first
move is one that touches *when and how much* each branch writes, since that single knob bears on two of
the three gaps at once without needing any new learned machinery, and I can read its effect as a clean
single-variable delta against whatever this floor scores. The rigidity gap — the full learned
depth-mixing — is the biggest prize but I will not reach for it first; I will earn my way up to it after
the cheap mechanical-and-temporal fix tells me how much of the deep-layer deficit is conditioning versus
genuine missing capacity.

I should put a number on the variance gap too, because "the deep Jacobians shrink toward `I`" is the
whole mechanistic bet and it deserves an estimate rather than a gesture. Take the benign case the init
rescaling nudges the network toward: linear variance growth, `σ²_{x_l} ≈ Θ(l)`, so the stream standard
deviation `σ_{x_l} ≈ √l`. From the first block to the last that is a growth of `√24 ≈ 4.9×` in the
scale of the running stream. Now feed that into the block Jacobian's normalization factor `1/σ_{x_l}`:
the deep layers' non-identity contribution is divided by a `σ` roughly five times larger than the
shallow layers', so their marginal transformation per unit change of input is about `1/4.9 ≈ 0.20` of
what a shallow layer delivers — the top of the stack is operating a fifth of the way toward a pure
identity map compared to the bottom, and that is the *friendly* growth law. If the growth ran the
exponential branch instead of the linear one it would be far worse and far earlier. This is also exactly
what the single final LayerNorm before the head is quietly doing: it has to absorb a ~5× scale spread
between the embedding-dominated bottom and the accumulator-dominated top so the logits see a normalized
representation, which is why the floor needs it and cannot drop it. So deep-layer death is not a rounding
error on a scalar loss — under the friendliest assumption the top third of the stack contributes a fifth
as much marginal transformation as the bottom, and a real chunk of the network's nominal 355M-parameter
capacity is being spent on near-identity maps. That is the capacity the ladder is trying to reclaim, and
it tells me the win, if there is one, should be largest on whatever downstream behavior actually needs
the deep layers to *do* something rather than pass their input through.

I want one honest verification that the Pre-LN-versus-Post-LN gradient story is not just a story, so let
me linearize a two-layer stack and actually compare. Drop the nonlinearities and treat each sublayer as
a linear map `A_k` with a normalization that divides by the running scale `s_k`. Pre-LN: `x_2 = x_1 +
A_1 x_1/s_1`, `x_3 = x_2 + A_2 x_2/s_2`, so `∂x_3/∂x_1 = (I + A_2/s_2)(I + A_1/s_1)`, whose leading term
is the bare `I` — the loss reaches `x_1` with unit coefficient no matter what `A_1, A_2` do. Post-LN:
`x_2 = (x_1 + A_1 x_1)/s_1`, `x_3 = (x_2 + A_2 x_2)/s_2`, so `∂x_3/∂x_1 = (I + A_2)(I + A_1)/(s_1 s_2)` —
every path to `x_1`, including what would have been the identity route, is divided by the product of
running scales `s_1 s_2`, and there is no term free of that denominator. So on the linearized stack the
Pre-LN gradient to the bottom has a scale-free unit component and the Post-LN gradient does not: the
difference the lineage claims is real in the algebra, not just asserted. That is the property the floor
buys, and it is the one thing the climb must never break — every later rung has to keep a clean route
from the loss to the shallow layers even as it makes the depth flow smarter.

Let me state that constraint sharply, because it is the invariant I hand forward to every rung above
this one and I want it written down before I have any freedom to violate it. Whatever I do to the depth
flow — weight it, schedule it, attend over it, split it into parallel copies — the forward map from
embedding to head must keep the leading `I` in its unrolled Jacobian, or something close to it, so that
`r ≈ 1` survives across the 48 writes and the `1.05^48 ≈ 10.4` / `0.95^48 ≈ 0.085` cliffs stay off. Two
corollaries fall out that will constrain every design choice later. First, whatever coefficient I put in
front of the *skip* (the carried stream) had better sit near one at initialization, because that
coefficient is what the identity route is made of; scale the skip down and I reintroduce the
multiplicative decay the residual connection was invented to kill. So the safe place to put any new knob
is on the *branch*, not the skip — dial the branch from nothing up to full and I interpolate cleanly
between the exact identity (`r = 1`, perfectly conditioned, does nothing) and vanilla (the working
floor), never leaving the survivable corridor. Second, any new routing coefficient that depends on the
stream's *content* must not be allowed to depend on the stream's *scale*, because the scale is the
diseased quantity that grows with depth; a router that keys on magnitude would re-import the very
depth-norm drift I am trying to cure, straight back through the routing path. I do not need either
corollary yet — the floor has no knobs at all — but they are the guardrails the cheap first move and
every richer move after it will have to respect, and deriving them now from the floor's own mechanism is
cheaper than rediscovering them after a rung diverges.

One more thing I want settled before I read the floor's number: what a *healthy* run looks like, so I can
tell a real result from a broken one. Because the edit is a no-op, the only way this rung fails is if the
fixed loop itself is misbehaving, and the floor's own conditioning tells me what to expect. The init
rescaling keeps the stream O(1), the Pre-LN highway keeps the deep gradient at the `1/√L`-graded scale,
and the cosine schedule with 4%-of-steps warm-up is exactly the band-aid that lets even a Post-LN-style
stack survive its first steps — so a Pre-LN stack, which needs the band-aid *less*, should train
monotonically down with no loss spikes. The validation cross-entropy should land where a 355M model on
~7B FineWeb tokens sits: I am expecting roughly `2.27–2.29`, which is a token perplexity of
`e^2.28 ≈ 9.8` on held-out FineWeb; a value far above that (say low-2.4s) would mean the highway is
leaking gradient or the schedule is mistuned, and a value implausibly below (high-2.1s) at this token
budget would make me suspect a validation leak rather than a triumph. WikiText-2 and LAMBADA are
out-of-distribution perplexities and will read much higher than the in-domain 9.8 — tens, not single
digits — because they are different corpora; that gap is domain shift, not a defect. The downstream
zero-shot tasks are the noisiest read: at 355M the model is barely competent, so ARC-Easy in the mid-50s
and HellaSwag in the low-30s (HellaSwag's four-way adversarial completion is close to its 25% floor at
this scale) is the honest expectation, and I will not over-read a percentage point of movement in either.
Fixing these expectations now matters because the next rung's win, if it comes, will be *small* — a
conditioning fix on a floor that already trains cleanly cannot produce a large jump — so I need to know
in advance which columns can carry a real signal (validation loss and the deep-layer-sensitive LAMBADA)
and which are noise (the downstream accuracies).

So the vanilla baseline is the weakest rung *by construction*: it has the cleanest backward highway the
plain residual can offer, and no mechanism at all for weighting, scheduling, or re-routing the depth
flow. It will train fine — Pre-LN GPT-2 Medium on 7B tokens is a well-trodden recipe, and I expect a
healthy validation loss in the low-2.2s (a cross-entropy of, say, ~2.28 is a token perplexity of
`e^2.28 ≈ 9.8` on held-out FineWeb, which is where a competently trained model of this size should sit),
sensible WikiText-2 and LAMBADA perplexities in the tens, and downstream accuracies in the mid-50s on
ARC-Easy and low-30s on HellaSwag, since at this scale the model is barely above chance on the harder
zero-shot tasks. What it will *not* do is exploit the residual stream as a designable object, because
there is nothing in the default loop that touches it. There is even a weak falsifiable reading here for
the whole climb: if the deep-layer-death mechanism I derived is real, then the metric most sensitive to
the deep layers — LAMBADA, the long-range last-word completion that leans hardest on the top of the
stack — is the one that should move most when a later rung wakes those layers up, more than a
broad-average metric like WikiText-2 perplexity. I cannot test that at this rung, but it tells me which
column to watch first once I have a second number. That is the floor the next rung has to clear: keep the
clean Pre-LN highway, but stop treating the depth axis as a fixed unit-weight sum — give the residual
flow a knob. The first move I will reach for is the cheapest possible one that changes *when and how
much* each branch writes to the stream without adding real parameters, so that whatever the vanilla
numbers turn out to be, I have a controlled, single-variable delta to read against them. The distilled
module and the literal scaffold fill are in the answer.
