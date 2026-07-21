ProRes moved the floor, and it moved it in exactly the place the mechanism predicted — but before I
design against the number I want to squeeze the two-row table for everything it says, because a single
seed's scalars still carry shape. Validation loss went `2.2763 → 2.2707`, a drop of `0.0056` nats, about
a quarter of a percent; in token perplexity that is `e^2.2763 = 9.741` down to `e^2.2707 = 9.686`, a
ratio of `0.9944`. Modest, exactly as a conditioning fix should be — I was fighting for fractions of a
nat and I got fractions of a nat, no capacity miracle. Now the secondary columns, read as *relative*
moves, because that is where the mechanism hides. WikiText-2 barely budged, `44.28 → 44.11`, a relative
improvement of `0.17/44.28 ≈ 0.38%`. LAMBADA dropped from `70.09` to `67.21`, a relative improvement of
`2.88/70.09 ≈ 4.1%`. Put those two side by side: the LAMBADA gain is about `4.1/0.38 ≈ 11×` the
WikiText-2 gain in relative terms. That asymmetry is not noise — it is the falsifiable signature I wrote
down at the floor and again at step 2, and it came in loud. LAMBADA scores the model on the final word of
a passage that needs long-range context, so it leans hardest on the top of the stack; WikiText-2 is a
broad next-token average any competent middle layer already serves. An 11× asymmetry in favor of the
deep-layer-sensitive metric says the shallow-to-deep warm-up did what it was supposed to: it woke the
deep layers up a little. ARC-Easy rose `54.12 → 55.35` (`+1.23`) and HellaSwag held at `33.82 → 33.91`
(`+0.09`, inside the noise I flagged for a task near its four-way chance floor). So the diagnosis is
confirmed — deep layers were under-used, and protecting them early helps — and the confirmation is
concentrated exactly where the theory said it would be.

But the same table exposes the schedule's ceiling, and it is the arithmetic I already worked out at step
2: with `T = 1000` the schedule strands the top third of the stack — layers 14 through 24 never reach
`α = 1` inside the 13,535 steps, layer 24 finishing at `≈ 0.56`. Read against the LAMBADA move that cuts
two ways. On one hand ProRes's deep-layer win is *real but incomplete* — the layers it helped most
(LAMBADA leans on them) were precisely the ones still held back at the final step, so there was more
sitting on the table. On the other it indicts the whole *fixed-schedule* idea for this budget: a schedule
meant to relax back to vanilla runs out of steps before it finishes relaxing, stranding exactly the deep
layers I most want at full strength. A schedule dictates *one* trajectory per layer, and here that
trajectory does not even complete. I want to keep what worked — gentle, conditioned control of how much
each branch writes — but hand the magnitude to the network so it can *set* each layer's strength directly
and *persistently*, with no clock running out. And I want to attack a second weakness the schedule never
touched at all.

Take the magnitude first. The natural move is the one ProRes deliberately avoided: stop *prescribing* the
residual weight and *learn* it. Put a scalar in front of each layer's residual write and let gradient
descent set it — the scalar-on-the-branch idea I flagged at step 2, ReZero and SkipInit, where each
branch gets a learnable `λ` and the block becomes `x_{l+1} = x_l + λ_l·F(LN(x_l))`. The appeal is direct:
a layer that wants to contribute more grows its `λ` above 1, a layer that wants to hold back shrinks it,
and the per-layer strengths are discovered rather than dictated by a clock. The classic version inits
`λ = 0` so the block starts at the exact identity, buying the well-conditioned start ProRes also had. But
I have a measured fact that argues against the zero init *here*, and it is worth doing the arithmetic
rather than gesturing at it. Vanilla, the all-`λ=1` model, already trains cleanly to 2.2763, and
ProRes — which is `λ` ramping from 0 toward 1 — only beat it by `0.0056`. So at 24 layers and 13.5k steps
the init-time conditioning problem is evidently mild; the identity start bought almost nothing. Now count
what a zero init *costs* on this budget. If I zero-init the 48 residual weights, the network spends the
opening stretch of a 13,535-step run growing them back toward 1 before it even has its vanilla capacity
online — and I just computed that ProRes, warming from zero on a *prescribed* monotone schedule, could
not finish warming the deep layers in the budget I have. A *learned* zero-init has no schedule
guaranteeing even monotone growth; the optimizer could dawdle on the deep `λ` exactly as ProRes's clock
did, or worse, and there is no shallow-to-deep ordering imposed on top. Paying the identity-start tax to
buy a `0.0056`-nat conditioning benefit, on a run too short to even collect it at depth, is a bad trade.
So the better init for *this* regime is `λ = 1`: start at exactly the vanilla residual that already works
and let the network *adjust* the weights from there rather than *build them up* from zero. Persistence and
per-layer freedom without re-earning the floor — that is the honest reading of the prores numbers, and it
says do not pay for the conditioning twice.

Now the second weakness, the one the schedule never addressed and the larger prize. Go back to the
unrolled stream, `x_l = x_1 + Σ_{i<l} F_i(LN(x_i))`. The token embedding `x_1` — here the post-dropout,
post-position representation entering the first block — is the one signal that carries pure *token
identity*: which token is actually here, before any attention has mixed it across positions or any MLP
has transformed it. In the additive stream that embedding survives only as one term in a growing sum, and
I can put a number on how fast it drowns. From the floor's variance analysis the stream's squared norm
grows roughly linearly with depth, `σ²_{x_l} ≈ Θ(l)`, while the embedding term stays a fixed O(1)
contribution. So the embedding's *share* of the stream variance at layer `l` is roughly `1/l`: at layer 6
the token identity is about a sixth of what the stream carries, and by layer 24 it is `≈ 1/24 ≈ 4%`. The
self-attention mixing and the MLP writes have piled on top of it until the raw token signal is a few
percent of the accumulated total — the over-smoothing deep attention stacks are known for. ProRes did
nothing about this; it scheduled *when* the branches write, not *what else* the stream could carry. But
there is a cheap, direct fix sitting in exactly the surface I am allowed to edit: keep a copy of `x_1` —
call it `x0` — and re-inject a scaled copy of it at every layer. Even a modest injection gain restores
token identity from `~4%` back toward an O(1) share at the deep end, and it gives every depth a *direct
route back to token identity* that does not have to survive the dilution of the accumulator: a forward
path that keeps the raw token signal available no matter how deep the stack mixes, and — this is the part
weight tying makes free — a gradient highway straight to the embedding.

Let me make the gradient claim exact, because it is load-bearing and the substrate hands me a bonus for
it. Write the layer-`i` update as `x_new = resid_lambda[i]·x + delta_i + x0_lambda[i]·x0`, where
`delta_i = block_out − x` is the block's own Pre-LN branch, recovered the same way ProRes recovered it.
Hold the running stream fixed and differentiate the explicit `x0` term: `∂x_new/∂x0 = x0_lambda[i]·I` at
every layer. So the loss reaches the embedding through `L` *direct*, additive routes — one per layer, each
weighted by that layer's `x0_lambda` — on top of the single bottom-of-the-chain route vanilla gives it.
And the embedding is not just any tensor: the substrate ties `wte` to `lm_head`, so `x0` shares its
weights with the output projection. A stronger, more direct gradient into the embedding is therefore
*also* a stronger gradient into the tied output head — the same parameters that turn the final
representation into logits. That is a concrete reason to expect the `x0` route to help knowledge-recall
behavior (ARC-Easy) and sharp next-token prediction (WikiText-2), not just to act as a generic skip. Now
read the two scalars as the different axes they are. `resid_lambda` controls the *carry* — how much of the
accumulated past survives into this layer: at 1 it is the plain residual, below 1 it lets a layer *forget*
some of the noisy stream (a learned leak that can counteract the variance climb the floor diagnosed),
above 1 it amplifies. `x0_lambda` controls the *injection* — how much fresh token identity enters. And the
init makes the whole thing reduce to vanilla at step zero: `resid_lambda = 1.0`, `x0_lambda = 0.0` give
`x_new = 1·x + delta + 0·x0 = x + delta`, which is bit-for-bit the vanilla residual. So at initialization
this is exactly the floor, and every deviation the network learns is a deliberate, gradient-driven
refinement — no identity-start tax, no schedule to expire, just two scalars per layer the network is free
to move.

`resid_lambda` sits right on top of the invariant I carried up from the floor: keep the leading `I`, keep
the skip coefficient near one, or the multiplicative `r^L` decay comes back. `resid_lambda` *is* the skip
coefficient — it multiplies the carried stream — so if the network learns it too far below 1 across a run
of deep layers, the identity highway is attenuated exactly the way the floor warned against. A number: if
a stretch of `k` deep layers all settled at `resid_lambda = 0.95`, the carried stream is multiplied by
`0.95^k`, and for `k = 10` that is `≈ 0.60`, a 40% attenuation of the gradient route through the deep half
— the very cliff the residual connection was invented to kill. So the knob is double-edged. Three things
hold it near 1, none a hard clamp. The init is 1, so the layer starts *on* the invariant and only leaves
it if gradient descent pushes it off. There is no weight decay on these scalars, so nothing systematically
drags it toward 0. And the loss is self-limiting: a `resid_lambda` low enough to starve the shallow
layers' gradient stalls their learning and *raises* the loss, so descent is pushed back toward keeping the
carry alive. The protection is the loss, not a constraint — which is the right kind, because it lets a
layer leak the carry *where the data says it helps* while refusing the wholesale attenuation that would
kill the highway. I expect learned `resid_lambda` near 1 with small signed deviations; if they collapsed
low, val_loss would move *up*, and I would read that as the invariant violated.

Why `x0` and not "blend in some earlier layer's output"? Because `x0` is the *least redundant* signal in
the stream. Every later representation `x_i` for `i > 1` is already reachable through the ordinary
additive residual — it is sitting in the running sum, retrievable by the very accumulation I am keeping.
The embedding is the one thing the additive stream is actively *losing* as depth grows, the `~1/l` share I
just computed, the one signal whose direct re-injection adds information the accumulator cannot recover on
its own. Re-injecting a generic hidden state would mostly duplicate what the residual already carries;
re-injecting the embedding restores the specific thing that gets diluted. And it is the cheapest such
route — `x0` is a `(B, T, D)` tensor already in hand at the bottom of the forward, broadcast-added at each
layer, no projection, no attention, no new shapes. I should also check the two scalars are not secretly
one knob. A single ReZero-style `λ` can only scale the branch; it can neither leak the carry *nor* inject
the embedding. My two scalars let a deep layer simultaneously attenuate the noisy accumulated stream
(`resid_lambda < 1`) *and* pull in clean token identity (`x0_lambda > 0`) to re-anchor itself — a
two-dimensional move a single coefficient cannot express, exactly the over-smoothing repair the additive
stream forbids. So the two-scalar form is strictly more expressive than the step-2 schedule and than a
one-scalar learnable design, and the whole thing costs `2·n_layer = 48` scalars against 355M
parameters — a few parts in ten million, free.

There is one detail in *training* these 48 scalars I have to get right, and it is the only place this step
touches the optimizer. They are gains, not weights — each sits at a leveraged point, multiplying a whole
layer's worth of signal — and they are one-dimensional, so they must not get weight decay. Decay would
pull `resid_lambda` toward 0 and `x0_lambda` toward 0, dragging the carry below 1 and fighting the very
init values I chose; a single mis-decayed gain on the residual carry could quietly leak the whole stream,
which is precisely the invariant-breaking regime I just talked myself out of. The scaffold's default
routing already sends `dim < 2` parameters to a no-decay group, which would catch these — but the cost of
getting it wrong is diffuse and hard to see in a scalar loss, so I want to be explicit and certain. In
`configure_optimizers` I pull the two scalar tensors out by `id()` into their own dedicated no-decay
group, leave the `dim ≥ 2` matrices in the decayed group and the other `dim < 2` parameters (LayerNorm
gains and the like) in the standard no-decay group, and keep everything at the base learning rate. That is
the entire optimizer change; the LR schedule and `CONFIG_OVERRIDES` stay default. And unlike ProRes there
is no step buffer, no schedule, no `self.training` gate — these are persistent learnable scalars that
apply identically at train and eval, which is the whole point of handing the magnitude to the network
rather than to a clock.

Placed in the edit surface, the mechanics mirror ProRes with one addition. The `Block` stays vanilla — it
still returns `block_out = x + delta`, so I recover the branch as `delta = block_out − x`. In
`GPT.__init__` I add the two parameter vectors, `resid_lambdas = ones(n_layer)` and
`x0_lambdas = zeros(n_layer)`. In the `GPT.forward` block loop I snapshot `x0 = x` *after* the embedding
and position-encoding are folded in but *before* the first block, so `x0` is the true embedding-stage
representation, then for each layer compute `block_out = block(x)`, recover `delta`, and write
`x = resid_lambdas[i]·x + delta + x0_lambdas[i]·x0`. In `configure_optimizers` I route the two scalar
tensors to their own no-decay group. The distilled module and the literal scaffold fill are in the answer.

So the delta from step 2 is precise: ProRes scheduled a *single* prescribed scalar per layer that ramped
0→1 and, on this budget, stranded the deep layers partway up; learned-scaling replaces it with *two
learnable* scalars per layer — `resid_lambda` (init 1, the persistent learnable carry) and `x0_lambda`
(init 0, the embedding-injection gain) — that start at exactly vanilla and refine from there, adding a
direct embedding-to-every-depth route the schedule never had. Here is what I expect against the 2.2707
prores number and the 2.2763 floor, and where I am unsure. The mechanism predicts I should beat *both*:
the learnable carry subsumes what the schedule was doing — a layer finds its own residual weight and
*keeps* it, with no clock to run out — and the `x0` injection adds a route neither earlier step had, so I
expect validation loss at or below 2.2707, with mid-2.26s a clean win. The `x0` signature is token
identity preserved at depth, so I expect it on the metrics that reward precise next-token prediction and
recall: WikiText-2 should finally move below prores's 44.11, and ARC-Easy (knowledge recall, which the
tied embedding route feeds directly) should hold its gain or extend past 55.35. LAMBADA is the interesting
one, and here I expect the two methods to *split*. ProRes won big on LAMBADA through the deep-layer
warm-up; learned-scaling's `resid_lambda` starts at 1 rather than ramping from 0, so it does not reproduce
that specific deep-layer-conditioning schedule, and I would not be surprised if LAMBADA lands slightly
*above* prores's 67.21 even as the overall val_loss improves — a tell that the two methods help through
different channels rather than one strictly dominating. The honest risk cuts the other way: with only 48
scalars and a 13.5k-step run the network may not push the `x0_lambdas` far from 0, in which case the step
collapses toward a learnable-`resid_lambda`-only model and lands near prores. If the win is real but
small, the diagnosis for the next step writes itself: a *scalar* per layer — scheduled or learned, one or
two — is still a rank-one knob on the depth flow. The stream is still fundamentally an *unweighted sum* of
past outputs, mixed the same way for every token, and what I have never given the network is the ability
to *choose which past layer's output to read*, per token, with a full-rank, content-dependent mixing —
attention over the depth axis itself.
