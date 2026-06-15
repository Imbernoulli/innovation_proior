ProRes moved the floor, and it moved it in exactly the place the mechanism predicted. Validation loss
went 2.2763 → 2.2707, a few thousandths under vanilla — modest, as a conditioning fix should be, not a
capacity jump. WikiText-2 barely budged (44.28 → 44.11), but LAMBADA dropped from 70.09 to 67.21, the
sharpest single move so far, and LAMBADA is the long-range completion task that leans hardest on the
deep layers. Read together: the shallow-to-deep warm-up did wake the deep layers up a little — the
deep-layer-sensitive metric improved most — but the overall gain is small. ARC-Easy rose to 55.35,
HellaSwag held at 33.91, both inside the noise I expected. So ProRes confirmed the diagnosis (deep
layers were under-used, and protecting them early helps) while exposing its own ceiling: a *fixed
schedule* is too blunt an instrument. It dictates *one* trajectory — `α(l,t) = min(t/(T·l), 1)` — for
every layer, and it ends at vanilla; it never lets a layer settle on a residual weight *other* than 1,
and it gives the network no say in the matter. The warm-up was a hand-set prior; once it expires every
layer is back to the rigid unit-weight accumulator I started from. I want to keep the part that worked —
gentle, conditioned control of how much each branch writes — but hand the magnitude to the network and
let it persist past the warm-up. And I want to attack a second weakness the schedule never touched.

Take the magnitude first. The natural move is the one ProRes deliberately avoided: stop *prescribing*
the residual weight and *learn* it. Put a scalar in front of each layer's residual write and let
gradient descent set it. This is the scalar-on-the-branch idea I flagged at step 2 — ReZero, SkipInit —
where each branch gets a learnable `λ` and the block becomes `x_{l+1} = x_l + λ_l·F(LN(x_l))`. The
appeal is obvious: a layer that wants to contribute more can grow its `λ` above 1, a layer that wants to
hold back can shrink it, the network discovers the per-layer residual strengths instead of having them
dictated by a schedule that expires. The classic version inits `λ = 0` so the block starts at the exact
identity, which buys the well-conditioned start ProRes also had. But I have a measured fact that makes me
hesitate over the zero init *here*: vanilla, the all-`λ=1` model, already trains cleanly to 2.2763, and
ProRes — which is `λ` ramping from 0 to 1 — only beat it by a hair. At 24 layers and 13.5k steps the
init-time conditioning problem is evidently mild; the stack is not so deep that I need to crawl out of
an identity start. If I zero-init the residual weights I spend a good chunk of my short run growing them
back toward 1 before the network even has its vanilla capacity online, and the LAMBADA result says the
benefit of the slow start is small. So the better init for *this* regime is `λ = 1` — start at exactly
the vanilla residual that already works, and let the network *adjust* the weights from there rather than
*build them up* from zero. The learnable scalar gives me persistence and per-layer freedom; initializing
at 1 instead of 0 says "the floor is good, refine it" rather than "the floor is suspect, re-earn it."
That is the honest reading of the prores numbers: the conditioning win was real but tiny, so I should
not pay for it twice.

Now the second weakness, the one the schedule never addressed and which I think is the larger prize. Go
back to the unrolled stream, `x_l = x_1 + Σ_{i<l} F_i(LN(x_i))`. The token embedding `x_1` (here the
post-dropout, post-position-encoding embedding, the thing entering the first block) is the one
representation that carries pure *token identity* — which token is actually here, before any attention
has mixed it across positions or any MLP has transformed it. In the additive stream that embedding is
present in `x_l` only as one term in a growing sum, and as depth grows it becomes a smaller and smaller
fraction of the accumulated total — the self-attention mixing and the MLP writes pile on top of it until
the original token signal is diluted, the over-smoothing that deep attention stacks are known for. ProRes
did nothing about this; it only scheduled *when* the branches write, not *what else* the stream could
carry. But I have a cheap, direct fix available in exactly the surface I am allowed to edit: re-inject
the embedding into the stream at every layer. Keep a copy of `x_1` — call it `x0` — and at each layer add
a scaled copy of it back in. That gives every depth a *direct route back to token identity* that does
not have to survive the dilution of the accumulator: a gradient highway from the embedding to every
layer, and a forward path that keeps the raw token signal available no matter how deep the stack mixes.
This is the modded-nanogpt speedrun move, and it composes naturally with the learnable per-layer scalar
because it is just a second learnable scalar on a second source.

So the residual write I want has two learnable scalars per layer, not one. Write it as
`x_new = resid_lambda[l]·x + delta + x0_lambda[l]·x0`, where `delta = block_out − x` is the block's
own contribution (the Pre-LN branch the block already computed, recovered the same way ProRes recovered
it), `resid_lambda[l]` scales the incoming residual stream, and `x0_lambda[l]` scales the embedding
injection. Read the two knobs separately. `resid_lambda` is the persistent, learnable version of the
ProRes idea: at 1 it is the plain residual, below 1 it lets a layer *forget* some of the accumulated
stream (a learned leak that can counteract the variance climb), above 1 it can amplify. `x0_lambda` is
the new route: at 0 it is off, and as the network grows it the layer pulls token identity straight from
the embedding. Crucially, the *init* makes this reduce to vanilla at step zero: `resid_lambda = 1.0` and
`x0_lambda = 0.0` give `x_new = 1·x + delta + 0·x0 = x + delta`, which is exactly the vanilla residual.
So at initialization this is bit-for-bit the floor, and every deviation the network learns from there is
a deliberate, gradient-driven refinement — no identity-start tax, no schedule to expire, just two
scalars per layer that the network is free to move.

Why `x0` and not "blend in some earlier layer's output"? Because `x0` is the *least redundant* signal in
the stream. Every later representation `x_i` for `i > 1` is already reachable through the ordinary
additive residual — it is sitting in the running sum. The embedding is the one thing the additive stream
is actively *losing* as depth grows, the one signal whose direct re-injection adds information the
accumulator cannot recover on its own. Re-injecting a generic earlier hidden state would mostly
duplicate what the residual already carries; re-injecting the embedding restores the specific thing that
gets diluted. And it is the cheapest possible such route: one scalar per layer, no projection, no
attention, no new tensor shapes — `x0` is a `(B, T, D)` tensor I already have in hand at the bottom of
the forward, broadcast-added at each layer.

Let me make sure the two scalars are not redundant with each other or with the schedule. `resid_lambda`
controls the *carry* — how much of the past stream survives into this layer. `x0_lambda` controls the
*injection* — how much fresh token identity is added. These are different axes: I can imagine a deep
layer that wants to attenuate the noisy accumulated stream (`resid_lambda < 1`) while simultaneously
pulling in clean token identity (`x0_lambda > 0`) to re-anchor itself — exactly the over-smoothing repair
the additive stream cannot express with a single coefficient. A single scalar (ReZero) can only scale
the branch; it cannot leak the carry *or* inject the embedding. So the two-scalar form is strictly more
expressive than the step-2 schedule and than the one-scalar learnable design, and it costs almost
nothing: `2·n_layer = 48` scalars total against 355M parameters.

There is one detail in *training* these scalars I have to get right, and it is the only place this rung
touches the optimizer. These are gains, not weights — each one sits at a leveraged point, multiplying a
whole layer's worth of signal. Two consequences. They are one-dimensional parameters, so they must not
get weight decay (decay would pull `resid_lambda` toward 0 and `x0_lambda` toward 0, fighting the very
values I initialized them at and degrading the carry). The scaffold's default optimizer routing already
sends `dim < 2` parameters to a no-decay group, which would catch them — but I want to be explicit and
*certain*, because a single mis-decayed gain on the residual carry could quietly hurt the whole stream.
So in `configure_optimizers` I pull the two scalar tensors out by identity into their own dedicated
no-decay group, leave the `dim >= 2` matrices in the decayed group and the other `dim < 2` parameters
(LayerNorm gains, etc.) in the standard no-decay group, and keep them at the base learning rate. That is
the entire optimizer change; the LR schedule and `CONFIG_OVERRIDES` stay at their defaults.

Now place it in the edit surface, because the mechanics mirror ProRes with one addition. The `Block`
stays vanilla — it still returns `block_out = x + delta`, the full Pre-LN residual, so I recover the
branch the same way: `delta = block_out − x`. In `GPT.__init__` I add the two parameter vectors,
`resid_lambdas = ones(n_layer)` and `x0_lambdas = zeros(n_layer)`. In the `GPT.forward` block loop I
first snapshot `x0 = x` *after* the embedding and position-encoding are in but *before* the first block
(so `x0` is the true embedding-stage representation), then for each layer compute `block_out = block(x)`,
recover `delta`, and write `x = resid_lambdas[i]·x + delta + x0_lambdas[i]·x0`. No step buffer, no
schedule, no `self.training` gate — these are persistent learnable scalars that apply at train and eval
identically, unlike the warm-up which only ran during training. In `configure_optimizers` I route the
two scalar tensors to their own no-decay group. The distilled module and the literal scaffold fill are
in the answer.

So the delta from step 2 is precise: ProRes scheduled a *single* prescribed scalar per layer that ramped
0→1 and expired into vanilla; learned-scaling replaces it with *two learnable* scalars per layer —
`resid_lambda` (init 1, the persistent learnable carry) and `x0_lambda` (init 0, the embedding-injection
gain) — that start at exactly vanilla and refine from there, adding a direct embedding-to-every-depth
route the schedule never had. Here is what I expect against the 2.2707 prores number and the 2.2763
floor, and where I am unsure. The mechanism predicts I should beat *both*: the learnable carry subsumes
what the schedule was doing (a layer can find its own residual weight and keep it), and the x0 injection
adds a route neither earlier rung had, so I expect validation loss at or below 2.2707 — somewhere in the
mid-2.26s would be a clean win. The x0-injection's signature is *token identity preserved at depth*, so
I expect it to help the metrics that reward precise next-token prediction: WikiText-2 perplexity should
finally move below prores's 44.11, and ARC-Easy (knowledge recall, which leans on token identity) should
hold its gain or extend it past 55.35. LAMBADA is the interesting one — ProRes already won big there via
the deep-layer warm-up; learned-scaling's gentler `resid_lambda` (init at 1, not ramping from 0) might
*not* match that specific deep-layer-conditioning win, so I would not be shocked if LAMBADA lands
slightly *above* prores's 67.21 even as the overall val_loss improves — a tell that the two methods help
through different channels. The honest risk: with only 48 scalars and a 13.5k-step run, the network may
not have enough signal to learn the `x0_lambdas` away from 0, in which case the rung collapses toward a
learnable-`resid_lambda`-only model and lands near prores. If the win is real but small, the diagnosis
for the next rung writes itself: a *scalar* per layer — whether scheduled or learned, whether one or two
— is still a rank-one knob on the depth flow; the stream is still fundamentally an *unweighted sum* of
past outputs, and what I have never given the network is the ability to *choose which past layer's output
to read*, per token, with a full-rank, content-dependent mixing — attention over the depth axis itself.
