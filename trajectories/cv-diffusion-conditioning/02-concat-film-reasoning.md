The adaptive-norm floor came back at 20.01 / 12.31 / 11.77 FID (Small / Medium / Large), and the shape of
those numbers tells me exactly what to try next. It is a real class-conditional model — nothing is broken,
the conditioner is genuinely steering the denoiser — but before I read it as "high," the *spacing* of the
three numbers is more informative than their level. From Small to Medium the
FID falls $20.01 \to 12.31$, a drop of $7.70$; from Medium to Large it falls $12.31 \to 11.77$, a drop of
$0.54$. So of the total $8.24$ points of improvement available across the $9\text{M}\to36\text{M}\to
140\text{M}$ parameter sweep, $93\%$ is bought in the first $4\times$ of capacity and only $7\%$ in the
second $4\times$. Read that mechanically: at Small the model is *capacity*-limited — give it more
parameters and FID plunges — but by Medium it has stopped caring about capacity; quadrupling to Large
barely moves it ($12.31$ vs $11.77$). Something *other than capacity* is capping the Medium and Large
models, and since the backbone, schedule, and sampler are all fixed, the only thing left to blame is the
conditioning operator. The floor's conditioner is the bottleneck at the scales where capacity is no longer
the bottleneck. That is the signal I want to act on.

Now *which* property of the conditioner is at fault — its routing or its bandwidth? The reason I put the
class *only* through the post-block path was to test that operator in isolation, and the result reads like
the bill for that choice. I left `prepare_conditioning` an identity, so the class signal never touched the
residual blocks' *own* adaptive group norm — the tuned, block-internal modulation socket the timestep
already uses. The class only got to nudge each feature map *after* the block had already computed it,
through a freshly-inserted, zero-init-gated conditioner that, as I traced when I built it, does not even
begin to condition until its gate climbs off zero: at init only the gate row of the regressor receives
gradient, and scale and shift stay frozen until the gate is nonzero. That is a strictly thinner *and*
slower-to-engage channel than the one the timestep enjoys. So before I reach for a richer operator, the
cheapest hypothesis to falsify is: the affine was fine; I was sending it down the wrong road. If I can move
the *same* family of conditioning onto the timestep's tuned socket and the FID improves, the floor's
ceiling was placement, not bandwidth — and I will have spent almost no parameters to find that out.

The backbone structure points straight at the fix. The timestep rides a finished, tuned pipeline —
sinusoidal embedding, the time MLP to `time_embed_dim`, then every residual block's `ResnetBlock2D` turning
that vector into a per-channel scale and shift on its GroupNorm: AdaGN *inside* the block, at the point
where features are actually computed, not after. The class label has *exactly the same shape* as the
timestep — one global per-example vector, no spatial layout. So the move the floor deliberately avoided is
to let the class ride the very same path: embed it to width `time_embed_dim`, **add** it to the time
embedding, and let the block-internal AdaGN carry the sum — the class modulating the blocks where the
timestep does, through machinery already tuned to do exactly this, with no new sublayer to climb off zero.

I want to be precise that adding the two embeddings is not a hack but the correct reduction of the affine.
Concatenation-then-linear is the textbook conditioning move: stack the conditioning vector onto the
features and let the next linear map mix them. Split that linear map across the two blocks and
$W[F; z] = W_F F + W_z z$ — the ordinary feature computation plus a conditioning-dependent *additive bias*
$\beta(z) = W_z z$. So concatenation, once a linear layer follows it, *is* an additive conditional bias;
it is the $\gamma = 1$ corner of the FiLM affine $\gamma(z)\cdot F + \beta(z)$. Now look at where I am
injecting. The block consumes the conditioning embedding $\text{emb}$ and produces, via its own learned
projection, the AdaGN scale and shift $(\gamma_t, \beta_t)$ from $\text{emb}$. If that projection were
linear and I set $\text{emb} = t_{\text{emb}} + c_{\text{emb}}$, then
$(\gamma, \beta) = P(t_{\text{emb}} + c_{\text{emb}}) = P\,t_{\text{emb}} + P\,c_{\text{emb}}$: the class
contributes an additive shift to the *modulation parameters themselves*, on top of the timestep's. The
class is acting as an additive bias on the AdaGN scale/shift — exactly the concat / $\gamma = 1$ corner of
FiLM, applied at the block's own modulation point. This is why the method is named "concat-FiLM": it is the
additive-bias reduction of feature-wise linear modulation, routed through the timestep's path. It is the
*simplest* conditioning operator — the cheapest corner of the family the floor used — but routed where the
floor's affine could not reach.

There is a subtlety worth tracing, because it tells me what the additive sum buys beyond a pure bias. The
diffusers `ResnetBlock2D` does not project the embedding with a bare linear map; it runs the embedding
through a nonlinearity (SiLU) before the projection that emits the AdaGN scale/shift. With a nonlinear $P$,
$P(t_{\text{emb}} + c_{\text{emb}})$ does *not* split into $P\,t_{\text{emb}} + P\,c_{\text{emb}}$ — the sum
is taken *before* the nonlinearity, so the class can *interact* with the timestep in setting the modulation
rather than only adding a fixed offset. That is a feature: the right scale/shift for "horse at high noise"
need not be the sum of the per-factor modulations, and folding the class into $\text{emb}$ before the
block's SiLU lets the modulation depend on the *joint* state $(t, c)$. So "additive" describes how the
class enters the *embedding* (the concat corner), which the block's nonlinear projection then expands, with
interaction, into a full $(t,c)$-dependent per-channel affine — a strictly richer use of the affine family
than a raw additive feature bias, for free by reusing the block's own regressor.

I should make sure I want the time path *alone* and not both paths — keeping the floor's post-block
`AdaLNBlock` *and* adding the sum, which strictly contains concat-FiLM as a special case and so cannot be
worse in principle. Two things rule it out. It would confound the experiment: the whole point of this step
is to isolate *routing*, and that reads off one number only if I change exactly one thing; leave a
post-block adaptive norm in and a gain could come from either path. And the redundancy is real in the
algebra — model the two as pure affines, block-internal $A_1(x) = \gamma_1\hat x + \beta_1$ and post-block
$A_2(y) = \gamma_2\hat y + \beta_2$; with nothing between them $A_2(A_1(x))$ collapses to a single affine
with scale $\gamma_1\gamma_2$ and shift $\gamma_2\beta_1 + \beta_2$. The block's intervening conv and
nonlinearity keep the collapse from being exact, but to leading order $A_2$ mostly re-expresses what $A_1$
already reaches, so two per-channel affines a couple of convs apart buy duplicated capacity at real
parameter cost. So the time path alone — which is also the *cheapest* fill on the whole ladder.

There is a functional reason, beyond the confound and the redundancy, that makes me actually *expect* the
time path to win rather than merely tie. The two placements condition different things. The floor's
post-block conditioner acts on the feature map the block has *already produced* — it can only re-color the
finished output. The block-internal AdaGN, by contrast, modulates the normalized feature *inside* the
block, before the block's second convolution, so the conditioning signal shapes *what the block computes*,
not just how its output is tinted. For the timestep this matters enormously — the noise level changes which
features are worth extracting at each block, not merely their post-hoc scaling — and it is presumably why
the backbone was built to inject the timestep in-block in the first place. Routing the class the same way
lets "which class" shape each block's computation too: a horse and a ship can bias the conv toward
different intermediate features, not just recolor a shared computation afterward. That is a qualitatively
larger lever than the floor had, obtained without a single new parameter, and it is the strongest
functional reason I think the routing move is not just cleaner but genuinely stronger.

So the design decision flips relative to step 1. There, I made `prepare_conditioning` an identity and put
the work in `ClassConditioner`. Here I do the opposite: `prepare_conditioning` returns
$t_{\text{emb}} + c_{\text{emb}}$, sending the class into the block-internal AdaGN, and `ClassConditioner`
becomes a **no-op** that returns its input — there is no extra post-block module at all. Read against the
floor, this is almost a control experiment: I have moved the same family of conditioning (an adaptive
affine) from the post-block path to the time path and, in doing so, dropped from the full
scale/shift/gate adaptive norm down to its additive-bias corner.

I should check the one thing that could break: dimensions and the existing class embedding. The substrate's
`class_embed = nn.Embedding(num_classes, time_embed_dim)` already emits a class vector at exactly
`time_embed_dim`, the same width as the time embedding handed into `prepare_conditioning`. So
$t_{\text{emb}} + c_{\text{emb}}$ is a straight elementwise sum — no projection, no concatenation widening
the MLP, no fixed split to decide, and crucially *no new trainable parameters*: the sum is parameter-free,
`ClassConditioner` is empty, and `class_embed` already exists in the frozen substrate. So the entire
editable region adds zero parameters. That makes concat-FiLM not merely the cheapest fill but a *strictly
zero-overhead* one against the budget — every parameter it uses was already in the model. Summation keeps
the dimension fixed and lets the block's *single* existing AdaGN projection produce all the modulation from
the combined vector, treating timestep and class symmetrically as two additive contributions to one
conditioning state. That symmetry is also what makes the class signal global and per-channel in the same
way the timestep is — which is the right inductive bias for a structureless label.

There is one subtlety worth naming about identity-at-init, because it is the mirror image of the floor's
climb-off-zero cost and it explains *why* I expect the route to win rather than just asserting it. Unlike
the floor's zero-init gate, this path has no explicit "start as identity" switch. But trace the gradient at
step zero and it is *better*, not worse, for it. The block computes $(\gamma,\beta) = P(\mathrm{SiLU}(
t_{\text{emb}} + c_{\text{emb}}))$ where $P$ is the *tuned* projection with nonzero weights, so the class
embedding's gradient, $\partial \text{loss}/\partial c_{\text{emb}}$, flows back through $P$ and the SiLU
from the very first step — the class starts learning immediately. Contrast the floor, where at init the
regressor's scale and shift rows received *no* gradient at all (they were gated by a zero gate) and only
the gate row learned; the class conditioning there was two-phase and slow to engage. Here there is no gate
to climb: `class_embed` is small random at init, so the class perturbs the modulation a little from step
zero, and that perturbation gets useful gradient at step zero. The perturbation is also *bounded* — it is a
unit-scale additive term entering an already-stable AdaGN through a tuned projection, not a fresh random
sublayer dropped into the residual stream — so I do not expect the floor's "undo the damage" cost either.
Immediate gradient through tuned machinery, bounded init perturbation, zero new parameters: that is the
mechanistic case for why re-routing should beat the post-block placement, and it is falsifiable by the FID.

One more structural difference from the floor changes how the conditioning composes across depth. The floor
inserted an *independent* regressor after every block, so the class was re-encoded from scratch at each
insertion point. Here the class embedding is formed once, summed into $t_{\text{emb}}$ once, and that one
vector is handed to every block's own tuned projection — still re-read at every resolution, but through a
single shared code the per-block projections specialize rather than a stack of independent regressors. For
a ten-way label that is the right amount of structure; the floor's per-block re-encoding was capacity spent
on independence the label does not need.

Leaning on the capacity/conditioning split gives two distinguishable predictions, not one. At Small, where
capacity is the wall, a better conditioner cannot move it — though it *can* stop wasting the small model's
scarce early training on climbing a zero gate, so I expect a real but capacity-capped gain. At Medium and
Large, where the conditioner itself is the wall, a better-placed one has the most room to help: a
disproportionate move there relative to the floor's own $0.54$ would confirm the near-flat $12.31\to 11.77$
was a conditioning ceiling, not a capacity one. If instead only Small moves and Medium/Large stay pinned,
the routing did not touch the true ceiling and the operator's bandwidth is the wall after all.

Now make it concrete in the edit surface, and note what the substrate lets me skip. The canonical FiLM story
allows an arbitrary regressor producing $(\gamma, \beta)$, and the canonical AdaGN story projects a summed
$t+c$ embedding to scale/shift per block. Here I do not write that projection at all — the diffusers
`ResnetBlock2D` *already is* the regressor; its `temb` projection turns whatever vector I feed it into the
block's modulation. So my entire edit is the routing: `prepare_conditioning` returns the sum, and
`ClassConditioner` is the empty module (constructor takes `(channels, cond_dim)` to match the contract and
does nothing; `forward` returns `h`). There is no FiLM layer to instantiate, no gate, no GroupNorm of my
own — I am reusing the block's tuned modulation as the conditioning operator (the full module is in the
answer). The training objective and sampler are unchanged; at every DDIM step the fixed class index is
summed into that step's $t_{\text{emb}}$, so the class rides with the timestep down the whole reverse
trajectory, modulating every block exactly as in training.

So the falsifiable bet: the additive-bias corner *routed through the block-internal AdaGN* beats the full
adaptive norm *confined to the post-block path* — i.e. the floor's 20.01 / 12.31 / 11.77 was paying for
placement, not missing bandwidth. I expect concat-FiLM *below* the floor at all three scales. On *where*, I
have two calls that could disagree: the capacity/conditioning split puts the headroom for a better
conditioner at Medium and Large, but the floor also hurt Small the most in absolute FID ($20.01$), and
riding the tuned path removes the climb-off-zero cost a small model can least afford — so I lean toward the
largest absolute drop at Small and a tighter Small-to-Large spread than the floor's $8.24$. If concat-FiLM
instead lands at or above the floor, my routing hypothesis is wrong and the post-block adaptive norm was
buying spatial/feature bandwidth — pointing the next step at a *content-dependent* operator rather than a
better route. Either way the result discriminates cleanly between "wrong road" and "blunt operator."
