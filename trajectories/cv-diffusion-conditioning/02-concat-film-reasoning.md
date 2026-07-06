The adaptive-norm floor came back at 20.01 / 12.31 / 11.77 FID (Small / Medium / Large), and the shape of
those numbers tells me exactly what to try next. It is a real class-conditional model — nothing is broken,
the conditioner is genuinely steering the denoiser — but before I read it as "high," let me actually work
the three numbers, because their *spacing* is more informative than their level. From Small to Medium the
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

Let me re-derive from the structure of the backbone, because the substrate is already telling me the
answer. The diffusers UNet I am wrapping conditions on the timestep through a finished pipeline: the
integer $t$ becomes a sinusoidal embedding, the time-embedding MLP lifts it to `time_embed_dim`, and that
vector is handed to *every* residual block, where the block's `ResnetBlock2D` turns it into a per-channel
scale and shift on its GroupNorm — adaptive group norm, FiLM living inside the block. This path is tuned,
it is global and per-channel, and it reaches the blocks at the point where they actually compute features,
not after. The class label has *exactly the same shape* as the timestep — one global per-example vector,
no spatial layout, no internal structure. So the obvious move, the one the floor deliberately avoided, is
to let the class ride the very same path: embed the class to a vector of width `time_embed_dim`, **add** it
to the time embedding, and let the block-internal AdaGN carry the sum. Then the class modulates the blocks
where the timestep does, through machinery already tuned to do exactly this, with no new sublayer to climb
off zero.

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

There is a subtlety in that derivation worth tracing, because it tells me what the additive sum actually
buys beyond a pure bias. The diffusers `ResnetBlock2D` does not project the embedding with a bare linear
map; it runs the conditioning embedding through a nonlinearity (SiLU) before the projection that emits the
AdaGN scale/shift. With a nonlinear $P$, $P(t_{\text{emb}} + c_{\text{emb}})$ does *not* split cleanly into
$P\,t_{\text{emb}} + P\,c_{\text{emb}}$ — the sum is taken *before* the nonlinearity, so the class can
*interact* with the timestep in determining the modulation rather than only adding a fixed offset to it.
That is actually a feature, not a bug: the right scale/shift for "class horse at high noise" need not be the
sum of the per-factor modulations, and folding the class into $\text{emb}$ before the block's own SiLU lets
the modulation depend on the *joint* state $(t, c)$. So "additive" describes how the class enters the
*embedding* ($t_{\text{emb}} + c_{\text{emb}}$, the concat corner), while the block's nonlinear projection
turns that summed embedding into a $(t, c)$-dependent affine. The class is therefore not strictly a
$\gamma=1$ bias on the *features* — it is a $\gamma=1$ additive contribution to the *embedding* that the
block then expands, with interaction, into a full per-channel scale/shift. That is a strictly richer use of
the affine family than a raw additive feature bias, achieved for free by reusing the block's own regressor.

Let me put one number on that interaction so I know it is not rhetorical. SiLU is $\mathrm{SiLU}(u) =
u\,\sigma(u)$. Take a coordinate where $t_{\text{emb}}$ contributes $1$ and $c_{\text{emb}}$ contributes
$1$. If the projection were additive I would get $\mathrm{SiLU}(1) + \mathrm{SiLU}(1) = 2\cdot 1\cdot
\sigma(1) = 2(0.731) = 1.462$. But the sum enters *before* the nonlinearity, so the block actually sees
$\mathrm{SiLU}(2) = 2\,\sigma(2) = 2(0.881) = 1.762$. The gap, $1.762 - 1.462 = 0.30$, is a genuine
coupling term — about a fifth of the additive prediction — that exists *only* because timestep and class
share the same pre-nonlinearity sum. So "the class interacts with the timestep" is a measurable $O(0.3)$
effect per active coordinate, not a figure of speech; the summed routing buys real joint $(t,c)$
modulation that a separate additive-bias path could not.

Before I commit, I should walk the two other routings I could pick and make sure the elimination is real,
not reflexive. Option A is the time path alone, what I have just derived. Option B keeps the floor's
post-block `AdaLNBlock` *and* adds the sum — belt and braces. Option C is the floor itself, post-block
alone, which I have already measured. Option B is the tempting one to argue against carefully, because it
*strictly contains* concat-FiLM as a special case (set the AdaLN gate to zero and you are back to the time
path alone), so in principle it cannot be worse. But three separate accountings knock it out. First, it
would stop this rung from *answering the question I am asking*: the whole point of step 2 is to isolate the
effect of *routing*, and the only way to read that off one number is to change exactly one thing — move the
same family of conditioning from the post-block path to the time path and nothing else. If I leave a
post-block adaptive norm in, a gain could come from either path and the experiment is confounded; I would
learn that "more conditioning helps" without learning *which* conditioning. Second, the parameter budget:
the floor's per-block `AdaLNBlock` regressors cost, from my earlier count, on the order of $3\,\text{cond
\_dim}\cdot C$ each, roughly $0.9$M summed at Small; carrying them in Option B spends that against the
$1.05\times$ budget for capacity I am arguing is redundant. Third, and this is the substantive one, the
redundancy *is* real, and I can see it in the algebra. Model the two as pure affines for a moment: the
block-internal AdaGN is $A_1(x) = \gamma_1\hat x + \beta_1$ and the post-block AdaLN is $A_2(y) =
\gamma_2\hat y + \beta_2$. With nothing between them, $A_2(A_1(x))$ collapses to a *single* affine with
scale $\gamma_1\gamma_2$ and shift $\gamma_2\beta_1 + \beta_2$ — the second layer buys nothing a
re-parameterized first layer could not. The block's intervening conv and nonlinearity keep the collapse
from being exact, but the leading-order behaviour is that $A_2$ mostly re-expresses what $A_1$ can already
reach; two per-channel affines a couple of convs apart are largely absorbable into one. So Option B mostly
buys duplicated capacity at real parameter cost while muddying the read. The clean choice is the time path
*alone*: Option A. It is also, not incidentally, the *cheapest* fill on the whole ladder — which is the
right thing for the simplest operator.

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

One more structural difference between this routing and the floor's is worth naming, because it changes how
the conditioning composes across depth. The floor inserted an *independent* module after every block —
each `AdaLNBlock` had its own regressor with its own weights, so the class was re-encoded from scratch at
every insertion point. Here there is a single object: the class embedding is formed once, summed into
$t_{\text{emb}}$ once, and that one combined vector is handed to *every* block's own tuned projection. The
class is still re-read at every resolution — every block modulates from the sum — but it is read through
one shared conditioning vector rather than a stack of independent regressors. That is more parameter-frugal
(nothing per-block to learn) and it forces a consistent class representation across depth: the same
$c_{\text{emb}}$ has to serve the coarse low-resolution blocks and the fine high-resolution ones, so the
embedding learns a single class code that the tuned per-block projections then specialize. For a signal as
simple as a ten-way label, one shared code specialized per block is the right amount of structure; the
floor's per-block re-encoding was capacity spent on independence the label does not need.

Let me also make the cross-scale prediction sharper by leaning on the capacity/conditioning split I read off
the floor. The $7.70$-vs-$0.54$ spacing said Small is capacity-bound and Medium/Large are conditioning-
bound. That gives two distinguishable predictions, not one. At Small, where capacity is the wall, a better
conditioner cannot move the wall — but it *can* stop wasting the small model's scarce early training on
climbing a zero gate, so I expect a real but capacity-capped gain there. At Medium and Large, where the
conditioner itself is the wall, a genuinely better-placed conditioner should have the most room to help —
if routing is the fix, this is where I expect to see it, and a large Medium/Large improvement would be the
cleanest confirmation that the floor's near-flat $12.31\to 11.77$ was a conditioning ceiling and not a
capacity one. So the discriminating pattern is: gains at all three scales, with the *routing* signature
being a disproportionate move at Medium and Large relative to the floor's own $0.54$ there. If instead only
Small moves and Medium/Large stay pinned, the routing did not touch the true ceiling and the operator's
bandwidth is the wall after all.

Now make it concrete in the edit surface, and note what the harness lets me skip. The canonical FiLM story
allows an arbitrary regressor producing $(\gamma, \beta)$, and the canonical AdaGN story projects a summed
$t+c$ embedding to scale/shift per block. Here I do not write that projection at all — the diffusers
`ResnetBlock2D` *already is* the regressor; its `temb` projection turns whatever vector I feed it into the
block's modulation. So my entire edit is the routing: `prepare_conditioning` returns the sum, and
`ClassConditioner` is the empty module (constructor takes `(channels, cond_dim)` to match the contract and
does nothing; `forward` returns `h`). There is no FiLM layer to instantiate, no gate, no GroupNorm of my
own — I am reusing the block's tuned modulation as the conditioning operator (the full scaffold module is
in the answer). The training objective and sampler are unchanged, and the sampling behaviour is worth a
sanity glance: at every one of the 50 DDIM steps the fixed class index is embedded and summed into that
step's $t_{\text{emb}}$, so the class rides *with* the timestep down the whole reverse trajectory,
modulating every block at every step exactly as it did in training — there is no inference-time seam where
the conditioning could drift, which is the kind of consistency the shared-embedding routing gives for free.

Let me close on the falsifiable expectation against the floor's numbers. I am betting that the additive-bias
corner *routed through the block-internal AdaGN* beats the full adaptive norm *confined to the post-block
path* — i.e. that the floor's 20.01 / 12.31 / 11.77 was paying for placement, not for missing bandwidth. So
I expect concat-FiLM to come in *below* the floor at all three scales, and here I can be sharper about
*where*. The floor's spacing told me Small is capacity-limited while Medium and Large are conditioning-
limited; the largest headroom for a better *conditioner* therefore sits at Medium and Large, where capacity
is not the wall. But the floor also hurt Small the most in absolute FID ($20.01$ is far above the others),
and riding the tuned path removes the climb-off-zero cost that a small, capacity-tight model can least
afford to pay in wasted early training. So I genuinely expect a gain at *every* scale, with the largest
absolute drop at Small — where the blunt, slow-to-engage floor cost the most — and a tighter Small-to-Large
spread than the floor's $8.24$. If instead concat-FiLM lands at or above the floor, my routing hypothesis is
wrong and the post-block adaptive norm was genuinely buying spatial/feature bandwidth — in which case the
next rung must add a *content-dependent* operator rather than just a better route, and I would already know
to reach for something that lets each spatial position read from the class differently. Either way the
result discriminates cleanly between "wrong road" and "blunt operator," which is exactly what I want the
cheapest next experiment to do before I spend parameters on a heavier mechanism.
