The adaptive-norm floor came back at 20.01 / 12.31 / 11.77 FID (Small / Medium / Large), and the shape of
those numbers tells me exactly what to try next. It is a real class-conditional model — nothing is broken,
the conditioner is genuinely steering the denoiser — but it is the highest FID I expect to see on this
ladder, and the gap from Small to Large (20.01 → 11.77) shows the conditioning isn't the thing capping the
big model so much as the small one: at Small, capacity is tight and a blunt conditioner costs you most.
The reason I put the class *only* through the post-block path was to test that operator in isolation, and
the result reads like the bill for that choice. I left `prepare_conditioning` an identity, so the class
signal never touched the residual blocks' *own* adaptive group norm — the tuned, block-internal modulation
socket the timestep already uses. The class only got to nudge each feature map *after* the block had
already computed it, through a freshly-inserted, zero-init-gated conditioner that has to climb off zero
before it does anything. That is a strictly thinner channel than the one the timestep enjoys, and 12.31 at
Medium against what a denoiser of this size should reach suggests the bottleneck is *routing*, not the
affine itself. So before I reach for a richer operator, the cheapest hypothesis to falsify is: the affine
was fine; I was sending it down the wrong road.

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

Before I commit, I should ask whether I want *both* paths — the time-path sum *and* a post-block
conditioner — rather than the time path alone. Keeping the floor's `AdaLNBlock` and adding the sum would
strictly contain concat-FiLM as a special case, so it cannot be worse in principle. But it would also stop
this rung from *answering the question I am asking*. The whole point of step 2 is to isolate the effect of
*routing*: I want to know whether the floor's high FID was the post-block placement or the affine's
bandwidth, and the only way to read that off one number is to change exactly one thing — move the same
family of conditioning from the post-block path to the time path and nothing else. If I leave a post-block
adaptive norm in, a gain could come from either path and the experiment is muddied. There is also a
parameter-budget reason: the task enforces a `1.05×`-of-cross-attention budget, and the floor's per-block
`AdaLNBlock` regressors are not free; dropping them entirely makes concat-FiLM the *cheapest* fill on the
ladder, which is the right thing for the simplest operator. And there is a redundancy argument: the
post-block AdaLN and the block-internal AdaGN are the *same* per-channel affine applied a few layers apart,
so stacking both is mostly duplicated capacity rather than new bandwidth — if the time path works, the
post-block copy earns little. So the clean choice is the time path *alone*.

So the design decision flips relative to step 1. There, I made `prepare_conditioning` an identity and put
the work in `ClassConditioner`. Here I do the opposite: `prepare_conditioning` returns
$t_{\text{emb}} + c_{\text{emb}}$, sending the class into the block-internal AdaGN, and `ClassConditioner`
becomes a **no-op** that returns its input — there is no extra post-block module at all. Read against the
floor, this is almost a control experiment: I have moved the same family of conditioning (an adaptive
affine) from the post-block path to the time path and, in doing so, dropped from the full
scale/shift/gate adaptive norm down to its additive-bias corner. If the FID *improves*, the lesson is
unambiguous — the floor's bottleneck was the route, the post-block-only placement, not the bandwidth of
the affine; riding the tuned block-internal modulation beats inserting a new gated sublayer after each
block. If it *doesn't* improve, then the post-block adaptive norm was buying something the additive bias
can't, and I'd know the next gain has to come from a richer operator rather than a better route.

I should check the one thing that could break: dimensions and the existing class embedding. The substrate's
`class_embed = nn.Embedding(num_classes, time_embed_dim)` already emits a class vector at exactly
`time_embed_dim`, the same width as the time embedding handed into `prepare_conditioning`. So
$t_{\text{emb}} + c_{\text{emb}}$ is a straight elementwise sum — no projection, no concatenation widening
the MLP, no fixed split to decide. Summation keeps the dimension fixed and lets the block's *single*
existing AdaGN projection produce all the modulation from the combined vector, treating timestep and class
symmetrically as two additive contributions to one conditioning state. That symmetry is also what makes
the class signal global and per-channel in the same way the timestep is — which is the right inductive bias
for a structureless label. There is one subtlety worth naming about identity-at-init: unlike the floor's
zero-init gate, this path has no explicit "start as identity" switch — at init the class embedding is small
random and the block projection is the tuned one, so the class perturbs the modulation a little from step
zero. But the perturbation is bounded (a small additive term into an already-stable AdaGN) rather than a
fresh random sublayer dropped into the residual stream, so I don't expect the floor's "climb off zero"
cost; the class simply starts as a faint bias on the modulation and the embedding learns to make it
meaningful.

Now make it concrete in the edit surface, and note what the harness lets me skip. The canonical FiLM story
allows an arbitrary regressor producing $(\gamma, \beta)$, and the canonical AdaGN story projects a summed
$t+c$ embedding to scale/shift per block. Here I do not write that projection at all — the diffusers
`ResnetBlock2D` *already is* the regressor; its `temb` projection turns whatever vector I feed it into the
block's modulation. So my entire edit is the routing: `prepare_conditioning` returns the sum, and
`ClassConditioner` is the empty module (constructor takes `(channels, cond_dim)` to match the contract and
does nothing; `forward` returns `h`). There is no FiLM layer to instantiate, no gate, no GroupNorm of my
own — I am reusing the block's tuned modulation as the conditioning operator (the full scaffold module is
in the answer). The training objective and sampler are unchanged.

Let me close on the falsifiable expectation against the floor's numbers. I am betting that the additive-bias
corner *routed through the block-internal AdaGN* beats the full adaptive norm *confined to the post-block
path* — i.e. that the floor's 20.01 / 12.31 / 11.77 was paying for placement, not for missing bandwidth. So
I expect concat-FiLM to come in *below* the floor at all three scales, with the largest absolute gain where
the floor hurt most. The floor's Small-to-Large spread (20.01 vs 11.77) said the blunt conditioner cost the
small model the most; riding the tuned path should help Small the most in absolute FID and tighten that
spread. If instead concat-FiLM lands at or above the floor, my routing hypothesis is wrong and the
post-block adaptive norm was genuinely buying spatial/feature bandwidth — in which case the next rung must
add a *content-dependent* operator rather than just a better route, and I would already know to reach for
something that lets each spatial position read from the class differently. Either way the result discriminates
cleanly between "wrong road" and "blunt operator," which is exactly what I want the cheapest next experiment
to do before I spend parameters on a heavier mechanism.
