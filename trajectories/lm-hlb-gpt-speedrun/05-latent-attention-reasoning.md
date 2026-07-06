I'm at ~136–140 seconds and I want to break a hundred. Where is the time now? Two places. One is the block
structure: every layer still does attention and MLP as *two separate sublayers*, each with its own
LayerNorm, its own input projection, its own output projection, its own residual add. That's two norms, two
in-projections, two out-projections, two residual adds, per layer — a lot of separate small ops and a lot of
memory round-trips, when the two sublayers are doing closely related things. The other is the batch
controller from earlier: it works, but it steers off a *local* signal (the per-step grad-norm jitter against
a fixed target), which is a bit blind — it doesn't know where in the run it is or how big the model is. Both
are worth attacking to get under 100s.

Take the block first, and let me be precise about what the duplication actually costs, because at this width
the per-block work isn't only matmul FLOPs — it's memory traffic through a lot of small ops, and that traffic
is often the binding cost. A LayerNorm reads the full B·L·d activation and writes it back; a residual add
reads two B·L·d tensors and writes one; each in- and out-projection is its own kernel launch shuttling
activations to and from HBM. Two sublayers means paying all of that *twice* per layer. Over six layers, the
second norm and the second residual add alone are a dozen extra full-width passes over the activation tensor
per forward, and the same again on the backward. If the two sublayers share almost all of that scaffolding,
that duplicated traffic is pure overhead I can delete. Count the big projections too: a two-sublayer block has
four of them — attention's in-projection (to queries, keys, values) and out-projection, plus the MLP's expand
and project — each a separate matmul kernel with its own activation read and write. If I can serve both
sublayers from a *single* up-projection and a *single* down-projection, that's four projection matmuls per
layer collapsing to two, halving the projection kernel launches and their round-trips on top of the norms and
adds. That's the shape of the win: not a new fast kernel, but *fewer* kernels doing the same work.

So why are attention and the MLP two separate sublayers? Historically because they do different jobs —
attention mixes across positions, the MLP mixes across channels — and the clean way to write that is two
modules. But look at what they share. Both start by normalizing the residual stream. Both project the normed
input up into some wider working space. Both end by projecting back down and adding to the residual. The
*only* genuinely different middle step is: attention does a softmax mix over positions; the MLP does a
pointwise gated nonlinearity over channels. Everything around that middle is duplicated. Before I go all the
way, there's a lighter fusion worth naming: I could share just the *norm*, running attention and the MLP in
parallel off one normalized input and summing their outputs — the parallel-block trick — which deletes one
norm per layer but keeps two up-projections, two down-projections, and a separate value projection. That's a
real but partial saving. I can go further if I do the up-projection *once*, carve the wide working space into
the pieces attention needs (queries, keys, values) *and* the pieces the MLP needs (a gated nonlinear path) at
the same time, run both middles, then project down *once*. Then I pay one norm, one in-projection, one
out-projection, one residual add per layer instead of two of each — the full fusion, not just the shared
norm.

Let me design that fused block. One shared LayerNorm. One `expand` projection from the residual width to a
wide space, which I split into four parts: a query slice, a key slice, and two slices for a gated nonlinear
path (a `linear` part and a `pre_gelu` part). The gated path computes a GeGLU — `linear ⊙ GELU(pre_gelu)` —
the same gated-linear-unit idea I already adopted for the MLP, with a GELU gate here rather than the SiLU gate
I used before; both are smooth members of the same gated family and the choice between them is second-order,
so I take the GELU gate and keep the value-times-gate structure that earned its place in the previous rung. To make the fusion tight, of
that gated nonlinear output I let *part* of the channels stay local (that's the MLP's contribution to this
position) and I use the *other* part as the *values* for attention. So the MLP's nonlinearity and the
attention's values come out of the *same* gated computation — the values attention mixes over positions are
themselves the nonlinear, gated features, not a separate linear projection. That's the projection I get to
delete that the parallel-block trick couldn't: attention no longer needs its own value matrix, because the
MLP already computed richer values than a linear value projection would have. And this is more than a
bookkeeping saving — it changes *what attention mixes*. In a standard block the values are a plain linear
projection of the normed input; here they are the nonlinear, gated GeGLU features, so attention is spreading
already-processed representations across positions rather than raw linear ones. That richer value is plausibly
why a single head suffices where a standard block wants several: the per-position features are doing more work
before they're ever mixed, so the mixing itself can be simpler. The fusion isn't just cheaper, it front-loads
the nonlinearity into the thing attention transports. Then attention mixes those
values across positions, I concatenate the local (MLP) part with the attended part, and one `project` maps
the whole thing back to the residual width:

```python
class LatentAttentionBlock(nn.Module):
    """ Fused latent-space attention + gated-MLP block. Linear keys/queries, nonlinear gated values. """
    def __init__(self, num_dim):
        super().__init__()
        self.dim        = num_dim
        self.qk_dim     = self.dim // hyp['net']['qk_dim_div']
        self.v_dim      = num_dim
        self.expand_dim = num_dim * hyp['net']['expand_factor']
        self.norm    = nn.LayerNorm(self.dim, bias=False)
        # one fused up-projection producing query, key, and the two gated-MLP slices
        self.expand  = nn.Parameter(.5 * hyp['net']['residual_depth']**-.5 * hyp['net']['expand_factor']**-1
                                    * torch.randn(2*self.qk_dim + 2*self.expand_dim, self.dim))
        # one down-projection, init-scaled by 1/num_blocks for residual-stream stability
        self.project = nn.Parameter(hyp['net']['residual_depth']**-.5 * hyp['net']['expand_factor']**-1
                                    * hyp['net']['num_blocks']**-1 * torch.randn((self.dim, self.expand_dim)))
        # one learnable per-layer position-bias slope (the same length-agnostic linear bias, per block)
        self.position_bias_mult = nn.Parameter(torch.tensor(1., device='cuda'))

    def forward(self, x):
        residual = x
        attn_mask = torch.where(causal_mask[:x.shape[1], :x.shape[1]],
                                F.softplus(self.position_bias_mult) * position_bias_base[:x.shape[1], :x.shape[1]],
                                negative_infinity_matrix_base[:x.shape[1], :x.shape[1]])
        x = self.norm(x)
        # ONE fused projection, split into query / key / gated-MLP-linear / gated-MLP-pregelu
        query, key, linear, pre_gelu = F.linear(x, self.expand).split(
            (self.qk_dim, self.qk_dim, self.expand_dim, self.expand_dim), dim=-1)
        geglu = linear * F.gelu(pre_gelu)                              # gated nonlinear features
        geglu_local, geglu_attention_value = geglu.split((self.expand_dim - self.v_dim, self.v_dim), -1)
        # the gated features ARE the attention values
        attention = F.scaled_dot_product_attention(query, key, geglu_attention_value, attn_mask=attn_mask)
        out = F.linear(torch.cat([geglu_local, attention], dim=-1), self.project)   # ONE down-projection
        return residual + out
```

Let me trace the widths so I'm sure the splits close, because a fused block is exactly where an off-by-a-slice
silently corrupts the model. `expand` produces 2·qk_dim + 2·expand_dim channels; the split peels off query
(qk_dim), key (qk_dim), and the two expand_dim slices `linear` and `pre_gelu`. The GeGLU multiplies those two
expand_dim slices elementwise, so `geglu` is expand_dim wide. I split *that* into `geglu_local` of width
expand_dim − v_dim and `geglu_attention_value` of width v_dim. Attention returns v_dim-wide output, and the
final `cat([geglu_local, attention])` is (expand_dim − v_dim) + v_dim = expand_dim wide — exactly what
`project` expects as its input, mapping expand_dim → dim. The split closes for any expand_factor, which is the
check I wanted: the local-MLP channels and the attended channels partition the gated features and reassemble
to the full working width with nothing dropped or double-counted.

Two design notes I want to get right. First, no attention heads — I run attention as a single head over the
full value width. Splitting into heads forces `.contiguous()` reshapes and per-head bookkeeping; a single
head over the fused values skips that. And the numbers say it's affordable: with `qk_dim_div = 8` the query
and key slices are dim/8 = 48 wide, while the values are the full v_dim = 384. So the score matrix is computed
from 48-dimensional queries and keys — cheap, one L×L map — and then used to mix 384-dimensional values. That
asymmetry is the whole point and it's what the docstring means by "linear keys/queries, nonlinear values":
cheap 48-dim linear projections decide *where* to attend, and the expensive 384-dim nonlinear gated features
decide *what* gets mixed. A single head at qk_dim 48 gives me full-width value mixing at an eighth of the
score-projection cost of full-width query/key, which is why one head is enough here — I put the width where it
buys representation (the values) and starve it where it only buys routing (the scores). Second, the per-layer
`position_bias_mult` carries forward the length-agnostic linear positional bias from before, one learnable
slope per block via softplus, riding inside the same additive float mask (bias where the causal mask permits,
−∞ elsewhere) so the sequence-length schedule still works unchanged.

One initialization detail the fusion forces me to redo. In the baseline I scaled residual projections by
1/√(2·num_blocks) because there were *two* residual adds per block. The fused block adds into the residual
stream *once* per layer, so there are num_blocks additions over the depth, and the `project` init carries a
1/num_blocks factor to keep the stream from growing with depth — the same depth-stability logic, re-derived
for one add per block instead of two. The `expand` gets its own 0.5 · residual_depth^(−½) · expand_factor^(−1)
scaling so the wide working space starts at a sane variance given how many channels feed the GeGLU. These are
the tight initializations that make a single fused up/down projection behave, and if the fused block is going
to be unstable it's here I'd expect it — a headless single-head attention over gated values is an unusual
enough object that I'm treating the init as load-bearing, not boilerplate.

Now the second target: the batch controller. The earlier version chased a fixed per-step grad-norm *delta* —
it looked at how much the grad norm dropped this step and nudged the batch to keep that drop near a constant.
That's a *local* target: it knows nothing about where in the run it is or how many parameters the model has,
so the same constant has to serve step 10 and step 1000 of a 30M model and a 300M model alike. I can do better
by giving it a *model-aware, run-aware* reference. Here's the reasoning: over a training run the grad norm
doesn't just wander, it *decays* — and empirically it decays roughly as a *power law* in the step count. If I
knew the expected grad-norm trajectory, I could schedule the microbatch size to *track* it: when the measured
grad-norm-per-parameter runs above the expected curve, the model is under-averaging and I grow the microbatch
count; when it runs below, I shrink it. That's a far less blind controller than "keep the per-step jitter near
a constant" — it has an explicit expectation for where the grad norm *should* be at this step, for this model
size.

So I posit `grad_norm_target = (scale · step)^pow`, a power-law decay, and I make both the exponent and the
scale depend on the model size. The exponent should be slightly more negative (faster expected decay) for
smaller models and less negative for larger ones; a clean form that does this is `pow = −0.677 ·
log(params)^(−0.2)`. Let me check that it moves the right way: at a million parameters log(params) = 13.8 and
pow = −0.677·13.8^(−.2) ≈ −0.400; at 30M, log(params) = 17.2 and pow ≈ −0.383; at a billion, log(params) =
20.7 and pow ≈ −0.369. So as the model grows the exponent creeps toward zero — a slower expected decay for
bigger models — exactly the size-dependence I wanted, and it does so gently, a 0.03 swing across three orders
of magnitude in size. There's a reassuring degenerate case in that limit: as pow → 0 the target
(scale·step)^pow flattens toward a step-independent constant, which is essentially the *old* controller's
model of the world — a fixed reference the batch tracks — so the power law doesn't do something wild for large
models, it gracefully relaxes back toward the constant-target behavior that a flatter grad-norm decay would
justify. The new controller contains the old one's regime as its zero-exponent limit, which is the sign that
I've generalized rather than replaced. The scale folds in the parameter count too, `scale = log(params) · params`, so the
curve sits at the right magnitude across model sizes. I normalize the measured grad norm per parameter
(`grad_norm / √params`) so the comparison is size-invariant, take the ratio of measured-to-expected, and push
the fractional microbatch count multiplicatively toward closing that ratio:

```python
microbatch_expected_grad_norm_pow = -.677 * math.log(total_trainable_params) ** -.2
microbatch_grad_norm_steps_scale  = math.log(total_trainable_params) * total_trainable_params

# every `sample_every` steps:
grad_norm_per_param = grad_norm / (total_trainable_params**.5)
grad_norm_target    = (microbatch_grad_norm_steps_scale * (curr_step + 1e-2)) ** microbatch_expected_grad_norm_pow
ratio_diff          = grad_norm_per_param / grad_norm_target
microbatch_steps   *= 1. + (sample_every * scale_lr * (ratio_diff - 1))   # grow if above the expected curve
microbatch_steps    = max(microbatch_steps, 1e-1)                          # floor so it can recover
base, dither_prob   = divmod(microbatch_steps, 1)
discrete_sampled_microbatch_steps = max(1, int(base + torch.bernoulli(torch.tensor(dither_prob)).item()))
```

Those constants look arbitrary, so let me check they actually put the target where the measurement lives —
because if `grad_norm_target` came out at 1e2 or 1e−10 while the measured per-param grad norm sits at ~1e−4,
the ratio would peg the controller at its floor or send it running away, and the whole scheme would be
decoration. Take a 30M-param model: `scale` = log(3e7)·3e7 = 17.2 · 3e7 ≈ 5.2e8, and `pow` ≈ −0.383. At step
100, `grad_norm_target` = (5.2e8 · 100)^(−0.383) ≈ (5.2e10)^(−0.383) ≈ 7.9e−5. The measured side: a global
grad norm of order ~0.4 normalized by √(3e7) = 5480 gives `grad_norm_per_param` ≈ 7.3e−5. Those are the *same
order* — ratio ≈ 0.93 — so at step 100 the controller sits near equilibrium and nudges gently. Running the same
check at step 30 (grad norm ~0.6 → measured 1.1e−4 vs target 1.2e−4, ratio ~0.88) and step 300 (grad norm
~0.25 → measured 4.6e−5 vs target 5.2e−5, ratio ~0.89) it stays O(1) throughout, drifting only slowly. I'm
estimating the measured grad norms rather than reading them off a real run, so I won't claim the ratio is
exactly 0.9 — but the point that survives the uncertainty is that the constants (the 0.677, the log(p)·p
scale, the /√p normalization) are tuned precisely so target and measurement land at the same few-×10⁻⁵ scale
for a model this size, which is the only regime in which the ratio-driven update does anything but saturate.

It keeps the Bernoulli dithering from before — the fractional `microbatch_steps` is the smooth accumulator,
each step uses an integer via `divmod` and a Bernoulli draw, unbiased so the average tracks the fraction — but
now the target is a principled, model-size-aware power-law trajectory for the grad norm instead of a fixed
local delta. Two smaller details close it out. I only sample the grad norm every `sample_every` steps because
computing it is the O(params) reduction pass from the earlier controller and isn't free; the update scales the
nudge by `sample_every` so adapting less often but by proportionally more keeps the effective per-step
adaptation rate constant, decoupling how often I measure from how fast the batch moves — sample every five
steps and each adjustment is five times larger, so the batch drifts at the same rate as if I'd measured every
step, for a fifth of the grad-norm passes. And the floor
`max(microbatch_steps, 1e−1)` sits below 1 deliberately: it lets the fractional accumulator dip under a single
microbatch and recover, rather than clamping at 1 and losing the ability to signal "shrink" once it's there.

It's worth noticing that this finale isn't a fresh idea so much as a synthesis of everything the ladder
already earned. The fused block still calls the flash `scaled_dot_product_attention` path from the baseline;
its positional signal is the length-agnostic linear bias from the sequence-length rung, one softplus slope per
block, so the length schedule keeps working untouched; its gated features are the gated-linear-unit idea from
the SiGLU rung, now doing double duty as attention values; and the whole thing runs in the pure-bf16 net from
that same rung. The power-law microbatch controller, likewise, is not a new mechanism but the principled
upgrade of the grad-norm batch controller from the throughput rung — same Bernoulli dithering, same
closed-loop idea, but steering off a model-and-run-aware trajectory instead of a blind local delta. The two
pieces of this rung are the last redundancies to collapse: the block's duplicated scaffolding, and the
controller's blindness to where it is in the run.

This is the closing move, so let me set the bar. The standing record going into this rung is ~136–140 seconds
on the A100. The bet is that fusing attention and the MLP into one block — one norm, one up-projection, one
down-projection, one residual add per layer, with the gated nonlinear features doing double duty as the
attention values so even the value projection disappears — cuts the per-step op count and memory traffic
substantially, while the expected-grad-norm power-law microbatch controller schedules the effective batch off
a principled trajectory instead of a blind local target, spending compute where the run actually needs it.
Together I expect that to drop the wall-clock *under 100 seconds* to ~3.8 val loss (perplexity ~44.7) on a
single 40 GB A100. The risks are real, and I can say what failure would look like: the headless single-head
attention and the tight `expand`/`project` initializations could cost stability, which would show up as the
val loss stalling above 3.8 rather than a blow-up; and the power-law controller could mis-track if the real
grad-norm decay departs from the assumed exponent, which would show up as the ratio drifting systematically to
one side and the batch either running away or pinning at the floor. The `1/num_blocks` project-init scaling and
the `max(·, 1e−1)` microbatch floor are the hedges against those two. If the fused block holds quality and the
controller tracks, this is the fastest known way to train a small GPT to ~3.8 val loss on one A100, and that
is where the ladder ends.
