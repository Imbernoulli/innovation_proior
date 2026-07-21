I'm at ~136–140 seconds and I want to break a hundred. Time now sits in two places. One is the block structure:
every layer still does attention and MLP as *two separate sublayers*, each with its own LayerNorm, input
projection, output projection, and residual add — a lot of separate small ops and memory round-trips for two
operations that share almost all their structure. The other is the batch controller from earlier: it works, but
it steers off a *local* signal (per-step grad-norm jitter against a fixed target), blind to where in the run it
is or how big the model is. Both are worth attacking.

Take the block first, and be precise about what the duplication costs, because at this width the binding cost
isn't matmul FLOPs — it's memory traffic through many small ops. A LayerNorm reads the full B·L·d activation
and writes it back; a residual add reads two B·L·d tensors and writes one; each in- and out-projection is its
own kernel shuttling activations to and from HBM. Two sublayers pay all of that *twice* per layer — over six
layers, the second norm and second add alone are a dozen extra full-width passes per forward, and again on the
backward. And a two-sublayer block has four big projections — attention's in-projection (q,k,v) and
out-projection plus the MLP's expand and project — four separate matmul kernels. If both sublayers can be served
from a *single* up-projection and a *single* down-projection, that's four projection matmuls collapsing to two.
The win isn't a new fast kernel; it's *fewer* kernels doing the same work.

Why are attention and the MLP two separate sublayers? Historically because they do different jobs — attention
mixes across positions, the MLP across channels — and two modules is the clean way to write that. But look at
what they share: both normalize the residual stream, both project up into a wider working space, both project
back down and add to the residual. The *only* genuinely different middle step is a softmax mix over positions
versus a pointwise gated nonlinearity over channels. A lighter fusion would share just the *norm*, running both
in parallel off one normalized input — the parallel-block trick — deleting one norm per layer but keeping two
up-projections, two down-projections, and a separate value projection. I can go further: do the up-projection
*once*, carve the wide space into the pieces attention needs (queries, keys) *and* the pieces the MLP needs (a
gated nonlinear path) at the same time, run both middles, then project down *once* — one norm, one
in-projection, one out-projection, one residual add per layer.

Designing that fused block: one shared LayerNorm, one `expand` from the residual width to a wide space split
into four parts — a query slice, a key slice, and two slices for a gated path (`linear` and `pre_gelu`). The
gated path computes a GeGLU, `linear ⊙ GELU(pre_gelu)` — the same gated-linear-unit idea I adopted for the MLP
in the previous rung, with a GELU gate here rather than the SiLU one; both are smooth members of the same family
and the choice is second-order. To make the fusion tight, of that gated output I let *part* of the channels stay
local (the MLP's contribution to this position) and use the *other* part as the *values* for attention. So the
MLP's nonlinearity and the attention's values come out of the *same* gated computation — the values attention
mixes are themselves the nonlinear, gated features, not a separate linear projection. That's the projection the
parallel-block trick couldn't delete: attention no longer needs its own value matrix, because the MLP already
computed richer values than a linear projection would have. And it changes *what* attention mixes — in a
standard block the values are a plain linear projection; here they are already-processed gated features, so
attention spreads processed representations across positions rather than raw ones. That richer value is
plausibly why a single head suffices where a standard block wants several: the per-position features do more
work before they're ever mixed, so the mixing itself can be simpler. Then attention mixes those values, I
concatenate the local part with the attended part, and one `project` maps the whole thing back to the residual
width (`LatentAttentionBlock` in the answer; the four-way split closes for any expand_factor — the local and
attended channels partition the gated features and reassemble to the full working width with nothing dropped).

Two design notes to get right. First, no attention heads — I run attention as a single head over the full value
width, because splitting into heads forces `.contiguous()` reshapes and per-head bookkeeping. And the numbers
make it affordable: with `qk_dim_div = 8` the query and key slices are dim/8 = 48 wide while the values are the
full 384. So the score matrix is computed from 48-dim queries and keys — cheap, one L×L map — and used to mix
384-dim values. That asymmetry is the whole point of "linear keys/queries, nonlinear values": cheap 48-dim
projections decide *where* to attend, expensive 384-dim gated features decide *what* gets mixed. I put the width
where it buys representation and starve it where it only buys routing. Second, the per-layer `position_bias_mult`
carries forward the length-agnostic linear positional bias, one softplus slope per block riding inside the same
additive float mask, so the sequence-length schedule keeps working untouched.

One initialization detail the fusion forces. In the baseline I scaled residual projections by 1/√(2·num_blocks)
because there were *two* residual adds per block. The fused block adds into the stream *once* per layer, so
there are num_blocks additions over the depth and the `project` init carries a 1/num_blocks factor — the same
depth-stability logic re-derived for one add per block instead of two. The `expand` gets its own
0.5·residual_depth^(−½)·expand_factor^(−1) scaling so the wide working space starts at a sane variance. A
headless single-head attention over gated values is unusual enough that I'm treating these inits as
load-bearing; if the fused block is going to be unstable it's here I'd expect it.

Now the second target: the batch controller. The earlier version chased a fixed per-step grad-norm *delta* — a
*local* target that knows nothing about where in the run it is or how many parameters the model has, so the same
constant has to serve step 10 and step 1000 of a 30M and a 300M model alike. A better reference is
*model-aware, run-aware*. Over a run the grad norm doesn't just wander, it *decays*, roughly as a power law in
the step count. If I knew the expected trajectory I could schedule the microbatch size to *track* it: when the
size-normalized measured grad norm runs above the expected curve the model is under-averaging and I grow the
count; below it, I shrink. That's far less blind than "keep the per-step jitter near a constant."

So I posit `grad_norm_target = (scale · step)^pow`, with both exponent and scale depending on model size. The
exponent should be slightly more negative (faster expected decay) for smaller models; a clean form is `pow =
−0.677 · log(params)^(−0.2)`. It moves the right way: at 1M params pow ≈ −0.400, at 30M ≈ −0.383, at 1B ≈
−0.369 — creeping toward zero as the model grows, a gentle 0.03 swing across three orders of magnitude. And
there's a reassuring degenerate case: as pow → 0 the target flattens toward a step-independent constant, which
is essentially the *old* controller's model of the world — so the power law contains the old one's regime as its
zero-exponent limit, the sign that I've generalized rather than replaced. The scale folds in the parameter count
too, `scale = log(params) · params`, so the curve sits at the right magnitude across sizes. I normalize the
measured grad norm per parameter (`grad_norm / √params`) so the comparison is size-invariant, take the ratio of
measured-to-expected, and push the fractional microbatch count multiplicatively toward closing that ratio (the
update is in the answer).

Those constants only work if they put the target where the measurement actually lives — if `grad_norm_target`
came out at 1e2 or 1e−10 while the measured per-param grad norm sits at ~1e−4, the ratio would peg the
controller at its floor or send it running away. For a 30M-param model, scale = log(3e7)·3e7 ≈ 5.2e8 and pow ≈
−0.383, so at step 100 the target is (5.2e8·100)^(−0.383) ≈ 7.9e−5, while a global grad norm of order ~0.4
normalized by √(3e7) ≈ 5480 gives ~7.3e−5 — the *same order*, ratio ≈ 0.9, so the controller sits near
equilibrium and nudges gently. I'm estimating the measured norms rather than reading a real run, so I won't
claim the ratio is exactly 0.9, but the point that survives the uncertainty is that the constants (the 0.677,
the log(p)·p scale, the /√p normalization) are tuned so target and measurement land at the same few-×10⁻⁵ scale
for a model this size — the only regime in which the ratio-driven update does anything but saturate.

It keeps the Bernoulli dithering from before — the fractional count is the smooth accumulator, each step uses an
integer via divmod and a Bernoulli draw, unbiased so the average tracks the fraction — but the target is now a
principled, size-aware power-law trajectory instead of a fixed local delta. Two smaller details close it out. I
sample the grad norm only every `sample_every` steps because that O(params) reduction isn't free, and I scale
the nudge by `sample_every` so adapting less often but proportionally more keeps the effective per-step
adaptation rate constant — sample every five steps and each adjustment is five times larger, so the batch drifts
at the same rate for a fifth of the grad-norm passes. And the floor `max(microbatch_steps, 1e−1)` sits *below* 1
deliberately, so the fractional accumulator can dip under a single microbatch and recover rather than clamping
at 1 and losing the ability to signal "shrink."

This finale is less a fresh idea than a synthesis of what the ladder already earned: the fused block still calls
the flash `scaled_dot_product_attention` from the baseline, carries the length-agnostic linear positional bias
from the sequence-length rung so the length schedule keeps working, uses the gated-linear-unit idea from the
SiGLU rung now doing double duty as attention values, and runs in the pure-bf16 net from that same rung; the
power-law controller is the grad-norm batch controller from the throughput rung, upgraded to steer off a
model-and-run-aware trajectory. The two pieces here are the last redundancies to collapse — the block's
duplicated scaffolding and the controller's blindness to where it is in the run.

This is the closing move. The standing record is ~136–140 seconds; fusing attention and the MLP into one block
(one norm, one up-projection, one down-projection, one residual add per layer, with the gated features doing
double duty as attention values so even the value projection disappears) cuts per-step op count and memory
traffic substantially, while the power-law microbatch controller schedules the effective batch off a principled
trajectory rather than a blind local target. Together I expect that under 100 seconds to ~3.8 val loss on a
single 40GB A100. The risks are locatable: the headless single-head attention and the tight expand/project
inits could cost stability, which would show as the val loss stalling above 3.8 rather than a blow-up; and the
power-law controller could mis-track if the real grad-norm decay departs from the assumed exponent, which would
show as the ratio drifting systematically to one side and the batch running away or pinning at the floor. The
1/num_blocks project init and the `max(·, 1e−1)` floor are the hedges. If the fused block holds quality and the
controller tracks, this is the fastest known way to train a small GPT to ~3.8 val loss on one A100, and that is
where the ladder ends.
