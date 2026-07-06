STE QAT did exactly what I predicted, and its numbers draw the next problem with a sharp edge. At INT4
it reached 11.70 PPL (degradation −1.50), beating the control's 12.61 by only 0.91 and sitting
comfortably below full precision — the 16-code grid leaves little for grid-aware training to add. At INT3
the first real separation appeared: STE dropped the control's 31.41 all the way to 13.75 (degradation
0.54), within a hair of full precision. Let me measure that separation in the honest unit, because it is
the pure QAT signal I built the control to isolate. The `d1 − d2` gap at INT3 is `18.21 − 0.54 = 17.66`
PPL, which in bits per token is `1.25 − 0.06 = 1.19` bits — putting the grid in the forward recovered
more than a full bit per token that the grid-blind finetune left on the table. At INT2 the same
subtraction is even more dramatic: STE brought INT2 from the control's 94034 (and `no_qat`'s 104422) down
to 72.55, and in bits that is `12.80 − 2.46 = 10.34` bits per token recovered — straight-through hauled
the model from *worse than uniform* back to something genuinely alive. So straight-through works, and it
works hardest exactly where the floor was highest.

But INT2 is where the construction shows its ceiling, and the residual is the number that sets this
rung's target. STE's 72.55 is a degradation of 59.35 PPL, or `ln 72.55 − ln 13.20 = 4.28 − 2.58 = 1.70`
nats, about `2.46` bits per token still lost. The model is *alive* at two bits but nowhere near usable.
And I know precisely why, because I left the reason in place deliberately when I built STE. STE trains the
weights against a grid whose step size is *fixed* at `S = max|w_g| / qmax` per group, recomputed each
forward from the current weights but never itself a parameter. At four codes, `qmax = 1`, so `S =
max|w_g|` and the in-range reconstruction points are spaced a whole group-max apart. Straight-through can
move the *weights* to fit those levels, but it cannot move the *levels* to fit the weights. The single
fixed knob places the grid crudely, and at four codes "crudely" costs `2.46` bits per token. The leftover
INT2 error is not in the weights anymore — STE already optimized those against the grid — it is in the
*grid placement* itself, which STE holds fixed.

So what is actually on the table to reduce that `2.46`-bit gap, given everything the format nails down? Let
me walk the options rather than jump. I could abandon the uniform grid for a per-group codebook that puts
levels wherever the weight histogram wants them — strictly more expressive — but that breaks the affine
map and with it the integer matmul, the whole reason the format exists, so it is off the table. I could
spend more bits where the loss is sensitive, mixed precision across layers — but the bit-width is fixed
uniformly at INT4/INT3/INT2 by the evaluation, so that lever does not exist here. I could train longer or
harder — but the schedule (500 steps, `lr=2e-5`, cosine) is held constant across every rung precisely so
comparisons stay clean, and in any case more steps only reorganize weights against the *same* fixed grid,
which is the thing that is stuck. The one lever the format leaves open is the step size itself: keep the
uniform affine grid, keep the integer matmul, but stop treating `S` as a fixed statistic and make each
per-group scale a *trained parameter*, learned jointly with the weights against the task loss, so the
optimizer places both the weights *and* the levels. That is the only move consistent with the format that
touches the thing STE cannot.

Why should learning the step size help when the weights are already optimized for the fixed step? Because
the max-abs scale is a *statistic*, not an optimum. It is the smallest step that covers the group without
clipping — a no-data choice — but at four codes the no-clip constraint is exactly the wrong thing to
respect. Setting `S = max|w_g|` means one or two outlier weights in the group define the spacing for all
128, and the bulk of the weights, which are much smaller, get snapped to a grid far too coarse for them.
A *smaller* step would round the bulk more finely at the cost of clipping the outliers to the rail — and
at the task loss, clipping a couple of outliers can be far cheaper than coarsening 126 ordinary weights.
The right step balances rounding error against clipping error, and that balance depends on the weight
distribution *and* on the loss, which only a trained scale can find. So I make each per-group scale `s` a
learnable parameter and let the same AdamW that trains the weights train it.

Let me trace the trade at INT2 on a concrete group so the mechanism is not a slogan. Take a group with one
outlier at `0.9` and a bulk of 127 weights around `±0.2`. STE fixes `S = max/qmax = 0.9/1 = 0.9`, so a
bulk weight `0.2` has `w/S = 0.22`, rounds to `0`, and is annihilated — every one of the 127 bulk weights
is crushed to zero, and only the outlier survives (`0.9 → 1 → 0.9`). Now let LSQ learn a clipped scale,
say `s ≈ 0.25`. The bulk weight `0.2` now has `w/s = 0.8`, rounds to `1`, and reconstructs to `0.25` — a
representation error of `0.05` instead of the full `0.2` it lost under STE. The cost is the outlier: `0.9/s
= 3.6`, clipped to `qmax = 1`, reconstructs to `0.25`, a clip error of `0.65`. So learning `s` trades one
large clip error on the single outlier for 127 small rounding errors saved across the bulk — and at the
task loss, coarsening 127 ordinary weights is far more expensive than clipping one stray large one, so the
trade pays. This is the rounding-versus-clipping balance from rung 1 made concrete: the max-abs scale sits
at the no-clip extreme, the worst corner at four codes, and the learned scale walks inward to where the
summed loss cost is minimized. STE could never take this step because `S` was a statistic it could not
touch.

Now the obstacle that made STE necessary returns, but now I have to push a gradient not only to the
weight but to the *scale*. The quantizer is `ŵ = round(clip(w/s, qmin, qmax))·s`. The round is flat
almost everywhere, so the path from `s` through `w/s` into the integer code has zero ordinary derivative
— ordinary backprop sees the final multiply by `s` but misses how `s` moves values toward or away from
bin transitions. The fix for the round is the same straight-through identity I already use. But I have to
be careful: STE applies to the *round node*, while the divide-by-`s`, the clip, and the multiply-by-`s` I
should differentiate honestly. Cutting corners here is what costs accuracy, so let me do the calculus. In
the interior, `−qmin < w/s < qmax`, the quantizer is `ŵ = round(w/s)·s`. Product rule: `∂ŵ/∂s = [∂
round(w/s)/∂s]·s + round(w/s)`. For the first term, STE treats round as identity, so `round(w/s) ≈ w/s`
for differentiation and `∂(w/s)/∂s = −w/s²`; times `s` that term is `−w/s`. The second term is
`round(w/s)`. So in the interior `∂ŵ/∂s = round(w/s) − w/s` — the negative signed residual between `w/s`
and the integer it rounds to. In the clipped regions it is simpler: if `w/s ≤ qmin` the code pins at
`qmin`, so `ŵ = qmin·s` and `∂ŵ/∂s = qmin`; symmetrically `∂ŵ/∂s = qmax` above. And the data gradient is
the plain STE: `∂ŵ/∂w = 1` inside the clip range, `0` outside (clipped weights get no weight gradient).

Let me put numbers on that interior gradient, because its shape is the whole reason to learn `s` and I
want to see it move. Take a weight with `w/s = 1.4`: it rounds to `1`, and `∂ŵ/∂s = 1 − 1.4 = −0.4` — the
scale is pushed to shrink so `w/s` slides down toward the code it already sits near. Now watch it across a
transition. Just below `w/s = 1.5` the code is `1` and the gradient is `1 − 1.5⁻ = −0.5`; just above, the
code flips to `2` and the gradient is `2 − 1.5⁺ = +0.5`. So the step-size gradient reverses sign across a
transition and reaches magnitude `±0.5` right at it — it is largest exactly where a small nudge to `s`
reassigns a weight's code — and it vanishes to `0` at a code center (`w/s = 1`, gradient `1 − 1 = 0`),
where the weight is already perfectly represented and `s` should not move on its account. That is
precisely the sensitivity a fixed scale throws away: the max-abs statistic never feels which weights are
stranded between codes, while the learned scale feels every one of them and settles where the summed
pressure balances. The task's edit surface implements this exactly as a custom `torch.autograd.Function`
whose `backward` returns the in-range-masked `grad_out` for `w` and the
`(below·qmin + above·qmax + inside·(round(w/s)−w/s))·grad_out` sum for `s` — the textbook step-size
gradient.

I should check that this gradient is not just the honest local derivative in disguise, because if it were
I would gain nothing over an ordinary autograd call. Fix `w = 1.4` and perturb `s` while staying inside
the rounding bin: at `s = 1.01`, `w/s = 1.386`, round `1`, `ŵ = 1.01`; at `s = 0.99`, `w/s = 1.414`, round
`1`, `ŵ = 0.99`. The finite difference is `(1.01 − 0.99)/0.02 = 1.0`, and indeed the true local derivative
is `round(w/s) = 1` — piecewise constant within the bin, identical whether `w/s = 1.01` or `1.49`. That is
exactly as useless as `round`'s own zero derivative: it carries no information about *which* weights are
stranded near a transition, so an optimizer riding it would feel nothing until a code actually flipped,
and then a discontinuous jump. The LSQ gradient `round(w/s) − w/s = 1 − 1.4 = −0.4` is a *different*
object: it replaces the locally-constant true term with the residual, injected through STE on the round,
in the same spirit that STE replaced `round`'s zero backward slope with one. So the step-size gradient is
itself a straight-through construction — not a finite-difference approximation of `∂ŵ/∂s` but a
deliberately smoothed surrogate that turns the invisible pressure between codes into a continuous signal.
That is why it can learn a scale the max-abs statistic could never find, and why I have to build it by
hand rather than let autograd differentiate the quantizer honestly.

There is one more thing the step-size gradient needs, and it is where I have to read this task's
implementation rather than import the generic recipe. A single scalar `s` per group is being optimized by
the same AdamW, same global learning rate, as the millions of weights. Training behaves well when, across
parameters, the ratio of update magnitude to parameter magnitude is balanced; a step size whose gradient
is far out of scale with the weights' will be over-driven and destabilize the run. The step-size gradient
sums over the elements it touches, so it scales like the count times the typical per-element gradient —
out of balance with a single weight's gradient by a factor that grows with the count and with `qmax`. The
cure is to scale the step-size gradient down by a `g_scale` that cancels the imbalance: `g_scale = 1 /
sqrt(N · qmax)`. Here is the diff from the canonical version I have to honor exactly, and it is worth
pricing. The task computes `N = w.numel()` — the *whole weight tensor's* element count — not the per-group
weight count, and applies one `g_scale` to every group's scale gradient. Take a representative
`2048×2048` projection: `N = 4.19M`, so at INT2 (`qmax = 1`) the task's `g_scale = 1/sqrt(4.19×10⁶) ≈
4.9×10⁻⁴`. A *faithful per-group* normalizer, where each scale governs only its 128 weights, would be
`1/sqrt(128) ≈ 8.8×10⁻²` — larger by `sqrt(4.19×10⁶/128) ≈ 181×`. So the task damps the scale gradient
about 181-fold harder than a per-group derivation would; it is the *per-tensor* normalizer applied to
*per-group* scales, a single coarse tensor-wide factor shared across all 16 groups of the row. It still
does the job — it keeps the scale updates in a stable band relative to the weights — but it is
conservative: the learned scales will move slowly, which trades some of their reach for stability. I land
it as the harness has it rather than refining it per group, and I note the trade so I can read the INT2
number against it later.

I should be honest that AdamW complicates this argument, because it already normalizes each parameter's
update by that parameter's own running second moment, so raw gradient magnitude is not the whole story
the way it would be under plain SGD, where `g_scale` was originally derived. Under Adam a persistently
tiny scale gradient is partly rescaled back up by the small second moment — but only partly, because the
`ε` floor in Adam's denominator and the shared, warmup-then-cosine learning rate still couple the scale's
effective step to how large its gradient is relative to the weights', especially in the first tens of
steps when the second-moment estimate is still warming up. So the `181×` damping does not fully cancel,
and its residual effect is to make the learned scales move cautiously. That is a double-edged prediction:
cautious scales are stable (they will not blow up at four codes), but they may also under-shoot the
loss-optimal clip in only 500 steps and settle somewhere merely-good. I cannot resolve which dominates
from the arithmetic alone — I will have to read it off the INT2 number, and it is exactly the kind
of "the learned grid is right in principle but the optimization of it is loose" concern I want to keep in
view.

A second place I must follow this task's harness and not the generic recipe is the eval-time
materialization. The harness's fixed post-training step calls `quantize_dequantize_weight` and overwrites
every wrapped weight with its return value — but a *max-abs* RTN there would clobber the learned LSQ
scales, snapping the weights back onto the very grid I trained away from, discarding the entire
contribution of this rung in the last step before evaluation. So in this task `quantize_dequantize_weight`
is a deliberate *no-op*: `return weight.clone()`, leaving the fp32 weights untouched. The real
quantization instead lives in the wrapper's `forward`, which branches on `self.training`: during training
it calls the differentiable LSQ fake-quant with the learnable scale; at eval it performs a genuine
no-grad `round(clip(w/s))·s` on the *learned* LSQ grid. That is how evaluation still sees a properly
quantized model even though the harness's RTN was neutralized — the LSQ grid, not the max-abs grid, is
what the weights are snapped to. It is worth being explicit that this is not a shortcut but a correctness
requirement: if I let the harness's max-abs RTN run, the reported perplexity would measure a *different*
model than the one I trained — weights placed for the learned grid, then re-snapped to the max-abs grid —
and the whole rung's contribution would be discarded in the last no-grad step before evaluation. The
no-op-plus-eval-branch is the only wiring under which the number I read is the number I trained. I also follow the task's initialization: each per-group scale starts at
`2·|W_g|.mean()/sqrt(qmax)`, clamped to `1e-8`, with shape `(out_features, n_groups, 1)` so it broadcasts
over the 128 columns of each group. Let me confirm the shapes close, because a broadcast bug here would
silently apply one group's scale to another's weights. The weight reshapes to `(out, n_groups,
group_size) = (out, 16, 128)`; the scale is `(out, 16, 1)`, so `w / s` broadcasts the trailing `1` across
the 128 columns — each group divides by its own scalar, correct. On the backward pass the per-element
scale gradient has the weight's shape `(out, 16, 128)`, and it must collapse to the parameter's shape
`(out, 16, 1)`, so it is summed over `dim = −1` (the 128-column axis) — which is exactly the "sum over the
elements the scale touches" the balance argument called for, and it lands the gradient on the same shape
as the parameter. The `g_scale` multiply is a scalar, so it does not disturb the shape. Everything closes,
and the reduction axis is the one the derivation demanded. Let me sanity-check that init against the max-abs scale STE used, since
I want the learned scale to *start* somewhere reasonable and improve, not somewhere broken. For a
zero-centered group with std `σ`, `mean|W_g| ≈ 0.8σ`, so the init is `≈ 1.6σ/sqrt(qmax)`; at INT2
(`qmax = 1`) that is `≈ 1.6σ`, against STE's max-abs `S ≈ 3.3σ`. So LSQ *starts already clipped* to about
half the max-abs cover — it begins on the "round the bulk finely, clip the outliers" side of the trade
and lets the gradient tune from there, which is the right starting bias at low bits. But the same init
formula lands differently as `qmax` grows, and that is worth noting because it predicts a small cost at
INT4. There the init is `1.6σ/sqrt(7) ≈ 0.61σ`, while STE's max-abs cover is `3.3σ/7 ≈ 0.47σ` — so at four
codes LSQ starts *coarser* than the max-abs grid, on the wrong side, and must learn its scale *down*
toward the cover to match STE. With the gradient damped 181-fold it may not travel the full distance in
500 steps, so it can settle slightly over-coarse — a mechanism for LSQ landing a touch worse than STE at
INT4 even though nothing is wrong with the idea. The init that is a helpful bias at INT2 is a mild
handicap at INT4, purely because `2·mean/sqrt(qmax)` scales differently from `max/qmax` in `qmax`. Activations stay full
precision (weight-only), and the LM head is restored to plain Linear, both unchanged from STE. The full
scaffold module is in the answer.

So the precise delta from STE: STE's `fake_quantize_weight` used a fixed max-abs scale and STE on the
round; LSQ adds a learnable per-group `lsq_scale` parameter, routes the forward through a custom autograd
function that carries the step-size gradient `round(w/s)−w/s` (clip endpoints outside the range) with the
`1/sqrt(N_elem·qmax)` normalizer, neutralizes the harness RTN to a no-op clone, and does real LSQ-grid QDQ
in the wrapper's eval branch. Everything else — the 500-step schedule, the wrap logic, the full-precision
head — is held constant, so the gap to STE is, by construction, exactly what *learning the grid* buys over
training the weights against a fixed grid.

What do I expect, falsifiably, against the numbers I have? At INT4 the grid is already fine and STE landed
at 11.70; learning the scale has almost no room to help, so I expect LSQ to land near there, maybe a touch
worse — just above STE's 11.70 rather than below it, since the learnable-scale machinery is overkill when the fixed scale is
nearly optimal at 16 codes and the free scale, driven even by its 181×-damped gradient, can jitter
slightly off the near-optimal max-abs grid and cost a little. At INT3 STE reached 13.75; I expect LSQ to
roughly match it, landing in the low-to-mid teens near full precision, perhaps slightly better as the
learned 8-code grid trims a little of the residual. INT2 is where I expect the payoff to be
unmistakable, because INT2 is exactly where the fixed scale was the bottleneck. STE's 72.55 was the
fixed-grid ceiling, a `2.46`-bit gap that is entirely grid placement; with the per-group step size now
free to clip outliers and round the bulk finely, I expect LSQ to fall *dramatically* — from STE's 72.55
into the high teens or low twenties, a degradation of a handful of points instead of 59, recovering most
of that `2.46`-bit gap. If that is what the numbers show — INT4 and INT3 unchanged from STE within noise,
INT2 cut from the seventies to the low twenties — then LSQ is the first method to satisfy the task's actual
requirement: low perplexity *uniformly* across all three bit-widths, including the extreme one, and
clearly beating the `finetune_then_ptq` control everywhere the floor bites. The remaining question it
would leave open is whether the learnable scale is *stable* enough at two bits — an unbounded `s` trained
by SGD, and damped so hard that it can only crawl, can still drift toward a poor clip and get stuck — and
that is the seam a method that *bounds* the learned grid would press on next.
