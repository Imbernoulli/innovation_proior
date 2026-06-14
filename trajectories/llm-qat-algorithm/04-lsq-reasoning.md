STE QAT did exactly what I predicted, and its numbers draw the next problem with a sharp edge. At INT4
it reached 11.70 PPL (degradation −1.50), essentially matching the control's 12.61 and sitting
comfortably below full precision — the 16-code grid leaves little for grid-aware training to add. At
INT3 the first real separation appeared: STE dropped the control's 31.41 all the way to 13.75
(degradation 0.54), within a hair of full precision. That gap — 31.41 for a grid-blind finetune versus
13.75 for the same finetune trained *through* the grid — is the pure QAT signal I built the control to
isolate, and it is large. So straight-through works: putting the rounding in the forward lets the
optimizer place weights that survive it. But INT2 is where the construction shows its ceiling. STE
brought INT2 from the control's 94034 (and `no_qat`'s 104422) down to 72.55 — collapse averted, five
digits to two, a genuine rescue — but 72.55 PPL against a 13.20 baseline is a degradation of 59.35. The
model is *alive* at two bits but nowhere near usable. And I know precisely why, because I left the
reason in place deliberately when I built STE.

STE trains the weights against a grid whose step size is *fixed* at `S = max|w_g| / qmax` per group,
recomputed each forward from the current weights but never itself a parameter. At four codes,
`qmax = 1`, so `S = max|w_g|` and the in-range reconstruction points are spaced a whole group-max apart.
Straight-through can move the *weights* to fit those levels, but it cannot move the *levels* to fit the
weights. The single fixed knob places the grid crudely, and at four codes "crudely" costs 59 points of
perplexity. The leftover INT2 error is not in the weights anymore — STE already optimized those against
the grid — it is in the *grid placement* itself, which STE holds fixed. So the move is forced: make the
per-group step size a trained parameter, learned jointly with the weights against the task loss, so the
optimizer places both the weights *and* the levels.

Why should learning the step size help when the weights are already optimized for the fixed step? Because
the max-abs scale is a *statistic*, not an optimum. It is the smallest step that covers the group without
clipping — a no-data choice — but at four codes the no-clip constraint is exactly the wrong thing to
respect. Setting `S = max|w_g|` means one or two outlier weights in the group define the spacing for all
128, and the bulk of the weights, which are much smaller, get snapped to a grid far too coarse for them.
A *smaller* step would round the bulk more finely at the cost of clipping the outliers to the rail —
and at the task loss, clipping a couple of outliers can be far cheaper than coarsening 126 ordinary
weights. The right step balances rounding error against clipping error, and that balance depends on the
weight distribution *and* on the loss, which only a trained scale can find. So I make each per-group
scale `s` a learnable parameter and let the same AdamW that trains the weights train it.

Now the obstacle that made STE necessary returns, but now I have to push a gradient not only to the
weight but to the *scale*. The quantizer is `ŵ = round(clip(w/s, qmin, qmax))·s`. The round is flat
almost everywhere, so the path from `s` through `w/s` into the integer code has zero ordinary derivative
— ordinary backprop sees the final multiply by `s` but misses how `s` moves values toward or away from
bin transitions. The fix for the round is the same straight-through identity I already use. But I have
to be careful: STE applies to the *round node*, while the divide-by-`s`, the clip, and the multiply-by-
`s` I should differentiate honestly. Cutting corners here is what costs accuracy, so let me do the
calculus. In the interior, `−qmin < w/s < qmax`, the quantizer is `ŵ = round(w/s)·s`. Product rule:
`∂ŵ/∂s = [∂ round(w/s)/∂s]·s + round(w/s)`. For the first term, STE treats round as identity, so
`round(w/s) ≈ w/s` for differentiation and `∂(w/s)/∂s = −w/s²`; times `s` that term is `−w/s`. The
second term is `round(w/s)`. So in the interior `∂ŵ/∂s = round(w/s) − w/s` — the negative signed
residual between `w/s` and the integer it rounds to. Stare at that: the gradient to the step size is
largest exactly when `w/s` sits near a bin transition (where a small nudge to `s` flips the code, a big
jump in `ŵ`) and goes to zero when `w/s` sits on a level. That is precisely the sensitivity a fixed
scale throws away. In the clipped regions it is simpler: if `w/s ≤ qmin` the code pins at `qmin`, so
`ŵ = qmin·s` and `∂ŵ/∂s = qmin`; symmetrically `∂ŵ/∂s = qmax` above. And the data gradient is the plain
STE: `∂ŵ/∂w = 1` inside the clip range, `0` outside (clipped weights get no weight gradient). The task's
edit surface implements this exactly as a custom `torch.autograd.Function` whose `backward` returns the
in-range-masked `grad_out` for `w` and the `(below·qmin + above·qmax + inside·(round(w/s)−w/s))·grad_out`
sum for `s` — the textbook step-size gradient.

There is one more thing the step-size gradient needs, and it is where I have to read this task's
implementation rather than import the generic recipe. A single scalar `s` per group is being optimized
by the same AdamW, same global learning rate, as the millions of weights. Training behaves well when,
across parameters, the ratio of update magnitude to parameter magnitude is balanced; a step size whose
gradient is far out of scale with the weights' will be over-driven and destabilize the run. The
step-size gradient sums over the elements it touches, so it scales like the square root of that count
times the typical per-element gradient — out of balance with a single weight's gradient by a factor that
grows with the count and with `qmax`. The cure is to scale the step-size gradient down by a `g_scale`
that cancels the imbalance: `g_scale = 1 / sqrt(N · qmax)`. Here is the diff from the canonical version I
have to honor exactly: the task computes `N = w.numel()` — the *whole weight tensor's* element count —
not the per-group weight count, and applies one `g_scale` to every group's scale gradient. So this is
not the per-layer `1/sqrt(N_W·qmax)` of the generic derivation applied per group; it is a single
tensor-level normalizer `1/sqrt(N_elem·qmax)` shared across all groups of the layer. It still does the
job — it brings the scale gradients into a stable band relative to the weights — but it is a coarser,
tensor-wide normalization, and I land it as the harness has it rather than refining it per group.

A second place I must follow this task's harness and not the generic recipe: the eval-time
materialization. The harness's
fixed post-training step calls `quantize_dequantize_weight` and overwrites every wrapped weight with its
return value — but a *max-abs* RTN there would clobber the learned LSQ scales, snapping the weights back
onto the very grid I trained away from. So in this task `quantize_dequantize_weight` is a deliberate
*no-op*: `return weight.clone()`, leaving the fp32 weights untouched. The real quantization instead
lives in the wrapper's `forward`, which branches on `self.training`: during training it calls the
differentiable LSQ fake-quant with the learnable scale; at eval it performs a genuine no-grad
`round(clip(w/s))·s` on the *learned* LSQ grid. That is how evaluation still sees a properly quantized
model even though the harness's RTN was neutralized — the LSQ grid, not the max-abs grid, is what the
weights are snapped to. I also follow the task's initialization: each per-group scale starts at
`2·|W_g|.mean()/sqrt(qmax)`, clamped to `1e-8`, with shape `(out_features, n_groups, 1)` so it
broadcasts over the 128 columns of each group. Activations stay full precision (weight-only), and the
LM head is restored to plain Linear, both unchanged from STE. The full scaffold module is in the answer.

So the precise delta from STE: STE's `fake_quantize_weight` used a fixed max-abs scale and STE on the
round; LSQ adds a learnable per-group `lsq_scale` parameter, routes the forward through a custom
autograd function that carries the step-size gradient `round(w/s)−w/s` (clip endpoints outside the
range) with the `1/sqrt(N_elem·qmax)` normalizer, neutralizes the harness RTN to a no-op clone, and does
real LSQ-grid QDQ in the wrapper's eval branch. Everything else — the 500-step schedule, the wrap logic,
the full-precision head — is held constant, so the gap to STE is, by construction, exactly what *learning
the grid* buys over training the weights against a fixed grid.

What do I expect, falsifiably, against the numbers I have? At INT4 the grid is already fine and STE
landed at 11.70; learning the scale has almost no room to help, so I expect LSQ to land near there,
maybe a touch worse — possibly around 12.2, since the learnable-scale machinery is overkill when the
fixed scale is nearly optimal at 16 codes and can even cost a little. At INT3 STE reached 13.75; I
expect LSQ to roughly match it, landing in the low-to-mid teens near full precision, perhaps slightly
better (around 13.5) as the learned 8-code grid trims a little of the residual. INT2 is where I expect
the payoff to be unmistakable, because INT2 is exactly where the fixed scale was the bottleneck. STE's
72.55 was the fixed-grid ceiling; with the per-group step size now free to clip outliers and round the
bulk finely, I expect LSQ to fall *dramatically* — from STE's 72.55 into the high teens or low twenties,
a degradation of a handful of points instead of 59. If that is what the numbers show — INT4 and INT3
unchanged from STE within noise, INT2 cut from the seventies to the low twenties — then LSQ is the first
method to satisfy the task's actual requirement: low perplexity *uniformly* across all three
bit-widths, including the extreme one, and clearly beating the `finetune_then_ptq` control everywhere
the floor bites. The remaining question it would leave open is whether the learnable scale is *stable*
enough at two bits — an unbounded `s` trained by SGD can still drift — and that is the seam a method
that *bounds* the learned grid would press on next.
