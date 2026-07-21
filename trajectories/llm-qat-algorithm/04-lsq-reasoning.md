STE QAT did exactly what I predicted, and its numbers draw the next problem with a sharp edge. At INT4
it reached 11.70 PPL (degradation −1.50), beating the control's 12.61 by only 0.91 and sitting below full
precision — the 16-code grid leaves little for grid-aware training to add. At INT3 the first real
separation appeared: STE dropped the control's 31.41 all the way to 13.75 (degradation 0.54), within a
hair of full precision. Measured in the honest unit, the pure QAT signal I built the control to isolate,
the `d1 − d2` gap at INT3 is `18.21 − 0.54 = 17.66` PPL, or `1.25 − 0.06 = 1.19` bits per token — putting
the grid in the forward recovered more than a full bit the grid-blind finetune left on the table. At INT2
the same subtraction is more dramatic: from the control's 94034 (and `no_qat`'s 104422) down to 72.55,
`12.80 − 2.46 = 10.34` bits recovered — straight-through hauled the model from worse than uniform back to
genuinely alive. So straight-through works, and hardest exactly where the floor was highest.

But INT2 is where the construction shows its ceiling, and the residual sets this rung's target. STE's
72.55 is `ln 72.55 − ln 13.20 = 4.28 − 2.58 = 1.70` nats, ~`2.46` bits per token still lost — alive at two
bits but nowhere near usable. And I know precisely why, because I left the reason in place when I built
STE. It trains the weights against a grid whose step size is *fixed* at `S = max|w_g| / qmax` per group,
recomputed each forward but never itself a parameter. At four codes `qmax = 1`, so `S = max|w_g|` and the
in-range reconstruction points are spaced a whole group-max apart. Straight-through can move the *weights*
to fit those levels but cannot move the *levels* to fit the weights — the single fixed knob places four
codes crudely, and at four codes "crudely" costs `2.46` bits. The leftover INT2 error is not in the
weights anymore; STE already optimized those against the grid. It is in the *grid placement*, which STE
holds fixed.

The format nails down almost everything else. A per-group codebook that puts levels wherever the
histogram wants would be strictly more expressive but breaks the affine map and the integer matmul;
mixed precision does not exist here since the bit-width is fixed uniformly; more steps only reorganize
weights against the *same* fixed grid. The one lever the format leaves open is the step size itself: keep
the uniform affine grid and the integer matmul, but stop treating `S` as a fixed statistic and make each
per-group scale a *trained parameter*, learned jointly with the weights against the task loss, so the
optimizer places both the weights *and* the levels.

Why should learning the step help when the weights are already optimized for the fixed step? Because the
max-abs scale is a *statistic*, not an optimum — the smallest step that covers the group without clipping,
a no-data choice, and at four codes the no-clip constraint is exactly the wrong thing to respect. Setting
`S = max|w_g|` means one or two outliers define the spacing for all 128, and the bulk, much smaller, gets
snapped to a grid far too coarse for it. A *smaller* step rounds the bulk more finely at the cost of
clipping the outliers to the rail — and at the task loss, clipping a couple of outliers can be far cheaper
than coarsening 126 ordinary weights. Trace the trade at INT2 on a concrete group: one outlier at `0.9`,
127 bulk weights around `±0.2`. STE fixes `S = 0.9/1 = 0.9`, so a bulk weight `0.2` has `w/S = 0.22`,
rounds to `0`, and is annihilated — all 127 bulk weights crushed to zero, only the outlier surviving. Let
LSQ learn `s ≈ 0.25`: the bulk weight `0.2` now has `w/s = 0.8`, rounds to `1`, reconstructs to `0.25` — a
`0.05` error instead of the full `0.2` it lost. The cost is the outlier: `0.9/s = 3.6`, clipped to `1`,
reconstructs to `0.25`, a `0.65` clip error. So learning `s` trades one large clip error on a single
outlier for 127 small rounding errors saved across the bulk, and at the loss that trade pays. This is the
rung-1 rounding-versus-clipping balance made concrete: the max-abs scale sits at the no-clip extreme, the
worst corner at four codes, and the learned scale walks inward to where the summed loss cost is minimized.
STE could never take this step because `S` was a statistic it could not touch.

Now the round obstacle returns, but I have to push a gradient not only to the weight but to the *scale*.
The quantizer is `ŵ = round(clip(w/s, qmin, qmax))·s`, and the round is flat almost everywhere, so
ordinary backprop sees the final multiply by `s` but misses how `s` moves values toward or away from bin
transitions. STE fixes the round; the divide-by-`s`, the clip, and the multiply-by-`s` I differentiate
honestly. In the interior, `ŵ = round(w/s)·s`, product rule: `∂ŵ/∂s = [∂round(w/s)/∂s]·s + round(w/s)`.
STE treats round as identity for the first term, so `∂(w/s)/∂s = −w/s²`, times `s` gives `−w/s`; the
second term is `round(w/s)`. So in the interior `∂ŵ/∂s = round(w/s) − w/s`, the negative signed residual
between `w/s` and the integer it rounds to. In the clipped regions it is simpler: `∂ŵ/∂s = qmin` below,
`qmax` above. The data gradient is plain STE: `∂ŵ/∂w = 1` inside the clip range, `0` outside.

The shape of that interior gradient is the whole reason to learn `s`. A weight with `w/s = 1.4` rounds to
`1`, `∂ŵ/∂s = 1 − 1.4 = −0.4` — the scale is pushed to shrink so `w/s` slides down toward the code it
sits near. Across a transition it reverses: just below `w/s = 1.5` the code is `1`, gradient `1 − 1.5 =
−0.5`; just above, the code flips to `2`, gradient `2 − 1.5 = +0.5`. So the step-size gradient reaches
`±0.5` right where a small nudge reassigns a weight's code, and vanishes to `0` at a code center (`w/s =
1`, gradient `0`) where the weight is already perfectly represented. That is precisely the sensitivity a
fixed scale throws away — the max-abs statistic never feels which weights are stranded between codes,
while the learned scale feels every one and settles where the summed pressure balances. And this is *not*
the honest local derivative in disguise: fix `w = 1.4` and perturb `s` within the bin — at `s = 1.01`,
`ŵ = 1.01`; at `s = 0.99`, `ŵ = 0.99`; the finite difference is `1.0`, the true local derivative
`round(w/s) = 1`, piecewise constant within the bin and carrying no information about which weights are
stranded near a transition. The LSQ gradient `round(w/s) − w/s = −0.4` is a different object — the true
locally-constant term replaced by the residual, injected through STE on the round, the same spirit in
which STE replaced round's zero backward slope with one. That is why it can learn a scale the max-abs
statistic could never find, and why I build it by hand rather than let autograd differentiate honestly.
The edit surface implements it as a `torch.autograd.Function` whose `backward` returns the in-range-masked
`grad_out` for `w` and the `(below·qmin + above·qmax + inside·(round(w/s)−w/s))·grad_out` sum for `s`.

One more thing the step-size gradient needs, and here I read this task's harness rather than import the
generic recipe. A single scalar `s` per group is optimized by the same AdamW, same global learning rate,
as millions of weights, and training behaves when update-to-parameter magnitude ratios are balanced. The
step-size gradient sums over the elements it touches, so it scales like count times per-element gradient —
out of balance with a single weight's by a factor growing with count and `qmax`. The cure is a `g_scale`
that cancels the imbalance, `g_scale = 1/sqrt(N · qmax)`. And here is the diff I have to honor: the task
computes `N = w.numel()` — the *whole tensor's* element count — not the per-group count, one `g_scale` per
tensor. On a representative `2048×2048` projection, `N = 4.19M`, so at INT2 the task's `g_scale =
1/sqrt(4.19e6) ≈ 4.9e−4`, whereas a faithful per-group `1/sqrt(128) ≈ 8.8e−2` would be `sqrt(4.19e6/128) ≈
181×` larger. So the task damps the scale gradient ~181-fold harder than a per-group derivation would — a
per-tensor normalizer applied to per-group scales. It still keeps scale updates in a stable band, but
conservatively: the learned scales move slowly, trading reach for stability. I land it as the harness has
it and note the trade so I can read the INT2 number against it.

AdamW complicates this, since it already normalizes each parameter's update by its own running second
moment, so raw gradient magnitude is not the whole story the way it is under the plain SGD `g_scale` was
derived for. A persistently tiny scale gradient is partly rescaled back up — but only partly, because
Adam's `ε` floor and the shared warmup-then-cosine learning rate still couple the scale's effective step
to how large its gradient is relative to the weights', especially in the first tens of steps while the
second moment warms up. So the `181×` damping does not fully cancel; its residual makes the learned scales
move cautiously. That is double-edged: cautious scales are stable (they will not blow up at four codes)
but may under-shoot the loss-optimal clip in only 500 steps and settle somewhere merely-good. I cannot
resolve which dominates from the arithmetic — I will read it off the INT2 number.

A second place I follow the harness is eval-time materialization. Its fixed post-training step calls
`quantize_dequantize_weight` and overwrites every wrapped weight with the return value — but a *max-abs*
RTN there would snap the weights back onto the grid I trained away from, discarding this rung's entire
contribution in the last no-grad step before evaluation, so the reported perplexity would measure a
*different* model than the one I trained. So `quantize_dequantize_weight` is a deliberate no-op,
`return weight.clone()`, and the real quantization lives in the wrapper's `forward`, branching on
`self.training`: during training the differentiable LSQ fake-quant with the learnable scale; at eval a
genuine no-grad `round(clip(w/s))·s` on the *learned* grid. That is the only wiring under which the number
I read is the number I trained.

I follow the task's initialization too: each per-group scale starts at `2·|W_g|.mean()/sqrt(qmax)`,
clamped to `1e-8`, shape `(out, n_groups, 1)` so it broadcasts over the 128 columns. The shapes have to
close or a broadcast bug silently applies one group's scale to another's weights: `w` reshapes to
`(out, 16, 128)`, the scale is `(out, 16, 1)`, `w/s` broadcasts the trailing `1` across 128 columns —
each group divides by its own scalar. On backward the per-element scale gradient has shape `(out, 16,
128)` and must collapse to `(out, 16, 1)`, so it sums over `dim=−1`, which is exactly the "sum over the
elements the scale touches" the balance argument called for; the `g_scale` multiply is a scalar and does
not disturb the shape. And the init sits somewhere reasonable to improve from. For a zero-centered group
with std `σ`, `mean|W_g| ≈ 0.8σ`, so the init is `≈ 1.6σ/sqrt(qmax)`; at INT2 that is `≈ 1.6σ` against
STE's max-abs `S ≈ 3.3σ` — LSQ *starts already clipped* to about half the cover, on the "round the bulk,
clip the outliers" side of the trade, the right starting bias at low bits. But the same formula lands
differently as `qmax` grows: at INT4 the init is `1.6σ/sqrt(7) ≈ 0.61σ` while STE's cover is `3.3σ/7 ≈
0.47σ`, so at four codes LSQ starts *coarser* than the max-abs grid and must learn its scale *down* toward
the cover — and with the gradient damped 181-fold it may not travel the full distance in 500 steps,
settling slightly over-coarse. The init that is a helpful bias at INT2 is a mild handicap at INT4, purely
because `2·mean/sqrt(qmax)` scales differently from `max/qmax` in `qmax`. Activations stay full precision
and the LM head is restored to plain Linear, both unchanged from STE. The full scaffold module is in the
answer.

So the precise delta from STE: LSQ adds a learnable per-group `lsq_scale`, routes the forward through the
custom autograd function carrying `round(w/s)−w/s` (clip endpoints outside the range) with the
`1/sqrt(N_elem·qmax)` normalizer, neutralizes the harness RTN to a no-op clone, and does real LSQ-grid QDQ
in the wrapper's eval branch. Everything else — the 500-step schedule, the wrap logic, the full-precision
head — is held constant, so the gap to STE is exactly what *learning the grid* buys over training the
weights against a fixed grid.

Falsifiably: at INT4 the grid is already fine and STE landed 11.70, so learning the scale has almost no
room — I expect LSQ near there, maybe a touch worse, since the free scale driven even by its 181×-damped
gradient can jitter slightly off the near-optimal max-abs grid and cost a little. At INT3 STE reached
13.75; I expect LSQ to roughly match it, low-to-mid teens near full precision, perhaps slightly better as
the learned 8-code grid trims a little residual. INT2 is where the payoff should be unmistakable, because
INT2 is exactly where the fixed scale was the bottleneck: STE's 72.55 was the fixed-grid ceiling, a
`2.46`-bit gap that is entirely grid placement, and with the per-group step now free to clip outliers and
round the bulk finely I expect LSQ to fall *dramatically*, from the seventies down into the low tens — a
degradation of a handful of points instead of 59, recovering most of that gap. If that holds — INT4 and
INT3 unchanged from STE within noise, INT2 cut from the seventies to a handful of points — then LSQ is the
first method to satisfy the task's actual requirement: low perplexity *uniformly* across all three
bit-widths, clearly beating the control everywhere the floor bites. The question it leaves open is whether
the learnable scale is *stable* enough at two bits — an unbounded `s`, damped so hard it can only crawl,
can still drift toward a poor clip and stick — and that is the seam a method that *bounds* the learned
grid would press on next.
