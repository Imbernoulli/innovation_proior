The control settled the confound, and the numbers draw a clean line. The full-precision
finetune-then-PTQ beat blind `no_qat` everywhere the grid left room: at INT4 it reached 12.61 PPL,
degradation −0.59 — *below* the 13.2033 baseline, the in-domain finetune pulling test perplexity down
with the 16-code grid fine enough that the finetuned weights survive it. At INT3 it dropped `no_qat`'s
42.54 to 31.41 (degradation 18.21) — a real improvement, but the `max|w_g|/3` floor still leaves it badly
above full precision. And at INT2 it stayed catastrophically collapsed: 94034.51, barely distinguishable
from `no_qat`'s 104422.

Run the subtraction I built the control for. The finetune contribution at each bit-width is `d0 − d1`.
At INT4, `1.81 − (−0.59) = 2.40` PPL, in bits `0.185 − (−0.066) = 0.25` — a quarter-bit of domain
adaptation, right in the range I estimated. At INT3, `29.34 − 18.21 = 11.13` PPL, `1.69 − 1.25 = 0.44`
bits — the same domain gain, now knocking a visible chunk off a larger loss. At INT2 the raw subtraction
looks enormous, `104409 − 94021 = 10388` PPL, but in bits it is `12.95 − 12.80 = 0.15`, the *smallest* of
the three, and the finetuned INT2 model at `ln 94034 = 11.45` nats still sits above the `10.83`-nat
uniform ceiling — worse than a model that has learned nothing. So the finetune bought the same quarter-bit
of domain everywhere, decisive at INT4, useful at INT3, irrelevant at INT2. That is the whole point of
the control: a finetune *blind to the grid* cannot rescue two-bit quantization, no matter how much it
lowers the loss it rounds from, because three codes destroy the weights through a rounding the finetune
never sees. The bar is now set — any "real QAT" has to clear 31.41 at INT3 and 94034 at INT2, and the
place it must prove itself is INT2, where the control is still worse than uniform. And the asymmetry I
predicted held: the finetune's own contribution was `0.25` bits at INT4 and only `0.15` at INT2, so it
owns the easy end and leaves the hard end almost untouched — whatever a grid-aware method achieves at INT2
will be almost entirely `d1 − d2`, pure QAT signal.

The reason the control collapses points straight at the fix. Its training forward is pure fp32 —
`fake_quantize_weight` is the identity — so the optimizer never sees the grid; every gradient is the
gradient of the *full-precision* function, which has no term saying "sit where rounding with step `S` is
cheap." If I want weights that survive rounding, the rounding has to be *in the forward pass during
training*: compute the loss on the quantize-dequantized weights, so every gradient already accounts for
the grid and the weight is nudged toward configurations where the *rounded* model has low loss. That is
the one structural change between this rung and the control, and it is exactly the scaffold default I
have been carrying.

But putting the grid in the forward hits the wall that made the control take the easy way out. The
fake-quant is `ŵ = S·clamp(round(w/S), qmin, qmax)`, and `round` is a staircase — slope zero almost
everywhere, undefined at the half-integers — while `clamp` is flat outside its range. Differentiate this
honestly and the chain rule multiplies by that zero Jacobian and annihilates the gradient: nothing
reaches `w`, the weight is frozen. This is the identical wall that pushed the field off hard-threshold
units onto sigmoids decades ago. So I need a nonzero backward proxy while keeping the forward exactly the
hard quantizer, and the choice of proxy is the whole method. Two proxies I can price and reject: finite
differences want one loss evaluation per parameter, `O(N)` forward passes for `N` in the billions —
astronomically more expensive than the single backprop I am replacing, dead on arrival; and a
score-function (REINFORCE) estimator gets back to one forward pass but sends the *global* scalar loss to
every weight with no local credit assignment, so each weight learns from a signal whose variance scales
with the whole network's noise — far too noisy over 500 short steps to place millions of weights on a
coarse grid. Smoothing the quantizer in *both* passes would make the loss I minimize differ from the loss
I am scored on, back to a grid the eval never uses. The sparsity, the true integer weights, and the
honest loss all live in the forward, so the forward stays hard; the only place left to intervene is the
backward edge.

The simplest possible recipe is the one that makes rectifiers train: pretend the slope was one.
Back-propagate through the quantize-dequantize *as if it had been the identity*, so `∂L/∂w = ∂L/∂ŵ`. The
forward still computes the true rounded weight; only the backward edge lies. The one thing that can sink
this is the sign — a proxy that points the wrong way climbs the loss — so check it. The quantizer is
monotone increasing in `w`: raise `w`, you raise `round(w/S)`, you raise `ŵ` within the clip range. Make
that quantitative for one weight: condition on everything downstream and treat `ŵ` as sitting between the
two levels `ℓ⁻`, `ℓ⁺` it can round to; moving `w` up past the transition raises `ŵ` by `+S`, so the
*average* response of `ŵ` to `w` across the cell is `+1` per unit of `w/S`. The honest pointwise
derivative is a comb of delta spikes at the transitions with zeros between; the identity proxy replaces
that comb with its mean value of one. So straight-through is not an arbitrary lie — it is the *average*
slope of the staircase: right sign everywhere, unbiased-in-the-mean magnitude, zero added variance. Its
bias is that it ignores *where* in the cell each weight sits, benign for a single quantized layer, which
is exactly this regime (one weight-quant per linear).

That bias is worth staring at, because the natural "more principled" alternative gets it backwards. A
bell-shaped surrogate multiplies the incoming gradient by a factor peaking at a bin center and decaying
toward the transitions — say `exp(−frac²/2τ²)` with `τ = 0.2`, which gives slope `1.0` at a center and
`exp(−0.5·0.25/0.04) ≈ 0.04` at a transition, a twenty-five-fold attenuation. But a weight on a bin
center is already quantized perfectly — zero error, no push needed — while a weight at a transition
carries the maximal `S/2` error and a small nudge flips its code: *that* is the weight I most need to
move, and the bell throttles it to near zero. The plain identity proxy keeps slope one regardless of
position, so the on-transition weight still gets a full push. The cruder proxy is the better one, and not
by accident: it does nothing where nothing is needed (an already-quantized weight, where STE degenerates
to ordinary finetuning — exactly right) and full strength where everything is needed.

One piece of bookkeeping the identity proxy hides is load-bearing. I cannot store the weight I am
learning at low precision, because each has to absorb a long run of tiny noisy gradient steps and
averaging those needs real resolution — coarse storage rounds every infinitesimal step to zero. So I keep
a *full-precision master weight*, quantize it on the fly each forward, use the quantized value in the
matmul (straight-through carrying the gradient back to the master), and accumulate updates in the master.
The numbers make it concrete: one AdamW step moves a weight by ~`lr = 2e-5` times a normalized gradient,
far below the INT2 step `S = max|w_g|`, so stored at low precision the update would round to zero every
step. Only the fp32 master can accumulate thousands of sub-step nudges until, collectively, they carry a
weight across a code boundary. In this edit surface the master *is* `self.linear.weight`, the fp32
parameter the harness's AdamW already optimizes, and the scale is recomputed from it every forward so the
per-group grid tracks the current weight. The fixed loop builds the optimizer over all `requires_grad`
parameters and runs the finetune through the wrapper's fake-quant forward, so I manage no shadow copies;
at the end the no-grad `quantize_dequantize_weight` materializes the integer model, no straight-through
needed since nothing differentiates.

I lean on `.detach()` for the one-line form, and it is easy to detach the wrong term and silently zero
the gradient, so verify it. Write `w_dq = w + (w_q − w).detach()`. The forward value is `w + w_q − w =
w_q`, the true rounded weight — the `w` and `−w` cancel numerically, so the matmul sees genuine INT-`B`
weights. But `(w_q − w)` is detached, so the only differentiable term is the bare leading `w`, derivative
`1`; the backward gradient is exactly `∂L/∂ŵ · 1` passed straight to the master — identity, including for
entries that rounded to a different code or got clamped, so nothing is frozen out. That last point
matters: even the weight whose forward value was dragged `0.20` away still receives a full gradient and
can be moved next step. Contrast the control — a bare `round(w/S)·S` with no detach has a zero Jacobian
and passes *nothing* back, which is precisely why `finetune_then_ptq` could not place grid-aware weights.

Now what this can and cannot do, which draws the rung's ceiling. Take the INT2 group `[0.90, −0.70, 0.20,
−0.15, …]` with `S = max|w_g| = 0.90` recomputed each forward. The weight `−0.70` has `w/S = −0.78`,
rounds to `−1`, reconstructs to `−0.90` — a `0.20` error. Straight-through hands it a live gradient, so
the optimizer can move it toward `−0.90` to round cleanly, or toward `0` if the loss prefers it dropped —
both reachable because the gradient is alive. But the only two reconstruction points available to that
weight are `−0.90` and `0`, a full `0.90` apart, because `S` is pinned to the group max. If the
loss-optimal home for `−0.70` is near `−0.70` itself, there is no code there. STE can move the *weights*
to fit the levels; it cannot move the *levels* to fit the weights, because `S = max|w_g|/qmax` is a
statistic it never touches. That is the seam I am deliberately leaving in place so I can measure what the
gradient alone buys.

The precise diff, so I land the *task's* STE and not a generic version. Versus the control: I flip
`fake_quantize_weight` from the identity back to the straight-through per-group symmetric RTN — `w_dq =
w + (w_q − w).detach()` — and the wrapper's forward calls it every step. Everything else is held constant:
the same 500-step schedule, the same `quantize_dequantize_weight`, the same head restore. So the *only*
change from `finetune_then_ptq` is that the grid is in the forward — the gap is, by construction, the pure
`d1 − d2` QAT contribution. Versus the scaffold default: this rung *is* the default fake-quant, and the
contribution I isolate is exactly the *gradient*, nothing more — the scale is a fixed max-abs statistic,
recomputed each forward, *not* learned. Keeping it plain leaves the obvious next idea, learning the step
size, for the rung after. The full scaffold module is in the answer.

So what do I expect, falsifiably? At INT4 the floor is already small, so grid-aware training has little
room — I expect STE to roughly match or slightly beat 12.61, sitting near or under 13.2033. At INT3 the
first clear separation: the control's weights were placed without regard to the 8-code grid, whereas STE
trains *through* it, so the optimizer can push weights into configurations that round cleanly. Eight codes
carry enough resolution to represent a group faithfully *if* the weights are placed to round cleanly — the
worst-case residual is `S/2 = max|w_g|/6 ≈ 0.17·max|w_g|`, RMS ~`0.10·max|w_g|`, small enough that a
nudged weight contributes little error — and STE's whole job is that placement, so I expect a large drop
from 31.41 toward the low-to-mid teens, close to full precision. That is checkable: if STE stalls in the
twenties it would mean the grid, not the placement, is the INT3 bottleneck, which the small residual makes
unlikely. INT2 is the real test and where I am least sure. Straight-through lets the optimizer fight the
four-code grid, so I expect a dramatic improvement over 94034 and 104422 — collapse averted, perplexity
brought from five digits to two, the model pulled back from worse-than-uniform to alive. But the ceiling
is already in the construction: STE trains against a grid whose step is fixed at `max|w_g|` per group, one
knob placing four codes crudely, and it can only move the weights to fit those levels. So I expect INT2
rescued from catastrophe but landing well above full precision, in the tens of PPL rather than the teens,
with a visible gap to whatever a method that *learns the step size* could reach. If that is the pattern —
INT4 and INT3 near full precision, INT2 saved but stubbornly high — then the next rung is already written:
the remaining INT2 error is in the grid placement STE holds fixed, so the move is to make the per-group
step size a trained parameter.
