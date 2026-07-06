The control rung settled the confound, and the numbers draw a clean line for me. The full-precision
finetune-then-PTQ beat the blind `no_qat` everywhere the grid left room: at INT4 it reached 12.61 PPL,
a degradation of −0.59 — *below* the 13.2033 full-precision baseline, exactly the in-domain finetune
pulling test perplexity down, with the 16-code grid fine enough that the finetuned weights survive it.
At INT3 it dropped `no_qat`'s 42.54 to 31.41 (degradation 18.21) — a real improvement, but the
`max|w_g|/3` floor still leaves it badly above full precision. And at INT2 it stayed catastrophically
collapsed: 94034.51, barely distinguishable from `no_qat`'s 104422 on any scale that matters.

Let me run the subtraction I set up as this control's whole purpose, because that bookkeeping is what
turns three perplexities into a diagnosis. The finetune contribution at each bit-width is
`d0 − d1`, the blind degradation minus the control's. At INT4 that is `1.81 − (−0.59) = 2.40` PPL, which
in bits per token is `0.185 − (−0.066) = 0.25` bits — a quarter-bit of domain adaptation, right in the
range I estimated. At INT3, `29.34 − 18.21 = 11.13` PPL, in bits `1.69 − 1.25 = 0.44` — again a few tenths
of a bit, the same domain gain, now knocking a visible chunk off a larger loss. At INT2 the raw
subtraction looks enormous — `104409 − 94021 = 10388` PPL — but that is the raw-perplexity scale lying to
me. In bits per token it is `12.95 − 12.80 = 0.15` bits, the *smallest* of the three, and the finetuned
INT2 model at `ln 94034 = 11.45` nats still sits above the `10.83`-nat uniform ceiling — worse than a
model that has learned nothing. So the finetune bought the same quarter-bit of domain everywhere, and
that quarter-bit is decisive at INT4, useful at INT3, and utterly irrelevant at INT2. That INT2 result is
the whole point of the control. It says: a finetune that is *blind to the grid* cannot rescue two-bit
quantization, no matter how much it lowers the loss it rounds from, because three codes destroy the
weights through a rounding the finetune never sees. So the finetune buys domain adaptation and a flatter
basin — and that is all. The bar is now set: any method I call "real QAT" has to clear
`finetune_then_ptq` at the bit-widths where the floor bites — 31.41 at INT3, 94034 at INT2 — and the
place it has to prove itself is INT2, where the control is still worse than uniform. And the asymmetry I
predicted when I built the control held: `d0 − d1`, the finetune's own contribution, was `0.25` bits at
INT4 and only `0.15` bits at INT2, so the finetune "owns" the easy end and leaves the hard end almost
untouched. That means whatever a grid-aware method achieves at INT2 will be almost entirely `d1 − d2` —
pure QAT signal, with the finetune already subtracted to nearly zero. The control has done its one job:
it turned the INT2 comparison into a clean measurement of quantization-awareness alone.

The reason the control collapses points straight at the fix. Its forward pass during training is pure
fp32 — `fake_quantize_weight` is the identity — so the optimizer never sees the grid. Every gradient it
takes is the gradient of the *full-precision* function, and full-precision loss has no term that says
"sit where rounding with step `S` is cheap." If I want the optimizer to place weights that survive
rounding, the rounding has to be *in the forward pass during training*: compute the loss on the
quantize-dequantized weights, so that every gradient already accounts for the grid. Then the weight is
nudged toward configurations where the rounded model — not the fp model — has low loss. That is the one
structural change between this rung and the control, and it is exactly the scaffold default I have been
carrying since the start.

But putting the grid in the forward immediately hits the wall that made the control take the easy way
out. The fake-quant is `ŵ = S·clamp(round(w/S), qmin, qmax)`, and `round` is a staircase — slope zero
almost everywhere, undefined at the half-integers — while `clamp` is flat outside its range. If I
differentiate this honestly, the chain rule multiplies by that zero Jacobian and annihilates the
gradient: nothing reaches `w`, and the weight is frozen. This is the identical wall that pushed the
field off hard-threshold units and onto sigmoids decades ago — a constant-by-parts graph kills gradient
learning. So the honest derivative is useless here, and I have to ask what nonzero proxy to put in its
place on the backward pass, while keeping the forward exactly the hard quantizer. Let me lay out the
options honestly rather than grab the first one, because the choice of proxy is the whole method. I could
estimate the gradient without differentiating at all — perturb the weights and measure, or treat the
rounded output as a sampled action and use a score-function (REINFORCE) estimator. Both are unbiased, and
both are unusable here for reasons I can put numbers to. Finite differences want one loss evaluation per
parameter, `O(N)` forward passes for `N` parameters — with `N` in the billions that is astronomically
more expensive than the single backprop pass I am trying to replace, dead on arrival. REINFORCE gets the
cost back to one forward pass, but it pays in variance and structure: it sends the *global* scalar loss
to every weight, correlated only with that weight's own rounding noise, with no local credit assignment
of the kind backprop hands me for free — each weight learns from a signal whose variance scales with the
whole network's noise, so over 500 short steps the estimate is far too noisy to place millions of weights
onto a coarse grid. I could smooth the quantizer — replace the hard
round with a differentiable surrogate in *both* passes — but then the forward no longer computes the true
integer weights, so the loss I minimize is not the loss I am scored on, and I would be back to a grid the
eval never uses. Or I could keep the forward exactly hard and fake only the backward Jacobian. The
sparsity, the true integer weights, the honest loss all live in the forward, so the forward has to stay
hard; the only place left to intervene is the backward edge.

The recipe that nags at me is the simplest possible one, the same one that makes rectifiers train:
pretend the slope was one. Back-propagate through the quantize-dequantize *as if it had been the
identity*. Then the gradient arriving at the quantized output, `∂L/∂ŵ`, is sent unchanged to the
underlying weight: `∂L/∂w = ∂L/∂ŵ`. The forward still computes the true rounded weight; only the
backward edge lies, pretending `∂ŵ/∂w = 1`. Before I get attached to it I have to check the one thing
that can sink it, the sign, because a proxy that points the wrong way is worse than useless — it climbs
the loss. The quantizer is monotone increasing in `w`: raise `w`, you raise `round(w/S)`, you raise `ŵ`
(within the clip range). So the true sign of "how `ŵ` responds to `w`" is positive, and the identity
proxy assigns slope `+1`, also positive. For a single quantized layer the straight-through gradient
therefore has the right sign — if downstream wants `ŵ` larger, it pushes `w` larger, exactly as it
should. It is biased — the magnitude is fabricated — but the direction is correct where it matters, and
that is what lets the weight reorganize to be quantization-friendly instead of frozen by a zero Jacobian.

I should poke at whether a less crude proxy is better, since "slope = 1" feels too cheap. The natural
alternative is to back-propagate as if the round were its smooth surrogate, multiplying the incoming
gradient by some bell-shaped factor that peaks at a bin center and decays toward the transitions. Let me
put numbers on what that does, because it sounds more principled than it is. A weight sitting near the
middle of a code cell — say `w/S = 2.0`, exactly on a level — is already quantized perfectly: it rounds to
itself, zero error, and it needs no push. A weight sitting near a transition — `w/S = 2.5`, halfway
between codes — is the one carrying the maximal `S/2` rounding error, and a small nudge to `w` flips which
code it lands on; *that* is the weight I most need to move. A bell-shaped surrogate assigns the on-center
weight a slope near its peak and the on-transition weight a slope near zero — concretely, a bump of the
form `exp(−(frac)²/2τ²)` with `τ = 0.2` gives slope `1.0` at a bin center (`frac = 0`) and
`exp(−0.5·0.25/0.04) ≈ 0.04` at a transition (`frac = 0.5`), a twenty-five-fold attenuation precisely on
the weights whose rounding error is maximal. That is backwards. The
plain identity proxy keeps slope one regardless of position in the cell, so the on-transition weight
still gets a full push to move. The cruder proxy is the better one here, and not by accident: it refuses
to attenuate where the useful signal lives.

Let me make the sign argument quantitative for one weight so I am not leaning on a verbal monotonicity
claim. Condition on everything downstream and treat the rounded weight `ŵ` as sitting between the two
levels `ℓ⁻` (below) and `ℓ⁺` (above) it can round to. Moving `w` up past the transition raises `ŵ` from
`ℓ⁻` to `ℓ⁺`, a jump of `+S`, so the *average* response of `ŵ` to `w` across the cell is `+S/S = +1` per
unit of `w/S` — genuinely positive, and on average exactly the slope the identity proxy fakes. The honest
pointwise derivative is a comb of delta spikes at the transitions with zeros between; the identity proxy
replaces that comb with its mean value of one. So straight-through is not an arbitrary lie — it is the
*average* slope of the staircase, which is the right sign everywhere and the right magnitude in the mean.
That is exactly the property a descent direction needs: correct sign, unbiased-in-the-mean magnitude,
zero added variance. The bias is that it ignores *where* in the cell each weight sits, and that bias is
benign for a single quantized layer and only compounds if I stack many — which for one weight-quant layer
per linear is the regime where it is most trustworthy.

Two limits confirm the proxy behaves sensibly at the extremes. Suppose every weight in a group happens to
sit exactly on a code, `w/S` integer for all of them: then `w_q = w`, the forward rounding error is zero,
and straight-through's identity backward passes the true fp gradient untouched — so on an already-quantized
weight, STE degenerates to ordinary finetuning, which is exactly right, since there is nothing to fight.
Now the opposite extreme, every weight pinned at a half-integer transition, `w/S ∈ {…, ±0.5, ±1.5, …}`:
the forward carries the maximal `S/2` error on every weight, and here the identity proxy hands each of
them a full-strength gradient to escape the transition — the case a bell-shaped surrogate would have
throttled to near zero. So the proxy does nothing where nothing is needed and everything where everything
is needed; the crude choice is the one that tracks the need.

There is one piece of bookkeeping the identity proxy hides, and it is load-bearing. I cannot store the
weight I am learning at low precision, because each weight has to absorb a long run of tiny, noisy
gradient steps, and averaging those out needs real resolution — coarse storage would round every
infinitesimal step to zero and nothing would move. So I keep a *full-precision master weight*, quantize
it on the fly each forward pass, use the quantized value in the matmul (with straight-through carrying
the gradient back to the master), and let the optimizer accumulate updates in the master. The numbers make
the necessity concrete: a single AdamW step here moves a weight by something on the order of the learning
rate `2e-5` times a normalized gradient — a change far below the INT2 step `S = max|w_g|`, so if I stored
the weight at low precision the update would round to zero every step and nothing would ever move. Only the
fp32 master can accumulate thousands of such sub-step nudges until, collectively, they carry a weight
across a code boundary. In this task's
edit surface that master *is* `self.linear.weight` — the fp32 parameter the harness's AdamW already
optimizes — and the scale is recomputed from it every forward, so the per-group grid tracks the current
weight. The fixed loop's design makes this automatic: it builds the optimizer over all `requires_grad`
parameters and runs the finetune through the wrapper's fake-quant forward, so I do not have to manage
shadow copies myself. At the end, the harness's no-grad `quantize_dequantize_weight` materializes the
genuine integer model, with no straight-through needed because nothing differentiates anymore.

I want to be sure the one-line form of straight-through actually does what I claim before I ship it,
because it leans on `.detach()` and it is easy to detach the wrong term and silently zero the gradient.
Write `w_dq = w + (w_q − w).detach()`. The forward *value* is `w + w_q − w = w_q`, the true rounded
weight — the `w` and `−w` cancel numerically, so the matmul sees genuine INT-`B` weights. But `(w_q − w)`
is detached, contributing no gradient, so the only differentiable term left is the bare leading `w`,
whose derivative is `1`. Hence the backward gradient is exactly `∂L/∂ŵ · 1` passed straight to the master
— identity, as intended. Trace it on the INT2 group: `w_q` reconstructs `−0.70` to `−0.90` in the
forward, so the matmul uses `−0.90`; on the backward pass an incoming `∂L/∂ŵ` for that entry arrives
unchanged at the master weight `−0.70`, including for the entries that rounded to a different code or got
clamped — nothing is frozen out. That last point is the one that matters: even the weight whose forward
value was dragged `0.20` away still receives a full gradient and can be moved next step. The contrast is
the control I already ran — a bare `round(w/S)·S` with no detach term has a zero Jacobian and passes
*nothing* back, which is precisely why `finetune_then_ptq` could not place grid-aware weights. The detach
trick is the minimal thing that manufactures a live path where the honest one is dead.

Let me trace one group at INT2 to see exactly what this can and cannot do, because it draws the ceiling
of the whole rung. Take the group from the PTQ analysis, `[0.90, −0.70, 0.20, −0.15, …]`, with the fixed
scale `S = max|w_g| = 0.90` recomputed each forward. The weight `−0.70` has `w/S = −0.78`, which rounds
to `−1` and reconstructs to `−0.90` — a rounding error of `0.20`. Straight-through hands this weight a
live gradient, so the optimizer can move it: nudge it toward `−0.90` so it rounds cleanly, or toward `0`
if the loss prefers it dropped. Both are configurations the optimizer can reach because the gradient is
alive. But look at what it *cannot* do: the only two reconstruction points available to that weight are
`−0.90` and `0`, spaced a full `0.90` apart, because `S` is pinned to the group max. If the
loss-optimal home for `−0.70` is somewhere near `−0.70` itself, there is no code there — straight-through
can march the master weight around, but the levels it must snap to are fixed by `S = max|w_g|/qmax`, a
statistic it never touches. STE can move the *weights* to fit the levels; it cannot move the *levels* to
fit the weights. That is the seam I am deliberately leaving in place this rung so I can measure what the
gradient alone buys, and it is already visible in the construction: at four codes, one fixed knob places
the grid crudely and the optimizer can only make peace with it.

Now the precise diff against the control, and against the scaffold default, so I land the *task's* STE
and not a paper's generic version. Versus the control: I flip `fake_quantize_weight` from the identity
back to the straight-through per-group symmetric RTN — `w_dq = w + (w_q − w).detach()`, which is `w_q` in
the forward and slope-one in the backward — and the wrapper's forward calls it every step:
`F.linear(x, fake_quantize_weight(self.linear.weight, ...), self.linear.bias)`. Everything else is held
constant: the same 500-step schedule (`lr=2e-5`, `warmup=50`, cosine to 10%, grad-accum 4, grad-clip
1.0), the same `quantize_dequantize_weight`, the same wrap-and-restore-the-head logic. So the *only*
change from `finetune_then_ptq` is that the grid is now in the forward — which means the gap between this
rung and the control is, by construction, the pure QAT contribution, the `d1 − d2` term, the finetune
effect subtracted out. Versus the scaffold default: this rung *is* the default fake-quant, but I should be
deliberate that the contribution I am isolating is exactly the *gradient*, nothing more. The scale here
is a fixed max-abs statistic, recomputed each forward, *not* learned. Keeping the scale plain isolates
what straight-through buys on its own — the ability to send a gradient through the round so the master
weight reorganizes to be quantization-friendly — and leaves the obvious next idea, *learning* the step
size to place the levels more cleverly, for the rung after this. The full scaffold module is in the
answer.

So what do I expect, falsifiably, against the numbers I now have? At INT4 the floor is already small, so I
expect STE to roughly match or slightly beat `finetune_then_ptq`'s 12.61 — grid-aware training has little
room to help when blind finetuning already lands below full precision; both should sit near or under
13.2033. At INT3 I expect the first clear separation: the control reached 31.41 because its weights were
placed without regard to the 8-code grid, whereas STE trains *through* that grid, so the optimizer can
push weights into configurations that round cleanly. INT3 was losing `1.69` bits blind and `1.25` after
the finetune; grid-aware training should recover most of that remaining bit, so I expect a large drop —
from the control's 31.41 toward the low-to-mid teens, close to full precision, the genuine QAT signal
finally visible. The reason to expect *most* of that `1.25` bits back, and not just a sliver, is that
eight codes carry enough resolution to represent a group faithfully *if* the weights are placed to round
cleanly — the worst-case residual after RTN is `S/2 = max|w_g|/6 ≈ 0.17·max|w_g|` per weight at INT3, with RMS
about `0.10·max|w_g|`, small enough that a weight nudged onto a code contributes little error. STE's whole job is to perform exactly that
placement, so at INT3 the gap between "weights placed blind" (the control's 31.41) and "weights trained
through the grid" should nearly close, landing within a few tenths of a bit of full precision. That is a
sharp, checkable prediction: if STE lands in the teens at INT3 it confirms eight codes are enough once the
optimizer can see them; if it stalls in the twenties it would mean the grid, not the placement, is the
INT3 bottleneck — which I do not expect, because the `0.05·max|w_g|` residual is too small to explain a
twenties-level perplexity. INT2 is the real test and the place I am least sure. Straight-through *does* let the
optimizer fight the four-code grid, so I expect a dramatic improvement over both the control's 94034 and
`no_qat`'s 104422 — collapse should be averted, perplexity brought down from five digits to two, the
model pulled back from worse-than-uniform to genuinely alive. But I can already feel the ceiling in the
construction: STE trains the weights against a grid whose *step size is fixed* at `max|w_g|/qmax =
max|w_g|` per group. At four codes that single fixed knob places the levels crudely, and straight-through
can only move the weights to fit those levels — it cannot move the *levels* to fit the weights. So I
expect INT2 to be rescued from catastrophe but to land well above full precision, perhaps in the tens of
PPL rather than the teens, with a visible gap to whatever a method that *learns the step size* could
reach. If that is what the numbers show — INT4 and INT3 near full precision, INT2 saved but stubbornly
high — then the diagnosis for the next rung is already written: the remaining INT2 error is in the *grid
placement*, which STE holds fixed, so the move is to make the per-group step size a trained parameter.
