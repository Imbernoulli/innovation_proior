The PTQ control told me, in numbers, exactly how the blind rounding scales with bit-width, and the
shape is precisely the one I predicted from the `S/2` floor. At INT4, `no_qat` landed at 15.01 PPL,
only 1.81 above the 13.2033 full-precision baseline — 15 in-range codes, ~14% relative weight jitter, the
damage real but survivable, the regime the old 8-bit inference work lived in. At INT3 it jumped to
42.54, a degradation of 29.34 — 7 codes, ~32% jitter, the floor climbing sharply just as the arithmetic
said. And at INT2 it detonated: 104422.66 PPL, a degradation of 104409 — three effective codes, ~95%
jitter, every weight smashed onto a grid so coarse that the composed function bears no resemblance to the
trained one.

Let me read those three numbers the way that will actually guide me, because raw perplexities on a
five-order-of-magnitude scale hide the mechanism. Perplexity is the exponentiated cross-entropy, so the
honest unit is nats (or bits) per token, `ln(ppl)`. The baseline is `ln 13.2033 = 2.58` nats. INT4 sits
at `ln 15.01 = 2.71`, a degradation of `0.13` nats, about `0.19` bits per token — a fifth of a bit of
extra surprise per token, barely perceptible. INT3 is at `ln 42.54 = 3.75`, `1.17` nats, about `1.69`
bits per token — a real bite, over a bit-and-a-half of information lost on every token. INT2 is at
`ln 104422 = 11.56`, `8.98` nats, about `12.9` bits per token. And here is the number that reframes the
whole problem: a uniform next-token distribution over this backbone's ~50k vocabulary has entropy
`ln 50304 = 10.83` nats, so the INT2 model's `11.56` nats is *worse than uniform*. It is not merely
uninformed — it is confidently wrong, placing mass on tokens the truth rules out. Blind two-bit rounding
does not degrade the model toward ignorance; it produces an actively miscalibrated one. That is a
sharper diagnosis than "collapse," and it tells me the fix cannot be a gentle nudge — the weights have
to be reorganized, not tuned.

The three degradations also confirm the *shape* I predicted rather than just the direction. In bits per
token they are `0.19, 1.69, 12.9`; each step multiplies the last by roughly nine and eight. In raw
degradation the ratios are even starker: `29.34 / 1.81 ≈ 16×` from INT4 to INT3, then `104409 / 29.34 ≈
3560×` from INT3 to INT2. The gaps widen super-linearly across the ladder, exactly as the `0.14 → 0.32 →
0.95` relative-error law demanded — the damage explodes precisely as the per-weight jitter approaches the
per-weight magnitude, when the rounding stops being a perturbation of the weight and becomes a
replacement of it. So the diagnosis from the control is confirmed and sharp: this is a *no-repair*
problem. The grid is fixed and the rounding is irreversible, and `no_qat` has no mechanism to compensate.
The only leverage left is to move the weights *before* I round, and moving weights needs the loss, the
data, and the optimizer that the control deliberately switched off.

But before I reach for a QAT signal, I have to be honest about a confound that the harness was built to
expose, and to see it I have to lay out what I could reach for right now. Three moves are on the table.
I could jump straight to putting the grid in the forward and training through it — the obvious QAT step —
but then when it beats `no_qat` I will not be able to say *why*, because two things changed at once
(the finetune and the grid-awareness). I could reach for calibration-style PTQ, choosing each group's
scale on a small held-out set to trade rounding against clipping without any gradient — but that is
already a data-aware grid, a different lineage, and it would muddy the clean "no algorithm" baseline I
want next to it. Worse, it would pre-empt the very question the ladder is built to answer: a calibrated
scale *is* a crude form of learning the grid, so putting it here would blur the line between "the format
plus a finetune" and "the format plus a trained grid," which are the two things I most need kept apart.
The whole design is a sequence of single-variable changes, and calibration would change two at once
(finetune *and* a data-aware grid), so it belongs later if at all, not in the control. Or I could isolate the one confound that threatens every later comparison. The QAT
methods are going to finetune on WikiText-2 train and be scored on WikiText-2 test — disjoint articles,
but the same domain. So if a QAT method beats `no_qat`, I cannot immediately credit the
*quantization-aware* part: a plain finetune on in-domain text lowers test cross-entropy all by itself,
with no bits involved at all, simply because the model arrives knowing language broadly and a few
hundred steps sharpen it for this particular corpus. I need to separate the in-domain finetune effect
from the genuine QAT effect, and the clean way to do that is to run a finetune that has *no QAT signal
whatsoever* and quantize it the same way `no_qat` did. That is this rung: full-precision finetune, then
the identical RTN PTQ. It is a control, not a contender — its job is to set the bar that a real QAT
method has to clear, so that later I can read any QAT method's improvement over `no_qat` and know how
much of it was just the finetune talking.

So what does a pre-rounding adjustment actually buy, and toward what do I adjust? The naive instinct is
"make the quantized weights close to the original weights" — minimize `‖ŵ − w‖`. Stare at that and it is
the wrong objective, and seeing why is the whole game. I do not ship the weights; I ship the *function*
the network computes, and I am scored on its loss on text, not on how near its weights sit to some
reference. Two weight sets far apart in MSE can compute nearly the same function, and a weight set
close in MSE can have noticeably worse loss — weight-space distance and loss-space distance are simply
different things, because the loss is a wildly anisotropic function of the weights and a small move
along a high-curvature direction hurts far more than a large move along a flat one. A two-parameter
caricature makes the decoupling concrete: let `L = 100·w₁² + 0.01·w₂²`, a valley 10⁴ times stiffer in
`w₁` than in `w₂`. Perturbation A nudges `w₁` by `0.1` — squared displacement `0.01`, loss cost
`100·0.01 = 1.0`. Perturbation B hurls `w₂` by `1.0` — squared displacement `1.0`, a hundred times larger
in MSE, yet loss cost `0.01·1.0 = 0.01`, a hundred times *smaller*. So the perturbation that is a hundred
times worse by weight-space distance is a hundred times better by loss. Minimizing `‖ŵ − w‖` optimizes
the wrong one of these two; the grid, of course, cares nothing for curvature and rounds `w₁` and `w₂`
with steps set by their magnitudes, so a blind scheme has no way to spend its precision where the loss is
stiff. The pretrained
weights were placed to minimize the *pretraining* loss in *full precision*; they have no reason to be
the weights that, after being smashed onto a coarse grid, minimize the test perplexity I care about. So
the thing to minimize before rounding is the task loss — next-token cross-entropy on the in-domain text —
and the pretrained weights are not its minimizer in any sense relevant to me.

That immediately tells me a plain finetune can buy two distinct things, and I keep them separate. The
first is *domain*: even with no quantization at all, a short finetune on text from the eval distribution
lowers cross-entropy on that distribution — ordinary adaptation, nothing to do with bits. This is exactly
the confound I am trying to measure. The second is *robustness to rounding*: a finetune driven by the
real loss tends to settle the weights into a flatter, lower region of the loss surface, and weights in a
flat basin are, almost by definition, ones where moving each a little — which is exactly what rounding
does — costs little loss. Let me make that second effect precise enough to bound it, because it is the
one that might, optimistically, help at low bits. Expand the post-rounding loss around the finetuned
weights: `ΔL ≈ ∇L·Δw + ½ Δwᵀ H Δw`. At a well-finetuned point `∇L ≈ 0`, so the damage is governed by the
quadratic term, the rounding jitter `Δw` contracted against the curvature `H`. A finetune can lower `L`
(the loss I round *from*) and can flatten `H` somewhat, both of which shrink `ΔL`. But `Δw` itself — the
size of the rounding jump — is set entirely by the grid, and the finetune never sees the grid, so it
cannot reduce `‖Δw‖`. At INT4 `‖Δw‖` is small (14% relative) and the quadratic term is a gentle
correction the flattening can help with; at INT2 `‖Δw‖` is ~95% relative, the second-order expansion is
not even valid — the rounding hurls each weight a full magnitude away — and no amount of lowering `L` or
softening `H` around the fp point can offset a displacement that large. So the flat-basin effect is real
but bit-width-gated: it helps where the grid is already fine and does essentially nothing where it is
coarse. Put a toy number on it: model one direction of the loss as `L(w) = ½ λ (w − w*)²` with curvature
`λ`, and let the finetune both lower the floor and halve the curvature it leaves behind, `λ → λ/2`. The
rounding jitter is `Δw ≈ S/sqrt(12)`, so the induced loss bump is `½ λ Δw²`, and halving `λ` halves that
bump — a clean win at INT4, where `Δw` is small (`S/sqrt(12) ≈ 0.04·max|w_g|` per weight) and the
quadratic model actually holds. At INT2, though, `Δw ≈ 0.29·max|w_g|` and the weight is flung a full
magnitude or more away from `w*`; the quadratic `½ λ Δw²` is no longer a valid local model, the loss along
that direction is nowhere near parabolic that far out, and halving a curvature that only describes the
neighborhood of `w*` does nothing for a jump that leaves the neighborhood entirely. So the same flattening
that buys real robustness at four bits is simply inapplicable at two — not because the finetune tries less
hard, but because the rounding at two bits is not a perturbation the basin's shape can absorb. Both effects, in any case, come from the same cheap thing — run ordinary SGD on the task loss, in
full precision, before I round — and I let it run with the master weights kept in fp32 the entire time
and round only at the very end, which is the right reading of the low-precision-training lineage: keep a
high-precision master copy through all the updates so thousands of small gradient steps actually
accumulate, and collapse to the grid exactly once at the finish.

Now I have to be careful, because there is a tempting next move that would change the recipe and quietly
turn this control into a QAT method. The accuracy-recovery line of work fine-tunes *with the
quantization in the loop* — quantize first, then retrain the quantized representation against the loss,
so the optimizer can see the grid and place weights to sit well on it. That is powerful, but it is no
longer the plain train-in-floating-point-then-quantize recipe, and it would defeat the purpose of this
rung. If I want the control cleanly, the forward pass during training must *not* simulate the
quantization at all. The weight goes into the matmul exactly as the fp32 weight — no fake-quant, no
rounding, no grid in sight. In the task's edit surface that means `fake_quantize_weight` is the
identity: `return weight`, full stop. The finetune is plain full-precision training, and the rounding
is bolted on, once, only after the loop is over. The very next rung will do the opposite, flipping
`fake_quantize_weight` back to the straight-through quantizer so the grid *is* in the forward; the gap
between that rung and this one will be the genuine QAT contribution, with the in-domain finetune held
constant because both use the identical schedule.

Let me make sure I see the limitation, because a plain fp finetune is *not* secretly optimizing the
rounded model. The finetune is blind to the grid by construction. It can drive the task loss down and it
tends to flatten the basin, but it has no term in its objective that says "place this weight so that
*after rounding with step S* the loss is still low." It cannot target the named failure modes either:
the cross-channel range spread of over a hundredfold and the outlier weights survive a finetune, because
nothing in next-token cross-entropy on fp32 weights directly penalizes a wide per-channel range or a
stray large value — those only become costly *through* the rounding, which the finetune never sees. In
fact, the schedule runs with `weight_decay = 0.0`, so there is not even the mild inward pull that decay
would provide; ordinary language-model finetuning left to itself tends to *grow* a few weights, and a
larger outlier in a group only inflates that group's `max|w_g|` and coarsens the grid for its 128
neighbors — so the finetune could even move the wrong way for the rounding it never anticipates, and the
schedule gives it no term that would discourage that. When I
finally apply the one-shot RTN, I still pay the `S/2` floor. The finetune lowered the loss I am rounding
*from* and put me somewhere a bit more robust, but it did not minimize the post-rounding error.

That lets me predict, against the `no_qat` numbers, exactly the failure signature I expect, and the
prediction is the falsifiable point of this rung. At INT4 the floor is small to begin with, so the
finetune's domain gain should dominate: I expect this control to *beat* `no_qat`'s 15.01 comfortably,
and — because the in-domain finetune can pull test PPL below even the full-precision baseline measured
without finetuning — it could land *under* 13.2033, a negative degradation, since the domain adaptation
is worth on the order of a few tenths of a bit per token and the INT4 grid only costs `0.19`. The 16-code
grid is fine enough that the finetuned weights survive it. At INT3 the picture should split: the finetune
lowers the loss I round from, so I expect a clear improvement over `no_qat`'s 42.54 — the domain gain of
a few tenths of a bit knocks a chunk off the `1.69` bits INT3 was losing — but the `max|w_g|/3` floor is
high enough that the rounded model is meaningfully worse than full precision, degradation well into the
double digits, somewhere between INT4's near-zero and INT2's catastrophe. At INT2, here is the crux: the
finetune is blind to the grid, the floor is `max|w_g|/2`, the rounding replaces each weight wholesale,
and no amount of full-precision loss minimization changes how brutally three codes destroy the weights.
The domain gain is worth a few tenths of a bit; the INT2 loss is `12.9` bits — the finetune cannot dent
a gap two orders of magnitude larger than the tool it brings. I expect INT2 to *stay collapsed*, better
than `no_qat`'s 104422 by at most a small fraction — still in the thousands-to-tens-of-thousands of PPL,
still worse than uniform, nowhere near usable. If INT2 collapses for this control but a QAT method later
holds INT2 near full precision, *that gap is the genuine QAT contribution*, proven, with the finetune
effect subtracted out. That is the entire reason this rung exists.

It is worth writing down the bookkeeping this rung unlocks, because the control is only as valuable as
the subtraction it lets me perform, and I want that subtraction defined before I have the numbers rather
than reverse-engineered after. Fix a bit-width and read three degradations off the same format and
harness: `d0` from `no_qat` (blind rounding, no training), `d1` from this control (fp finetune, then the
same rounding), and later `d2` from a genuine QAT method (grid in the forward, same schedule). Then
`d0 − d1` is the *finetune* contribution — pure in-domain adaptation plus flat-basin robustness, no bits
involved — and `d1 − d2` is the *quantization-aware* contribution — what putting the grid in the forward
buys once the finetune is held fixed. Without this rung I would only have `d0 − d2` and no way to split
it; with it, every later method reports a clean QAT number. At INT4 I expect `d0 − d1` to be most of the
story (the grid is nearly free, so there is little QAT signal left to find) and at INT2 I expect `d0 − d1`
to be almost nothing (a grid-blind finetune cannot touch three-code collapse), so that whatever a QAT
method later achieves at INT2 is essentially *all* `d1 − d2` — the cleanest possible attribution, handed
to me for free by the identical-schedule design. That asymmetry — finetune owns INT4, QAT must own INT2 —
is the prediction that makes this control worth a run of its own.

Let me sanity-check the boldest single claim, that INT4 could land *below* the 13.2033 baseline, since a
negative degradation is the kind of result that looks like a bug if I have not earned it. The baseline
was measured on the pretrained model with no finetune at all. A few hundred steps of in-domain finetuning
on WikiText-2 train typically lowers held-out cross-entropy on the same corpus by something on the order
of a couple of tenths of a nat for a model this size — call it `0.15–0.25` nats. The INT4 grid, by the
rung-1 accounting, costs about `0.13` nats (`0.19` bits). If the domain gain exceeds the grid cost, the
finetuned-then-rounded INT4 model ends up *below* the un-finetuned fp baseline — a negative degradation
that is not a paradox but exactly the confound this rung exists to expose: the "improvement" is domain
adaptation the baseline never received, and it has nothing to do with quantization. Reading a negative
INT4 degradation as QAT success would be the precise error the control is built to prevent. So I predict
it, and I predict it means nothing about bits.

So the recipe is forced and almost embarrassingly simple. Run the ordinary finetune in fp32 with the
fixed text-modeling schedule — the same `learning_rate=2e-5`, `num_steps=500`, `batch_size=2`,
`gradient_accumulation_steps=4`, `max_grad_norm=1.0`, `warmup_steps=50`, `weight_decay=0.0` that every
QAT rung will use, so the finetune is held constant across the comparison — with all master weights in
fp32 throughout. Then, after the loop, the fixed harness walks every wrapped linear and applies the
no-grad per-group symmetric RTN, materializing the genuine INT-`B` model, and evaluates. The wrapper's
forward is a plain linear on `self.linear.weight`, which is correct both during training (pure fp, no
fake-quant) and at eval (the harness has already overwritten the weight with the materialized INT-`B`
value). This differs from the scaffold default in exactly one place — the default's `fake_quantize_weight`
simulates the grid in the forward, whereas this control's is the identity — and that single change is the
whole definition of "no QAT signal." The `quantize_dequantize_weight` and the `prepare_qat_model`
wrap-and-restore-the-head logic are identical to `no_qat`'s. The full scaffold module is in the answer.

The causal chain in one breath: the PTQ control's measured collapse — 15.01 / 42.54 / 104422 across
INT4/INT3/INT2, which in bits per token is `0.19 / 1.69 / 12.9` and puts INT2 past the uniform-model
ceiling — confirms a no-repair problem, so the leverage is moving weights before rounding; but because
train and eval share a domain, I first have to measure the plain in-domain finetune effect, so I run a
full-precision finetune (identity `fake_quantize_weight`) on the same schedule, then the identical RTN;
this buys domain adaptation (a few tenths of a bit) and a flatter basin but is blind to the grid by
construction, so I expect it to beat `no_qat` clearly at INT4 (likely below 13.2033) and INT3, yet
*still collapse* at INT2 — and that surviving INT2 gap is precisely the bar the real QAT methods on the
next rungs must clear to prove their gains are QAT signal, not finetune. The one structural change that
would flip this control into a method is a single line — put the grid back in the training forward instead
of the identity — and the fact that so small an edit separates "collapses at INT2" from whatever comes
next is itself the cleanest statement of what the control isolates: everything except grid-awareness is
already here, held fixed, waiting to be subtracted.
