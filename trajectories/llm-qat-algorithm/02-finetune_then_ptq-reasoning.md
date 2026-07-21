The PTQ control told me, in numbers, exactly how the blind rounding scales with bit-width, and the
shape is the one I predicted from the `S/2` floor. At INT4, `no_qat` landed at 15.01 PPL, only 1.81
above the 13.2033 full-precision baseline — 15 in-range codes, ~14% jitter, the damage real but
survivable. At INT3 it jumped to 42.54, a degradation of 29.34 — 7 codes, ~32% jitter, the floor
climbing sharply just as the arithmetic said. And at INT2 it detonated: 104422.66 PPL, degradation
104409 — three effective codes, ~95% jitter, every weight smashed onto a grid so coarse the composed
function bears no resemblance to the trained one.

Raw perplexities on a five-order-of-magnitude scale hide the mechanism, so read them in nats — perplexity
is exponentiated cross-entropy, `ln(ppl)`. The baseline is `ln 13.2033 = 2.58` nats. INT4 sits at `2.71`,
a degradation of `0.13` nats (~`0.19` bits/token), barely perceptible. INT3 is at `3.75`, `1.17` nats
(~`1.69` bits) — a real bite. INT2 is at `ln 104422 = 11.56`, `8.98` nats (~`12.9` bits). And here is the
number that reframes the problem: a uniform distribution over this backbone's ~50k vocabulary has entropy
`ln 50304 = 10.83` nats, so the INT2 model's `11.56` nats is *worse than uniform* — not merely uninformed
but confidently wrong, placing mass on tokens the truth rules out. Blind two-bit rounding produces an
actively miscalibrated model, and that tells me the fix cannot be a gentle nudge: the weights have to be
reorganized, not tuned. In bits/token the three degradations are `0.19, 1.69, 12.9`, and in raw
degradation the ratios are `29.34/1.81 ≈ 16×` then `104409/29.34 ≈ 3560×` — the gaps widen
super-linearly exactly as the `0.14 → 0.32 → 0.95` relative-error law demanded, the damage exploding
precisely as the per-weight jitter approaches the per-weight magnitude. So the diagnosis is confirmed and
sharp: a *no-repair* problem, and the only leverage left is to move the weights *before* I round, which
needs the loss, the data, and the optimizer the control switched off.

But before I reach for a QAT signal I have to be honest about a confound the harness was built to expose.
The QAT methods will finetune on WikiText-2 train and be scored on WikiText-2 test — disjoint articles,
same domain. So if a QAT method beats `no_qat`, I cannot immediately credit the *quantization-aware*
part: a plain finetune on in-domain text lowers test cross-entropy all by itself, with no bits involved,
simply because a few hundred steps sharpen a broadly-competent model for this corpus. I need to separate
the in-domain finetune effect from the genuine QAT effect, and the clean way is to run a finetune with
*no QAT signal whatsoever* and quantize it the way `no_qat` did. (A calibration-style PTQ — choosing each
group's scale on a held-out set — would be tempting, but a calibrated scale is already a crude form of
learning the grid, so it would blur the line between "the format plus a finetune" and "the format plus a
trained grid," the two things I most need kept apart; it belongs later if at all.) This rung is the
control: full-precision finetune, then the identical RTN PTQ. Its job is to set the bar a real QAT method
has to clear, so that later I can read any method's improvement over `no_qat` and know how much was just
the finetune talking.

So what does a pre-rounding adjustment buy, and toward what do I adjust? The naive instinct is "make the
quantized weights close to the original weights" — minimize `‖ŵ − w‖`. That is the wrong objective, and
seeing why is the whole game. I do not ship the weights; I ship the *function*, and I am scored on its
loss on text, not on how near its weights sit to some reference. Weight-space distance and loss-space
distance are simply different things, because the loss is wildly anisotropic and a small move along a
high-curvature direction hurts far more than a large move along a flat one. A two-parameter caricature
makes it concrete: let `L = 100·w₁² + 0.01·w₂²`, a valley 10⁴ times stiffer in `w₁`. Nudge `w₁` by `0.1`
— squared displacement `0.01`, loss cost `1.0`. Hurl `w₂` by `1.0` — squared displacement a hundred
times larger, yet loss cost `0.01`, a hundred times *smaller*. Minimizing `‖ŵ − w‖` optimizes the wrong
one of these, and the grid cares nothing for curvature — it rounds `w₁` and `w₂` with steps set by their
magnitudes, so a blind scheme has no way to spend precision where the loss is stiff. The pretrained
weights were placed to minimize the *pretraining* loss in full precision; they have no reason to be the
weights that, after a coarse grid, minimize the test perplexity I care about. So the thing to minimize
before rounding is the task loss.

That tells me a plain finetune buys two distinct things, and I keep them separate. The first is *domain*:
even with no quantization, a short finetune on eval-distribution text lowers cross-entropy — the very
confound I am measuring. The second is *robustness to rounding*: a finetune driven by the real loss
tends to settle weights into a flatter, lower basin, and weights in a flat basin are, almost by
definition, ones where moving each a little — which is what rounding does — costs little loss. Make that
precise. Expand the post-rounding loss around the finetuned weights: `ΔL ≈ ∇L·Δw + ½ Δwᵀ H Δw`. At a
well-finetuned point `∇L ≈ 0`, so the damage is the quadratic term — the jitter `Δw` contracted against
curvature `H`. A finetune can lower `L` and flatten `H`, both shrinking `ΔL`. But `Δw` itself is set
entirely by the grid, which the finetune never sees, so it cannot reduce `‖Δw‖`. At INT4 `‖Δw‖` is small
(~14% relative) and the quadratic term is a gentle correction the flattening helps with; if I model one
direction as `L = ½λ(w−w*)²` and let the finetune halve `λ`, the induced bump `½λΔw²` halves too — a
clean win where the parabola actually holds. At INT2 `Δw` flings each weight a full magnitude away from
`w*`, the second-order expansion is not even valid, and halving a curvature that only describes the
neighborhood of `w*` does nothing for a jump that leaves the neighborhood entirely. So the flat-basin
effect is real but bit-width-gated: it helps where the grid is already fine and does essentially nothing
where it is coarse — not because the finetune tries less hard but because two-bit rounding is not a
perturbation the basin's shape can absorb. Both effects come from the same cheap thing — run ordinary SGD
on the task loss, in full precision, keeping the master weights in fp32 the entire time and collapsing to
the grid exactly once at the finish.

Now the care: there is a tempting next move that would quietly turn this control into a QAT method.
Accuracy-recovery work fine-tunes *with the quantization in the loop*, so the optimizer sees the grid and
places weights to sit well on it — powerful, but no longer the plain train-in-fp-then-quantize recipe. If
I want the control cleanly, the training forward must *not* simulate the quantization at all: the weight
goes into the matmul as the fp32 weight, no fake-quant, no grid in sight. In the edit surface that means
`fake_quantize_weight` is the identity — `return weight`, full stop — and the rounding is bolted on once,
after the loop. The very next rung flips it back to the straight-through quantizer so the grid *is* in the
forward; the gap between that rung and this one is the genuine QAT contribution, with the in-domain
finetune held constant because both use the identical schedule.

And the limitation must be seen clearly: a plain fp finetune is *not* secretly optimizing the rounded
model. It has no term saying "place this weight so that after rounding with step `S` the loss is still
low," and it cannot target the named failure modes — the hundredfold cross-channel range spread and the
outlier weights survive a finetune, because nothing in fp32 cross-entropy penalizes a wide per-channel
range or a stray large value; those only become costly *through* the rounding, which the finetune never
sees. Worse, the schedule runs with `weight_decay = 0.0`, so there is not even a mild inward pull;
ordinary language-model finetuning tends to *grow* a few weights, and a larger outlier only inflates its
group's `max|w_g|` and coarsens the grid for its 128 neighbors — the finetune could even move the wrong
way for a rounding it never anticipates. When I finally apply the one-shot RTN, I still pay the `S/2`
floor.

The control is only as valuable as the subtraction it lets me perform, and I want that defined before I
have the numbers rather than reverse-engineered after. Fix a bit-width and read three degradations off
the same format and harness: `d0` from `no_qat` (blind rounding), `d1` from this control (fp finetune
then the same rounding), and later `d2` from a genuine QAT method. Then `d0 − d1` is the *finetune*
contribution — pure adaptation plus flat-basin robustness, no bits — and `d1 − d2` is the
*quantization-aware* contribution, what putting the grid in the forward buys once the finetune is held
fixed. Without this rung I would only have `d0 − d2` and no way to split it. And the split should be
asymmetric: at INT4 I expect `d0 − d1` to be most of the story (the grid is nearly free, little QAT
signal left) and at INT2 almost nothing (a grid-blind finetune cannot touch three-code collapse), so
whatever a QAT method later achieves at INT2 is essentially *all* `d1 − d2` — the cleanest possible
attribution, handed to me by the identical-schedule design. Finetune owns INT4, QAT must own INT2.

That lets me predict the failure signature. At INT4 the floor is small, so the finetune's domain gain
should dominate: I expect this control to beat `no_qat`'s 15.01 comfortably, and — because in-domain
finetuning can pull test PPL below even the un-finetuned full-precision baseline — it could land *under*
13.2033, a negative degradation. The domain gain for a model this size is on the order of a couple of
tenths of a nat, and the INT4 grid costs only `0.13`; if the gain exceeds the cost, the finetuned INT4
model ends up below the baseline the baseline never received a finetune to match. That is not a paradox
but exactly the confound this rung exists to expose — reading a negative INT4 degradation as QAT success
would be the precise error the control prevents, so I predict it and predict it means nothing about bits.
At INT3 the finetune lowers the loss I round from, so I expect a clear improvement over 42.54, but the
`max|w_g|/3` floor leaves it meaningfully worse than full precision, degradation well into the double
digits. At INT2, here is the crux: the finetune is blind to the grid, the floor is `max|w_g|/2`, the
rounding replaces each weight wholesale, and the domain gain of a few tenths of a bit cannot dent a gap
of ~12.9 bits. I expect INT2 to *stay collapsed*, better than `no_qat`'s 104422 by at most a small
fraction — still in the thousands-to-tens-of-thousands, still worse than uniform, nowhere near usable. If
INT2 collapses for this control but a QAT method later holds INT2 near full precision, *that gap is the
genuine QAT contribution*, proven, with the finetune subtracted out. That is the entire reason this rung
exists.

So the recipe is forced and almost embarrassingly simple. Run the ordinary finetune in fp32 with the
fixed schedule — the same `learning_rate=2e-5`, `num_steps=500`, `batch_size=2`, `grad_accum=4`,
`max_grad_norm=1.0`, `warmup_steps=50`, `weight_decay=0.0` every QAT rung will use, so the finetune is
held constant — master weights in fp32 throughout. Then the fixed harness walks every wrapped linear,
applies the no-grad per-group symmetric RTN, and evaluates. The wrapper's forward is a plain linear on
`self.linear.weight`, correct both during training (pure fp) and at eval (the harness has already
overwritten the weight with the materialized INT-`B` value). This differs from the scaffold default in
exactly one place — the default's `fake_quantize_weight` simulates the grid, this control's is the
identity — and that single change is the whole definition of "no QAT signal." The
`quantize_dequantize_weight` and the wrap-and-restore-the-head logic are identical to `no_qat`'s. The
full scaffold module is in the answer. The one structural change that would flip this control into a
method is a single line — put the grid back in the training forward instead of the identity — and that so
small an edit separates "collapses at INT2" from whatever comes next is itself the cleanest statement of
what the control isolates: everything except grid-awareness is already here, held fixed, waiting to be
subtracted.
