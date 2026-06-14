The PTQ control told me, in numbers, exactly how the blind rounding scales with bit-width, and the
shape is precisely the one I predicted from the `S/2` floor. At INT4, `no_qat` landed at 15.01 PPL,
only 1.81 above the 13.2033 full-precision baseline — 16 codes, in-range spacing `max|w_g|/7`, the
damage real but survivable, the regime the old 8-bit inference work lived in. At INT3 it jumped to
42.54, a degradation of 29.34 — 8 codes, spacing `max|w_g|/3`, the floor climbing sharply just as the
arithmetic said. And at INT2 it detonated: 104422.66 PPL, a degradation of 104409 — four codes, spacing
`max|w_g|`, floor `max|w_g|/2`, every weight smashed onto a grid so coarse that the composed function
bears no resemblance to the trained one. The model is effectively destroyed. So the diagnosis from the
control is confirmed and sharp: this is a *no-repair* problem. The grid is fixed and the rounding is
irreversible, and `no_qat` has no mechanism to compensate. The only leverage left is to move the
weights *before* I round, and moving weights needs the loss, the data, and the optimizer that the
control deliberately switched off.

But before I reach for a QAT signal, I have to be honest about a confound that the harness was built to
expose. The QAT methods are going to finetune on WikiText-2 train and be scored on WikiText-2 test —
disjoint articles, but the same domain. So if a QAT method beats `no_qat`, I cannot immediately credit
the *quantization-aware* part: a plain finetune on in-domain text lowers test cross-entropy all by
itself, with no bits involved at all, simply because the model arrives knowing language broadly and a
few hundred steps sharpen it for this particular corpus. I need to separate the in-domain finetune
effect from the genuine QAT effect, and the clean way to do that is to run a finetune that has *no QAT
signal whatsoever* and quantize it the same way `no_qat` did. That is this rung: full-precision
finetune, then the identical RTN PTQ. It is a control, not a contender — its job is to set the bar that
a real QAT method has to clear, so that later I can read any QAT method's improvement over `no_qat` and
know how much of it was just the finetune talking.

So what does a pre-rounding adjustment actually buy, and toward what do I adjust? The naive instinct is
"make the quantized weights close to the original weights" — minimize `‖ŵ − w‖`. Stare at that and it is
the wrong objective, and seeing why is the whole game. I do not ship the weights; I ship the *function*
the network computes, and I am scored on its loss on text, not on how near its weights sit to some
reference. Two weight sets far apart in MSE can compute nearly the same function, and a weight set
close in MSE can have noticeably worse loss — weight-space distance and loss-space distance are simply
different things. The pretrained weights were placed to minimize the *pretraining* loss in *full
precision*; they have no reason to be the weights that, after being smashed onto a coarse grid,
minimize the test perplexity I care about. So the thing to minimize before rounding is the task loss —
next-token cross-entropy on the in-domain text — and the pretrained weights are not its minimizer in
any sense relevant to me.

That immediately tells me a plain finetune can buy two distinct things, and I keep them separate. The
first is *domain*: even with no quantization at all, a short finetune on text from the eval
distribution lowers cross-entropy on that distribution — ordinary adaptation, nothing to do with bits.
This is exactly the confound I am trying to measure. The second is *robustness to rounding*: a finetune
driven by the real loss tends to settle the weights into a flatter, lower region of the loss surface,
and weights in a flat basin are, almost by definition, ones where moving each a little — which is
exactly what rounding does — costs little loss. So a plain finetune does double duty: it pulls the loss
down for the domain, and it tends to leave the weights somewhere the rounding hurts less. Both effects
come from the same cheap thing — run ordinary SGD on the task loss, in full precision, before I round.
And I let it run with the master weights kept in fp32 the entire time and round only at the very end,
which is the right reading of the low-precision-training lineage: keep a high-precision master copy
through all the updates so thousands of small gradient steps actually accumulate, and collapse to the
grid exactly once at the finish.

Now I have to be careful, because there is a tempting next move that would change the recipe and quietly
turn this control into a QAT method. The accuracy-recovery line of work fine-tunes *with the
quantization in the loop* — quantize first, then retrain the quantized representation against the loss,
so the optimizer can see the grid and place weights to sit well on it. That is powerful, but it is no
longer the plain train-in-floating-point-then-quantize recipe, and it would defeat the purpose of this
rung. If I want the control cleanly, the forward pass during training must *not* simulate the
quantization at all. The weight goes into the matmul exactly as the fp32 weight — no fake-quant, no
rounding, no grid in sight. In the task's edit surface that means `fake_quantize_weight` is the
identity: `return weight`, full stop. The finetune is plain full-precision training, and the rounding
is bolted on, once, only after the loop is over. The very next rung — STE QAT — will do the opposite,
flipping `fake_quantize_weight` back to the straight-through quantizer so the grid *is* in the forward;
the gap between that rung and this one will be the genuine QAT contribution, with the in-domain finetune
held constant because both use the identical schedule.

Let me make sure I see the limitation, because a plain fp finetune is *not* secretly optimizing the
rounded model. The finetune is blind to the grid by construction. It can drive the task loss down and
it tends to flatten the basin, but it has no term in its objective that says "place this weight so that
*after rounding with step S* the loss is still low." It cannot target the named failure modes either:
the cross-channel range spread of over a hundredfold and the outlier weights survive a finetune,
because nothing in next-token cross-entropy on fp32 weights directly penalizes a wide per-channel range
or a stray large value — those only become costly *through* the rounding, which the finetune never
sees. So when I finally apply the one-shot RTN, I still pay the `S/2` floor. The finetune lowered the
loss I am rounding *from* and put me somewhere a bit more robust, but it did not minimize the
post-rounding error.

That lets me predict, against the `no_qat` numbers, exactly the failure signature I expect, and the
prediction is the falsifiable point of this rung. At INT4 the floor is small to begin with, so the
finetune's domain gain should dominate: I expect this control to *beat* `no_qat`'s 15.01 comfortably,
and — because the in-domain finetune can pull test PPL below even the full-precision baseline measured
without finetuning — it could land *under* 13.2033, a negative degradation. The 16-code grid is fine
enough that the finetuned weights survive it. At INT3 the picture should split: the finetune lowers the
loss I round from, so I expect a clear improvement over `no_qat`'s 42.54, but the `max|w_g|/3` floor is
high enough that the rounded model is meaningfully worse than full precision — degradation well into the
double digits, somewhere between INT4's near-zero and INT2's catastrophe. At INT2, here is the crux: the
finetune is blind to the grid, the floor is `max|w_g|/2`, and no amount of full-precision loss
minimization changes how brutally four codes destroy the weights. I expect INT2 to *stay collapsed* —
better than `no_qat`'s 104422 only because the finetuned weights are a little more clustered, but still
in the thousands-to-tens-of-thousands of PPL, nowhere near usable. If INT2 collapses for this control
but a QAT method later holds INT2 near full precision, *that gap is the genuine QAT contribution*,
proven, with the finetune effect subtracted out. That is the entire reason this rung exists.

So the recipe is forced and almost embarrassingly simple. Run the ordinary finetune in fp32 with the
fixed text-modeling schedule — the same `learning_rate=2e-5`, `num_steps=500`, `batch_size=2`,
`gradient_accumulation_steps=4`, `max_grad_norm=1.0`, `warmup_steps=50`, `weight_decay=0.0` that every
QAT rung will use, so the finetune is held constant across the comparison — with all master weights in
fp32 throughout. Then, after the loop, the fixed harness walks every wrapped linear and applies the
no-grad per-group symmetric RTN, materializing the genuine INT-`B` model, and evaluates. The wrapper's
forward is a plain linear on `self.linear.weight`, which is correct both during training (pure fp, no
fake-quant) and at eval (the harness has already overwritten the weight with the materialized INT-`B`
value). This differs from the scaffold default in exactly one place — the default's
`fake_quantize_weight` simulates the grid in the forward, whereas this control's is the identity — and
that single change is the whole definition of "no QAT signal." The `quantize_dequantize_weight` and the
`prepare_qat_model` wrap-and-restore-the-head logic are identical to `no_qat`'s. The full scaffold
module is in the answer.

The causal chain in one breath: the PTQ control's measured collapse — 15.01 / 42.54 / 104422 across
INT4/INT3/INT2 — confirms a no-repair problem, so the leverage is moving weights before rounding; but
because train and eval share a domain, I first have to measure the plain in-domain finetune effect, so
I run a full-precision finetune (identity `fake_quantize_weight`) on the same schedule, then the
identical RTN; this buys domain adaptation and a flatter basin but is blind to the grid by
construction, so I expect it to beat `no_qat` clearly at INT4 (likely below 13.2033) and INT3, yet
*still collapse* at INT2 — and that surviving INT2 gap is precisely the bar the real QAT methods on the
next rungs must clear to prove their gains are QAT signal, not finetune.
