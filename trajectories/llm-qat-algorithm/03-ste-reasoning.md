The control rung settled the confound, and the numbers draw a clean line for me. The full-precision
finetune-then-PTQ beat the blind `no_qat` everywhere the grid left room: at INT4 it reached 12.61 PPL,
a degradation of −0.59 — *below* the 13.2033 full-precision baseline, exactly the in-domain finetune
pulling test perplexity down, with the 16-code grid fine enough that the finetuned weights survive it.
At INT3 it dropped `no_qat`'s 42.54 to 31.41 (degradation 18.21) — a real improvement, but the
`max|w_g|/3` floor still leaves it badly above full precision. And at INT2 it stayed catastrophically
collapsed: 94034.51, barely distinguishable from `no_qat`'s 104422 on any scale that matters. That INT2
result is the whole point of the control. It says: a finetune that is *blind to the grid* cannot rescue
two-bit quantization, no matter how much it lowers the loss it rounds from, because four codes destroy
the weights through a rounding the finetune never sees. So the finetune buys domain adaptation and a
flatter basin — and that is all. The bar is now set: any method I call "real QAT" has to clear
`finetune_then_ptq` at the bit-widths where the floor bites, and the place it has to prove itself is
INT2, where the control is still in the tens of thousands.

The reason the control collapses points straight at the fix. Its forward pass during training is pure
fp32 — `fake_quantize_weight` is the identity — so the optimizer never sees the grid. Every gradient it
takes is the gradient of the *full-precision* function, and full-precision loss has no term that says
"sit where rounding with step `S` is cheap." If I want the optimizer to place weights that survive
rounding, the rounding has to be *in the forward pass during training*: compute the loss on the
quantize-dequantized weights, so that every gradient already accounts for the grid. Then the weight is
nudged toward configurations where the rounded model — not the fp model — has low loss. That is the one
structural change between this rung and the control, and it is exactly the scaffold default I have been
carrying since the start: STE QAT.

But putting the grid in the forward immediately hits the wall that made the control take the easy way
out. The fake-quant is `ŵ = S·clamp(round(w/S), qmin, qmax)`, and `round` is a staircase — slope zero
almost everywhere, undefined at the half-integers — while `clamp` is flat outside its range. If I
differentiate this honestly, the chain rule multiplies by that zero Jacobian and annihilates the
gradient: nothing reaches `w`, and the weight is frozen. This is the identical wall that pushed the
field off hard-threshold units and onto sigmoids decades ago — a constant-by-parts graph kills gradient
learning. So the honest derivative is useless here, and I have to ask what nonzero proxy to put in its
place on the backward pass, while keeping the forward exactly the hard quantizer.

The recipe that nags at me is the simplest possible one, the same one that makes rectifiers train:
pretend the slope was one. Back-propagate through the quantize-dequantize *as if it had been the
identity*. Then the gradient arriving at the quantized output, `∂L/∂ŵ`, is sent unchanged to the
underlying weight: `∂L/∂w = ∂L/∂ŵ`. The forward still computes the true rounded weight; only the
backward edge lies, pretending `∂ŵ/∂w = 1`. Let me sanity-check the sign, because a proxy that points
the wrong way is worse than useless. The quantizer is monotone increasing in `w`: raise `w`, you raise
`round(w/S)`, you raise `ŵ` (within the clip range). So the true sign of "how `ŵ` responds to `w`" is
positive, and the identity proxy assigns slope `+1`, also positive. For a single quantized layer the
straight-through gradient therefore has the right sign — if downstream wants `ŵ` larger, it pushes `w`
larger, exactly as it should. It is biased — the magnitude is fabricated — but the direction is correct
where it matters, and that is what lets the weight reorganize to be quantization-friendly instead of
frozen by a zero Jacobian.

I should poke at whether a less crude proxy is better, since "slope = 1" feels too cheap. The natural
alternative is to back-propagate as if the round were its smooth surrogate, multiplying the incoming
gradient by some bell-shaped factor that peaks at a bin center and decays toward the transitions. But
that *attenuates the gradient exactly where I need it most*. A weight sitting confidently in the middle
of a code cell is already well-quantized; the weights I need to move are the ones near a transition,
where a small nudge flips which code they round to and where the rounding error is largest. The plain
identity proxy keeps the signal alive everywhere — slope one regardless of position in the cell — so a
weight that is rounding badly still gets a full push to move. The cruder proxy is the better one here,
and not by accident: it refuses to attenuate where the useful signal lives.

There is one piece of bookkeeping the identity proxy hides, and it is load-bearing. I cannot store the
weight I am learning at low precision, because each weight has to absorb a long run of tiny, noisy
gradient steps, and averaging those out needs real resolution — coarse storage would round every
infinitesimal step to zero and nothing would move. So I keep a *full-precision master weight*, quantize
it on the fly each forward pass, use the quantized value in the matmul (with straight-through carrying
the gradient back to the master), and let the optimizer accumulate updates in the master. In this
task's edit surface that master *is* `self.linear.weight` — the fp32 parameter the harness's AdamW
already optimizes — and the scale is recomputed from it every forward, so the per-group grid tracks the
current weight. The fixed loop's design makes this automatic: it builds the optimizer over all
`requires_grad` parameters and runs the finetune through the wrapper's fake-quant forward, so I do not
have to manage shadow copies myself. At the end, the harness's no-grad `quantize_dequantize_weight`
materializes the genuine integer model, with no straight-through needed because nothing differentiates
anymore.

Now the precise diff against the control, and against the scaffold default, so I land the *task's* STE
and not a paper's generic version. Versus the control: I flip `fake_quantize_weight` from the identity
back to the straight-through per-group symmetric RTN — `w_dq = w + (w_q − w).detach()`, which is `w_q`
in the forward and slope-one in the backward — and the wrapper's forward calls it every step:
`F.linear(x, fake_quantize_weight(self.linear.weight, ...), self.linear.bias)`. Everything else is held
constant: the same 500-step schedule (`lr=2e-5`, `warmup=50`, cosine to 10%, grad-accum 4, grad-clip
1.0), the same `quantize_dequantize_weight`, the same wrap-and-restore-the-head logic. So the *only*
change from `finetune_then_ptq` is that the grid is now in the forward — which means the gap between
this rung and the control is, by construction, the pure QAT contribution, the finetune effect
subtracted out. Versus the scaffold default: this rung *is* the default fake-quant, but I should be
deliberate that the contribution I am isolating is exactly the *gradient*, nothing more. The scale here
is a fixed max-abs statistic, recomputed each forward, *not* learned. Keeping the scale plain isolates
what straight-through buys on its own — the ability to send a gradient through the round so the master
weight reorganizes to be quantization-friendly — and leaves the obvious next idea, *learning* the step
size to place the levels more cleverly, for the rung after this. The full scaffold module is in the
answer.

So what do I expect, falsifiably, against the numbers I now have? At INT4 the floor is already small,
so I expect STE to roughly match or slightly beat `finetune_then_ptq`'s 12.61 — grid-aware training has
little room to help when blind finetuning already lands below full precision; both should sit near or
under 13.2033. At INT3 I expect the first clear separation: the control reached 31.41 because its
weights were placed without regard to the 8-code grid, whereas STE trains *through* that grid, so the
optimizer can push weights into configurations that round cleanly. I expect a large drop — from the
control's 31.41 toward the low-to-mid teens, close to full precision, the genuine QAT signal finally
visible. INT2 is the real test and the place I am least sure. Straight-through *does* let the optimizer
fight the four-code grid, so I expect a dramatic improvement over both the control's 94034 and
`no_qat`'s 104422 — collapse should be averted, perplexity brought down from five digits to two. But I
can already feel a ceiling in the construction: STE trains the weights against a grid whose *step size
is fixed* at `max|w_g|/qmax = max|w_g|` per group. At four codes that single fixed knob places the
levels crudely, and straight-through can only move the weights to fit those levels — it cannot move the
*levels* to fit the weights. So I expect INT2 to be rescued from catastrophe but to land well above full
precision, perhaps in the tens of PPL rather than the teens, with a visible gap to whatever a method
that *learns the step size* could reach. If that is what the numbers show — INT4 and INT3 near full
precision, INT2 saved but stubbornly high — then the diagnosis for the next rung is already written:
the remaining INT2 error is in the *grid placement*, which STE holds fixed, so the move is to make the
per-group step size a trained parameter.
