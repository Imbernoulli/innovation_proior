The parallel rung came back and cleared the half I most needed cleared: it trained stably to completion —
val_loss 2.3112, WikiText-2 perplexity 45.98, LAMBADA 70.96, ARC-Easy 54.76, HellaSwag 32.93, and the fastest
wall-clock I will see, 19,747s training against 406s eval — so dropping the mean-subtraction and the bias cost
me nothing catastrophic, exactly as I argued from the spread-control angle. The init check I did on paper held
up: two writes per block, `1/√48` still matching, no variance pathology from summing the two branches. So I
carry RMSNorm as the substrate norm from here without re-litigating it.

Before acting on the headline number I read the whole feedback row, because a clean diagnosis is what tells me
where to spend next. Exponentiating, the in-distribution FineWeb perplexity is `e^2.3112 ≈ 10.09`, four to
seven times below the held-out `45.98` on WikiText-2 and `70.96` on LAMBADA — as it should be, since the model
trained on FineWeb and is scored out-of-distribution on the others; what matters is not the gap but that all
three point the same way, so I read the perplexities as *direction* witnesses for `val_loss`, not commensurable
numbers. Wall-clock is `19747`s training against `406`s eval — eval is under 2.1% of the run, so any
rung-to-rung speed comparison is dominated by training, `19747/12030 ≈ 1.64` s/iter, my first fixed point on
what a step of this model costs. The downstream row read against chance tells me which columns carry signal:
ARC-Easy `54.76` sits well clear of its ~25% floor, HellaSwag `32.93` is only ~8 points over 25% (a task a
355M model barely does), PIQA `64.42` is 14 over the two-way 50%, WinoGrande `50.2` is *at* chance and carries
nothing. So on later rungs the downstream witnesses worth reading are ARC-Easy and PIQA; HellaSwag and
WinoGrande are too close to their floors to move meaningfully. And the discipline the setup forces: this is one
seed, so I have no run-to-run variance estimate at all — a `val_loss` move of order a thousandth could sit
inside noise a second seed would reveal, so I will not over-read small gaps.

The number itself: 2.3112 is the *worst* validation loss on the board, exactly as I predicted for the rung
built to pay the small-scale parallel tax, and the witnesses read consistently — perplexities high and
agreeing in direction (a uniform quality shift, not a stranger interaction), downstream unremarkable, and
wall-clock cheapest, confirming the modest speed win I guessed. The quality bill is *not* about the norm rule,
which read neutral; it is the wiring. "Fastest but worst" says I have spare quality budget, and the lever that
spent it is the block structure. So I stop simplifying the block.

Here is the structural problem the parallel rung made vivid. In a pre-norm residual stream nothing ever
normalizes the *output* of a sublayer before it is added back: `x ← x + Attn(LN(x))` normalizes attention's
*input*, but whatever magnitude its projection produces is written into the stream raw, and the same for the
MLP. Across 24 blocks those raw writes accumulate and residual-stream variance grows monotonically with depth —
the well-known cost of pure pre-norm, the reason it lands slightly above a successfully-trained post-norm
model: later blocks write their contribution onto a stream whose magnitude has drifted far from what the
norm-then-project pipeline was implicitly calibrated for, and the relative size of each new contribution
shrinks as the stream grows. The parallel rung made this worse, summing two raw branch outputs per block behind
a non-recentering norm with nothing behind them. So my measured failure and pre-norm's inherent failure are the
same family — un-normalized writes into a growing stream — and the parallel rung just turned the dial up.

Now, how to spend the recovered budget. The conservative move is to put the sublayers back in sequential
order — restoring the cross-talk the parallel rung lost — and stop; but that does nothing about the
un-normalized-write half, which the parallel rung just made vivid and which pure pre-norm carries anyway. Pure
post-LN, `x ← LN(x + Attn(x))`, controls the variance beautifully by renormalizing the whole stream at every
block, but it puts the norm back on the *main residual path* and destroys the identity shortcut that gives
pre-norm its depth-stable, warmup-free gradients — and to train it cleanly here I would almost certainly have
to reach into `CONFIG_OVERRIDES` and lengthen the `0.04 × 12030 ≈ 481`-iteration warmup or drop the learning
rate, dragging the optimizer into a change whose diagnosis was wiring, not optimization. So post-LN is out. The
move that gets *both* is to keep pre-norm's input normalization for the gradient and separately normalize each
sublayer's *output* before the residual add — which *subsumes* the sequential revert too. So I am not choosing
between "revert" and "add control"; the real question is only whether the extra output norm earns its keep.

One non-norm alternative deserves a hearing, because it also attacks un-normalized writes: scale each branch
output by a learnable per-channel diagonal initialized near zero, a LayerScale-style gate
`x ← x + diag(λ)·Attn(LN(x))` with `λ ≈ 1e-4`. It starts every contribution near zero so the residual barely
grows early, and the optimizer dials each branch up as it earns its place — which sounds like the variance
discipline I want. But walk what it guarantees: a learned diagonal is a *fixed* multiplier once trained, not a
renormalizer. If a sublayer's `c_proj` grows large during training, `diag(λ)` scales that output by whatever
constant the optimizer settled on, not back toward a controlled magnitude — so it puts *no bound* on trained
variance growth, precisely the quantity I am trying to bound. It is a gating/capacity change (a new length-1024
vector per branch), and its near-zero init trades the stream-growth problem for a slow-start problem I have no
warmup budget to absorb. It treats the *symptom*, not the mechanism I want — a per-block operation that re-pins
each contribution no matter how the projection drifts. I set it aside.

That leaves the sandwich: keep the pre-norm on each sublayer's input — `LN_pre(x)` into attention, preserving
the well-conditioned gradient at initialization — and add a second norm on the sublayer's *output* before the
residual add, `x ← x + LN_post(Attn(LN_pre(x)))`, and identically `x ← x + LN_post2(MLP(LN_pre2(x)))`. `LN_pre`
controls the distribution the sublayer *sees*, which keeps the gradient bounded and depth-independent — the
pre-norm property I refuse to lose. `LN_post` controls the magnitude of what the sublayer *contributes*, so
each block writes a controlled-scale quantity regardless of how large its projection has grown, bounding the
variance growth the way post-norm does — but with the norm on the branch output *before* the add, never on the
main path, so the identity shortcut `x ← x + (normalized branch)` stays intact. That is the sandwich
(CogView-style) placement: pre-LN for the gradient, post-LN for the variance, the residual add threaded
between, all four norms the RMSNorm carried from the first rung. I am spending normalization, the exact
opposite trade from the parallel rung, on the bet that controlled branch contributions are worth more at 355M
than the half-norm speed saving was.

The four norms cost negligible capacity: four length-1024 gains per block, 98,304 total, double the plain
sequential block and quadruple the parallel rung — but 0.028% of a 355M model, essentially no capacity. So this
rung tests *placement*, not parameters. In wall-clock it is the most normalization on the ladder, four times
the parallel rung's one; from the single point I have (19,747s at one norm) I cannot yet pin the per-norm cost,
but a norm is a small, memory-bound kernel, so three extra per block should push training into the low-20,000s,
the *slowest* rung — a near-mirror of the parallel speed win run in reverse. I consent to that price.

The initialization is where my first instinct is wrong, and I only catch it by doing the algebra. My reflex:
`LN_post` gains start at 1 and `c_proj` already scales the branch output down by `1/√48`, so the initial
contribution should be small and conservative. But `LN_post` is applied *after* `c_proj` and is
scale-invariant — `LN_post(αy) = αy/RMS(αy)·γ = y/RMS(y)·γ`, the `α` divides straight out — so the
`1/√48 ≈ 0.1443` that `c_proj` applies is erased: the raw branch output sits at `RMS ≈ 0.1443` relative to a
unit-RMS input, the small conservative write the substrate intends, and `LN_post` re-pins it to `RMS = |γ| = 1`.
That is `√48 ≈ 6.9×` *larger* than the plain block's `1/√48`-scaled write; summed over 48 writes the residual
RMS at the top of the stack starts near `√48`, a much steeper init climb than the plain block's near-doubling.
So my reflex was backwards — the sandwich starts by writing *bigger* contributions, not smaller. Is that a
pathology? No: `LN_pre` re-normalizes each sublayer's input to unit RMS regardless of how large the residual
has grown, so no sublayer sees the inflated stream directly, and the final `ln_f` normalizes before the LM
head. The `1/√48` bookkeeping is untouched in *count* (still two writes per block) even though `LN_post`
overrides its effect on each write's magnitude. So the honest picture is that this is not variance control *at
initialization* — at init it grows the stream faster; the real value of `LN_post` is a bound on the *trained*
growth, re-pinning each contribution back toward the gain as `c_proj` drifts. The benefit is about late
training, not the start.

And that exposes the caveat: `LN_post`'s bound is only as tight as its learned gain, which starts at 1 but the
optimizer is free to grow. If it does, the "controlled" contribution re-inflates and the variance control leaks
— a *soft* prior, not a hard cap; the network can learn to partially undo it wherever undoing it lowers the
loss. At 24 layers — deep enough for variance growth to be a real effect but not so deep that pure pre-norm is
in trouble — the benefit may be modest: it should clearly beat the parallel rung, which had *no* output control
and lost cross-talk on top, but whether it beats simply swapping in RMSNorm with no output control, cross-talk
intact and the clean pre-norm gradient, is the genuinely open question this rung answers. My prior is that the
output norm is stability insurance more than a final-loss lever at this depth. I touch no `CONFIG_OVERRIDES`:
the parallel rung's problem was wiring, not optimization, and I have no diagnostic pointing at the optimizer.

So this rung keeps the first rung's RMSNorm class verbatim and replaces `Block` with the sandwich — `ln_pre1`,
attention, `ln_post1`; `ln_pre2`, MLP, `ln_post2`; the residual adds threaded between, both sublayers
sequential again. The distilled module and the literal scaffold code are in the answer.

The falsifiable expectations, against the parallel rung's actual numbers. Most confident: the sandwich should
*beat* 2.3112 — it restores the cross-talk the parallel rung gave up *and* adds output-variance control, two
moves in the quality direction with no regression I can name; if it does not, the "spend structure on
normalization" thesis is dead and I abandon block restructuring. The perplexities should follow down,
WikiText-2 below 45.98 and LAMBADA below 70.96. The one I am genuinely unsure of is whether it is the *best*
arrangement: my prior is that the output norm is insurance more than a lever at 24 layers, so I expect it to
sit *between* the parallel floor and a bare RMSNorm swap, recovering most of the gap but not all — and if it
instead clears what the bare swap would give, output-variance control is a real final-loss lever and the next
move pushes it harder. And it should be the *slowest* rung, low-20,000s, four norms per block being the most
normalization arithmetic on the ladder.
