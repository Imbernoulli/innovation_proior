The sandwich rung came back and settled the question I was unsure of, in the direction I half-expected but by
a margin small enough to be its own verdict. Validation loss 2.3104 against the parallel rung's 2.3112: it did
beat the parallel floor, so the first part of the thesis held — restore the cross-talk, control the branch
outputs, get back some quality. But it beat it by *0.0008* — eight ten-thousandths of a nat — and paid with the
slowest run on the board, 21,661s training against the parallel rung's 19,747s. Worse, the held-out witnesses
went the *wrong* way: both perplexities rose and PIQA dropped. The headline improved by a whisper while
generalization got worse — the signature of a change that is *insurance, not improvement*. Exactly the
soft-prior outcome I flagged: at 24 layers the output norm is a stabilizer the model did not need, and I paid
the most wall-clock on the ladder for it.

The *sign* pattern across the row is the whole diagnosis. Headline `2.3112 → 2.3104` is `0.0008` nats,
`0.035%`. The held-out witnesses moved the *other* way and by an order of magnitude more: WikiText-2
`45.98 → 46.8` is `+1.78%`, LAMBADA `70.96 → 72.08` is `+1.58%`. Same experiment, opposite signs, a fifty-fold
difference in size — not a better language model but a differently-regularized one whose training-distribution
loss is a whisper lower and whose generalization is measurably worse. Downstream corroborates on the columns
that carry signal: ARC-Easy flat (`54.76 → 54.76`), HellaSwag and WinoGrande moving inside their near-chance
noise, and the one downstream number with any size moving the wrong way — PIQA `64.42 → 62.46`, `−1.96`, a
regression on a task held out of selection. Every witness that moved appreciably moved against the sandwich. And
the cost: training `19747 → 21661`s, `+1914`s, `+9.7%`, `1.64 → 1.80` s/iter; eval `406 → 586`s, two more norms
per block showing up even in the short pass.

So I have two structural experiments and their verdicts. The parallel rung *removed* normalization and
structure and came in worst on val_loss, losing cross-talk at small scale. The sandwich *added* normalization
and structure and came in marginally better on val_loss but worse on perplexity and slowest in time. Read
together they bracket the answer: more block restructuring in either direction is not buying language-modeling
quality at this scale — the parallel simplification cost cross-talk, the sandwich elaboration cost
generalization and speed, and neither moved the metric that matters more than a whisker. I have been spending my
edit budget on the *wiring*, and the wiring is not where the quality is. The one change unambiguously safe
across both rungs was the normalization *rule* — RMSNorm trained stably both times and I never had a reason to
doubt it. So the move the data points to is to stop restructuring, keep the cheaper well-behaved RMSNorm, and
return to the plain sequential pre-norm block the default used — the minimal change from the default fill.

Why the plain block should be the strongest of the three is relative to the two failures I measured. It keeps
everything the substrate's `1/√(2·n_layer)` residual init, learning rate, and schedule were chosen for — two
residual writes per block, one pre-norm per sublayer, the gradient flowing straight down the identity path — and
changes only the normalization rule, LayerNorm for RMSNorm, which I validated twice as harmless — drop the
mean and the bias, keep `a/RMS(a)·γ` and its self-regulating quadratic gradient. One detail I lean on here:
the `1/n` inside the root, rather than a plain L2 `‖a‖`, normalizes per-coordinate, so the scheme behaves
consistently across the 1024-d feature vectors that an L2 projection onto a unit sphere would not.

Reconciled with the two failures: against the parallel rung, the plain block restores sequential ordering, so
the MLP reads the post-attention residual and the intra-block cross-talk that cost 2.3112 is back — recovered by
doing nothing more exotic than not parallelizing. Against the sandwich, it has *one* norm per sublayer, not two,
so it drops the output-variance constraint I just watched over-constrain the model and push the perplexities up;
each sublayer writes its natural contribution into the residual, and at 24 layers — deep enough to train cleanly
under pre-norm but not so deep that un-normalized residual growth hurts final loss — that freedom is worth more
than the sandwich's variance insurance. So the plain block sits at the sweet spot the two restructured rungs
straddled — the parallel rung's cross-talk back without its simplification, the default's clean pre-norm
gradient without the sandwich's over-constraint — and it is the one arrangement that matches the substrate on
every axis, so unlike the restructured rungs it incurs no init-vs-wiring mismatch: every block writes twice at
`1/√48`, residual RMS starting near `√2`, the near-doubling the init was designed for — not the sandwich's
`√48 ≈ 6.9`, where the scale-invariant `LN_post` divided the `1/√48` straight back out.

None of this is a capacity change — the plain block carries `49,152` gains (no biases), the sandwich `98,304`,
the parallel `24,576`, spanning `0.014%`/`0.028%`/`0.007%` of the model, capacity-invisible. So the entire
measured `val_loss` spread across the two restructured rungs, `0.0008` nats, was produced at effectively fixed
capacity: a pure placement-and-wiring effect. Whatever the plain block lands at, the difference is attributable
to *where the norms sit and how the sublayers are ordered*, nothing else.

So I expect not a dramatic drop but the plain block to claim the *best* val_loss on the ladder by a margin
similar to the gaps already seen, a few thousandths of a nat — and, crucially, to do it *while the perplexities
go the right way*, because the thing that made the sandwich's perplexity rise is exactly what the plain block
removes. `LN_post` re-pinned every branch contribution to a magnitude set by its learned gain, flattening the
model's freedom to write a large contribution from one block and a small one from another; on the training
distribution the optimizer could route around the constraint (why the in-distribution cost never showed and
`val_loss` even dipped), but on held-out text the lost degree of freedom is not recoverable at test time, and
that is what the `+1.78%`/`+1.58%` perplexity rise measures. The plain block hands that freedom back — each
sublayer writes its natural, un-re-pinned contribution, bounded only by the substrate's `1/√48` init. So the
sharp prediction is that removing exactly the operation that flattened contribution magnitudes should *reverse*
the sandwich's regression, not merely stop it: WikiText-2 back below 46.8, LAMBADA below 72.08, ideally toward
the parallel rung's 45.98 and 70.96. That perplexity reversal is the claim I care about most — if the plain
block lowers val_loss but the perplexities do not improve, the val_loss ordering is noise at this scale and the
honest conclusion is that norm placement does not matter here.

I touch no `CONFIG_OVERRIDES`: the whole point of returning to the plain block is that it is the arrangement the
learning rate, the `0.04 × 12030 ≈ 481`-iteration warmup, and the cosine decay to `min_lr` were tuned for, so
changing any of them would undo the alignment I am deliberately restoring. So the strongest rung is the *least*
elaborate fill: keep rung 1's RMSNorm class verbatim, and replace `Block` with the plain sequential pre-norm
block — `ln_1`, attention, residual add; `ln_2`, MLP, residual add — the default block with LayerNorm swapped
for RMSNorm and nothing else. The distilled module and the literal scaffold code are in the answer.

Two more predictions I can finally pin down quantitatively. On wall-clock the sandwich handed me a second data
point, so I can price a norm rather than just order the rungs: `1` norm per block at `19747`s, `4` at `21661`s,
so `1914`s over three extra norms is `638`s per norm. Extrapolating to the plain block's `2` norms gives
`≈ 20385`s — but the `1`-norm point is the *parallel* block, which also had a shortened critical path (one
input, no write-ordering wait), so it runs a touch below a `1`-norm *sequential* block and my slope under-counts;
I nudge the plain-block estimate into the low-20,000s, between the two prior rungs. On downstream, ARC-Easy sat
flat at `54.76` across both prior rungs and I expect no move; HellaSwag and WinoGrande sit within a tenth of
their near-chance floors and carry no signal; PIQA is the one column with headroom and a measured swing
(`64.42 → 62.46` under the sandwich), so I predict it recovers most of the way toward `64.42` — and if it does
not while `val_loss` improves, that is another way the over-constraint story fails. So the ladder ends where it
is simplest: the cheaper RMSNorm rule, safe every time, in the plain sequential pre-norm block the substrate was
built for, the two restructuring experiments having shown in measured numbers that spending the edit budget on
wiring in either direction buys nothing at 355M the plain block does not already have.
