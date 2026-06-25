The objective is the whole task here — everything else is frozen — so the floor to start from is the
question of what is even wrong with the loss I already have, plain next-token cross-entropy, and which
of the three obvious handles to pull first. Let me write the loss down and actually stare at it instead
of reciting it. For one position with logits `z` over the vocabulary and target index `y`, the loss is
`-log softmax(z)_y = -z_y + log Σ_j exp(z_j)`. This is maximum likelihood and it is correct; that is not
in dispute. What bugs me is where its optimum sits. To drive this to zero I need `softmax(z)_y → 1`, and
that happens only as the gap `z_y - max_{j≠y} z_j` runs off to `+∞`. There is no finite `z` that
minimizes this loss. On data that is even close to separable at the margin — and token-level language
data effectively is, because for most contexts there genuinely is one overwhelmingly likely next token —
the loss is, quite literally, a standing instruction to grow the correct logit's lead without bound. The
gradient `∂ℓ/∂z_k = p_k - 1[k=y]` is bounded in `[-1, 1]`, so each step is gentle, but the target it is
chasing is at infinity, and over thirteen thousand iterations a bounded gradient pointed at infinity
walks the logits steadily upward.

So what actually goes wrong from that runaway, and why is it worth fixing before I even reach for the
sharper instabilities? Two things, and they are the same thing seen from two sides. The model gets more
and more confident on its training targets — it learns to slam probability mass onto the observed next
token of every context, which is the textbook signature of over-confidence and, on the held-out
distribution, of overfitting: fitting the training labels says nothing about generalizing to held-out
ones. And the unbounded growth of the largest-minus-rest logit gap, against a bounded gradient, makes
the model rigid — once a huge gap is established the model is slow to revise it, because a bounded
gradient can only chip away at a gap that the loss keeps trying to make infinite. The unifying word for
both is over-confidence. This is exactly the thing the cheapest loss-layer modification should be built to
fix, and the diagnosis is clean: hard one-hot targets drive `z_y ≫ z_k`, which both overfits and reduces
adaptability, so the fix is to stop demanding probability one.

There are three distinct places I could intervene, and the reason I take this one *first* is that it is
the cheapest, the most thoroughly understood, and the one whose failure mode will point cleanly at the
next rung. I could attack the absolute logit *level* — add a penalty on the log-partition so the
otherwise-free overall magnitude stops drifting. I could attack the logit *values* themselves —
structurally squash them through a bounded map before the softmax. Or I could attack the *target*: stop
asking for probability one in the first place. The target attack is the most surgical statement of "the
optimum is at infinity, so move the optimum to a finite place," it is a one-line change to the loss, and
it is the right thing to establish as the baseline because if even the gentlest, best-understood
over-confidence fix already buys something, then the harder numerical interventions have a real target
to beat; if it does not, I have learned that this short run is not over-confidence-limited and the next
rung had better attack a different handle.

If the problem is that the target lives at infinity, the cure suggests itself: do not put the target at
infinity. Give every token a small floor of target probability so the correct logit has no incentive to
escape to `+∞`. Take the one-hot and bleed a little mass `ε` onto a fixed distribution `u` over the
vocabulary, `q'(k) = (1-ε)·δ_{k,y} + ε·u(k)`. The natural `u`, absent any prior knowledge, is uniform,
`u(k) = 1/V`, so `q'(k) = (1-ε)·δ_{k,y} + ε/V`. Now check that this actually kills the runaway. Every
entry of `q'` is at least `ε/V > 0`. If `z_y` tried to run off to `+∞`, then `p_y → 1` and `p_k → 0`
for `k ≠ y`, and the cross-entropy `-Σ_k q'(k) log p_k` would blow up on those wrong-class terms,
because `q'(k) = ε/V` is positive but `log p_k → -∞`. So an infinite logit gap is now infinitely
*expensive*, not free. The target no longer sits at infinity; it sits at a finite, `ε`-controlled
configuration of the logits, with optimal predicted probabilities `1-ε` on the true token and `ε/(V-1)`
on each other. That is label smoothing, and it is almost nothing to implement.

Let me rewrite the loss to see its structure, because the structure is what tells me what I am really
doing. Cross-entropy is linear in the target, so
`H(q', p) = -Σ_k q'(k) log p_k = (1-ε)·(-Σ_k δ_{k,y} log p_k) + ε·(-Σ_k u(k) log p_k) = (1-ε)·H(q, p) + ε·H(u, p)`.
So smoothing is exactly the ordinary hard-label cross-entropy, downweighted by `(1-ε)`, plus an
`ε`-weighted term `H(u, p)` that pulls the prediction toward the prior `u`. And `H(u, p) = D_KL(u‖p) + H(u)`;
`H(u)` is a constant, so the second term is, up to a constant, a penalty on how far the prediction `p`
has drifted from uniform, with relative weight `ε/(1-ε)`. It is a regularizer that says "stay a bit
humble, do not get too far from the prior." With `u` uniform, `H(u, p) = Σ_k (1/V)(-log p_k) = mean_k(-log p_k)`,
so I never have to materialize the smoothed target vector — the second term is just `ε` times the mean
of the negative log-probabilities over the vocabulary.

Now the part that is specific to *this* task and is exactly where the harness diverges from the generic
recipe, and I have to get it right or I am cheating. The validation metric is the honest modeling
cross-entropy on FineWeb — plain `-log p_y`, no smoothing. Label smoothing deliberately fits the model
to a *softened* distribution rather than the data, so a model trained to put `1-ε` on the true token and
`ε/(V-1)` on the rest is a worse density estimator under the true one-hot likelihood than a model trained
on the data directly; that is a known and accepted trade in the original setting, where smoothing buys
calibration and downstream gains at the cost of raw likelihood. But here I am *graded* on that raw
likelihood. If I left the smoothing on during evaluation, the reported number would be the cross-entropy
against the softened target, not against the data, and that would be exactly the "do not lower the
reported loss by distorting the distribution" violation the contract forbids. So the smoothing has to be
applied **only during training**: when gradients are enabled I smooth, and under the no-grad evaluation
pass I fall back to standard cross-entropy, so `val_loss` is computed against the true targets and stays
comparable across every method on the ladder. PyTorch's `F.cross_entropy` exposes `label_smoothing`
directly, so the whole edit is one call with the smoothing coefficient gated on `torch.is_grad_enabled()`.
The full scaffold function is in the answer.

The second divergence from the canonical recipe is the coefficient. The original Inception/Transformer
setting uses `ε = 0.1`. This is a *short* run — thirteen thousand iterations, a tiny fraction of a full
GPT-2 schedule — on a large vocabulary (~50k), and on a short run the model never gets far enough into
the over-confident regime for a heavy `0.1` regularizer to pay for itself; an aggressive smoothing
mostly just biases the training objective away from the data without the long-run overfitting it would
otherwise prevent. A lighter `ε = 0.05` is the task-local choice: enough of a floor to take the sharpest
edge off the runaway, light enough that the (1-ε) reweighting of the true-likelihood term is barely
perturbed. So this rung is not canonical label smoothing — it is `ε = 0.05`, training-only, via the
library's `label_smoothing` argument with the `-1` ignore index the harness uses for packed boundaries.

Now reason about what this floor must do, because that is the entire point of running it first. Smoothing
attacks the target, and through the target it caps the *gap* between the true-class logit and the rest —
the difference the softmax actually sees. What it pointedly does *not* touch is the absolute *level* of
the logits: a model can sit at logits `(1000, 990, 990, …)` and have exactly the smoothed-optimal gaps
while being numerically enormous, because cross-entropy and its smoothed cousin are both invariant to a
uniform shift of all logits. So smoothing is a statement about differences, and it leaves the overall
magnitude — the gauge — completely free. That matters because the run trains in bfloat16, where large
logit magnitudes are where the exponential in the softmax becomes unfaithful; smoothing does nothing
about that. It also leaves a real question about whether trading away true-likelihood for a milder gap is
even the right trade when the *evaluation* is true likelihood: I am training against a slightly different
distribution than I am graded on, and only the regularization benefit, if any, can repay that gap.

So here is what I expect, stated against numbers I can be wrong about. On the primary `val_loss`, light
training-only smoothing should land *close* to plain cross-entropy — within a few hundredths of a nat —
and it is genuinely unclear whether it lands above or below, because the regularization benefit on a
short run is small and it is fighting the fact that I have biased training away from the evaluated
objective. On the downstream accuracies (ARC-Easy, HellaSwag, PIQA, WinoGrande), where a better-calibrated
next-token distribution can help even when likelihood does not, I would expect smoothing to be at worst
neutral. If this rung comes in *behind* a method that attacks the logit level or the logit values
directly, the diagnosis is already written: smoothing pulled on the gap, the gauge-invariant handle,
and left untouched the absolute magnitude that the bfloat16 softmax actually cares about — which says
the next rung should stop attacking the target and start attacking the level. This is the weakest rung
on the ladder by design: it is the most conservative, target-side over-confidence fix, and whatever it
leaves on the table is precisely what the level-side and value-side interventions are for.
