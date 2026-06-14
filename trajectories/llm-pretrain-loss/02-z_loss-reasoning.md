Label smoothing told me something precise, and it told me in numbers. It did not just fail to help — it
landed *behind* plain cross-entropy on the metric I care about: `val_loss` 2.3377 against the default's
neighborhood, with WikiText-2 perplexity at 47.13 and LAMBADA at 71.80, both noticeably worse than I
would want, and the downstream accuracies (arc_easy 54.04, hellaswag 33.63, piqa 63.71, winogrande 51.78)
sitting in a respectable-but-unremarkable band. Read that the right way. I trained against a softened
distribution and was graded against the true one, and on a thirteen-thousand-iteration run the
regularization benefit was too small to repay the bias I introduced into the objective. But the deeper
lesson is structural, and it is the one that points at the next move: smoothing pulled on the logit
*gap* — the difference between the true-class logit and the rest, the part the softmax actually sees —
and the gap is the gauge-invariant part of the logits. It never touched the absolute *level*. And on
this run, in bfloat16, the level is exactly where I should expect trouble. So I am going to stop
attacking the target and go after the handle smoothing structurally cannot reach.

Let me start from what actually goes wrong at the level, because the symptom is concrete and it is not
the same symptom smoothing was built for. I am pretraining a big decoder-only Transformer in bfloat16,
and runs at this scale are prone to a particular failure: the loss sits on its nice downward curve and
then, with no warning, jumps — a spike — and the gradient norm tends to drift upward over the run, with
spike frequency rising as it grows. Gradient clipping is on (`grad_clip=1.0`) and the spikes come anyway,
which already tells me something: whatever is going wrong has produced a state from which even a
norm-limited step is bad, so clipping is treating a symptom downstream of the cause. I have two coupled
things — a slow growth and sudden discontinuities — and I want to find what is drifting and stop it at
the source, not keep restarting from checkpoints and skipping batches.

Where would a drift like that even live? Most of the forward pass has normalization stapled to it — the
attention, the MLPs, the layer norms — that keeps its scales in check. The one place in the whole
forward pass where a wide vector of unbounded real numbers gets pushed through an exponential with
nothing normalizing it first is the final softmax over the vocabulary. That is the suspect, and it is
the *same* layer label smoothing operated on, which is the point: I am at the right boundary, I just have
to pull the right handle this time. So let me write down what the objective there really is, in a form I
can manipulate. For one position, logits `l = (l_1, …, l_V)`, predicted distribution `p_j = exp(l_j)/Z`
with normalizer `Z = Σ_k exp(l_k)`, and cross-entropy against the true next token `y` is
`CE = -log p_y = -(l_y - log Z) = log Z - l_y`, where `log Z = log Σ_k exp(l_k)` is the log-partition.
So the loss decomposes, exactly, into two pieces pulling opposite ways: raise the true-token logit `l_y`,
lower the log-partition `log Z`.

Now the crux, and it is precisely the thing label smoothing could not see. Cross-entropy depends on the
logits *only through their differences*. Add a constant `c` to every logit, `l → l + c·1`: the softmax
does not move at all, `exp(l_j + c)/Σ_k exp(l_k + c) = e^c exp(l_j)/(e^c Σ_k exp(l_k)) = p_j`, the `e^c`
cancels, so `CE` is unchanged. But look at the two pieces individually: `l_y → l_y + c`, and `Z → e^c Z`
so `log Z → log Z + c`; both terms moved by `+c` and inside `CE = log Z - l_y` the two `c`'s cancel.
Cross-entropy is *exactly* invariant to a uniform shift of all logits. There is a whole one-parameter
family of logit vectors — `l + c·1` for every real `c` — that produce identical predictions and identical
loss, and along that family `log Z` ranges over the entire real line while the loss the optimizer sees is
constant. Cross-entropy pins down the gaps and says nothing whatsoever about the level. The level is a
free gauge with no restoring force. And this is exactly why label smoothing did nothing here: smoothing
sharpens the *gaps*, it lives entirely in the gauge-invariant subspace, so it cannot supply the missing
constraint on the level — the gauge is precisely the direction smoothing is blind to. The 2.3377 was
smoothing pulling hard on a handle that was already constrained while the genuinely free quantity drifted
underneath.

Which way does the gauge actually drift in practice? Not down, and not randomly. There is gradient flow
into the logits from many places, weight decay interacts with the tied embedding/unembedding, and with no
restoring force any push that inflates the logits is free as far as the loss is concerned — so the level
wanders, and the documented behavior is that it wanders *up*: the activations feeding the final softmax
grow over the run, the model getting more extreme in magnitude without the loss objecting. That is the
slow gradient-norm growth. And why is a drifting `log Z` dangerous *here*, in bfloat16 specifically?
Because these large numbers go straight into an exponential. bfloat16 keeps float32's eight exponent bits
but only seven mantissa bits against float32's twenty-three, so within any binade it rounds about
`2^16 ≈ 65536` times more coarsely, and because a fixed mantissa width spans each interval
`[2^k, 2^{k+1})`, the *absolute* roundoff grows with the magnitude of the number. Now push a large,
coarsely-rounded logit through `exp`: the differential is `d(exp x) = exp(x)·dx`, so a small absolute
error `dx` in the argument becomes a *relative* error of size `dx` in the output. Make the logits large
and `dx` grows with them, so the softmax outputs are corrupted by an amount that grows as the level
drifts up. Concretely: ten logits at 128 and one at 128.5 in bfloat16 — the 0.5 gap is at the rounding
threshold for that magnitude and can round away, collapsing the softmax weight on the distinguished
token from about 0.142 to about 0.091, a 36% swing manufactured entirely by roundoff. The free gauge lets
the logits drift large; large logits are where the bfloat16 exponential becomes unfaithful; an unfaithful
exponential corrupts the softmax and, when it tips over, produces a spike. The slow growth and the sudden
discontinuity are the same disease at two timescales.

So I need a restoring force on the gauge, and cross-entropy cannot supply it by construction. I have to
add something that *does* care about the level. What do I want it to prefer? The logits not to drift
large; equivalently `log Z` not to drift large. The cleanest target is `log Z = 0`, i.e. `Z = 1`,
i.e. `Σ_k exp(l_k) = 1` — the slice where the raw logits behave like normalized log-probabilities. I do
not want a hard projection onto that slice (a constraint to enforce, fighting the optimizer); I want a
soft nudge — a penalty minimized at `log Z = 0` and growing as `log Z` moves away. What shape? Call
`s = log Z`. A signed penalty `s` has no minimum, it would shove `log Z → -∞`; it must be symmetric,
since the representative with `Z` near 0 is no more meaningful than the one with `Z` huge. `|s|` is
symmetric and minimized at 0 but has a kink there, with gradient `sign(s)` — a constant pull of fixed
strength right up to the minimum, yanking even when `log Z` is already tiny. `s²` is symmetric, minimized
at 0, smooth everywhere including the minimum, convex, and — the property I want — its gradient `2s` is
*proportional to the violation*: gentle when `log Z` is near 0, stronger the further it has drifted. That
is exactly a restoring force; it leaves a well-behaved gauge alone and pulls hard only on a runaway. So
the penalty is `(log Z)²`. I will square the log-partition.

Let me confirm the gradient actually does what I think. The augmented per-position objective is
`L = CE + λ·(log Z)² = (log Z - l_y) + λ·(log Z)²`. The key derivative is
`d(log Z)/d l_j = exp(l_j)/Σ_k exp(l_k) = p_j`. So `d(log Z - l_y)/d l_j = p_j - 1[j=y]`, the familiar
predicted-minus-one-hot, and `d/d l_j[λ·(log Z)²] = 2λ·log Z·p_j`. Stacked:
`dL/d l_j = (p_j - 1[j=y]) + 2λ·log Z·p_j`. Read the new term: it is a probability-weighted pull on
every logit. When `log Z > 0` (the logits have drifted up, the dangerous case) it is positive on every
coordinate, weighted by `p_j`, so descent subtracts it — pushing down hardest on the logits that
contribute most to `Z`, shrinking `log Z` back toward 0. When `log Z < 0` it reverses and pushes those
same high-probability coordinates back up. It is a restoring force on the gauge whose strength
`2λ·log Z` is proportional to the drift, and with `λ` tiny it fixes the level without replacing
maximum likelihood as the main force on the probability gaps.

Does pinning the gauge actually bound the magnitudes that feed the exp? I want a guarantee. Log-sum-exp
is sandwiched: `Z ≥ exp(max_k l_k)` gives `log Z ≥ max_k l_k`, and `Z ≤ V·exp(max_k l_k)` gives
`log Z ≤ max_k l_k + log V`. So `max_k l_k ≤ log Z ≤ max_k l_k + log V` — `log Z` and the largest logit
differ by at most `log V`. If the penalty holds `log Z` near 0, the largest logit cannot become large and
positive; for a 50k vocabulary it lives no more than about 11 nats below `log Z` and never above it. That
caps the top logits — the ones that dominate the exponential — which is the difference between the exp
seeing numbers near the normal log-probability scale and seeing 128. And `log Z = 0`, i.e.
`Σ exp(l_k) = 1`, doubles as making the raw logits behave like honest normalized log-probabilities rather
than log-probabilities-plus-junk. The gauge-fixing and the make-the-logits-meaningful goals are the same
move.

This is also where I can see, cleanly, why this is the right lever and label smoothing was the wrong one
— not as a guess now but confirmed by the 2.3377. Smoothing changes the *target distribution*, fitting
the model to a softened version of the data; that lowers true likelihood (which is what I was graded on)
and acts on the gap, the gauge-invariant part. The z-loss changes neither — it leaves the target the
data and touches only the free level through `log Z`, so it sits on top of plain cross-entropy without
becoming label smoothing in disguise. Different handle, and the one a numerical drift actually has. A
hard logit clamp is worse: it clamps an already-corrupted value after the roundoff has happened and adds
a kink the optimizer must route around; the penalty acts smoothly *during* optimization to discourage the
large logits from ever forming. Gradient-norm clipping is further downstream still, reacting after the
bad step is computed. The penalty is the only one that addresses the cause.

Now `λ`. Two jobs fix its scale by an order-of-magnitude argument, not a guess. It must be small enough
that the penalty is a gentle regularizer and cross-entropy stays essentially maximum likelihood, or I am
back to changing what is modeled — the very thing that cost smoothing its 2.3377. And it must be large
enough that the restoring force keeps `log Z` near 0 against the drift. Cross-entropy per token is order
one-to-ten nats early, settling to a few; a healthy `log Z` is order one, so `(log Z)²` is order one. At
`λ = 1e-4` the penalty contributes order `1e-4` against a CE of order one — about a ten-thousandth of the
scale — yet its gradient `2λ·log Z = 2e-4·log Z` applies a steady downward pull that accumulates over
many steps to hold the level down. Invisible to the modeling objective, persistent on the gauge. Much
larger and it would start crushing `log Z` and competing with CE for the logits; much smaller and it
loses to the drift. `1e-4` is the canonical large-vocabulary coefficient, and unlike smoothing's `ε` it
needs no training-vs-eval split: it does not distort the target, so I can leave it on through the whole
run and the evaluated cross-entropy is unaffected by it.

Two implementation details or the whole thing is subtly wrong. First, reduction: cross-entropy ignores
the `-1` packed-boundary positions and averages over valid ones; the penalty has to be averaged over
*exactly* those same positions, both because the ignored positions' logits are untrained garbage whose
`log Z` is meaningless, and because a different denominator would change the effective `λ` away from the
clean ratio I just reasoned about. So mask to `targets != -1` and take the mean of `(log Z)²` over those.
Second, compute `log Z` with the max-subtracting stabilized `torch.logsumexp` — I would be embarrassed to
introduce an overflow while fixing a numerical drift — which is exact and never exponentiates a positive
argument. The whole edit is then plain cross-entropy (the library handles the ignore index and averaging)
plus `1e-4` times the masked-mean squared log-partition. The full scaffold function is in the answer.

So the delta from step 1: where smoothing softened the target and pulled on the gauge-invariant gap, I
leave the target the data and add the one term that touches the free level — a small squared-log-partition
penalty whose proportional restoring force holds `log Z` near 0, caps the top logits via the log-sum-exp
sandwich, and keeps the bfloat16 exp in its faithful regime. Reading smoothing's shape, here is what I
expect, falsifiably. The primary `val_loss` should move *below* smoothing's 2.3377 and below plain
cross-entropy — because, unlike smoothing, this neither distorts the evaluated objective nor leaves the
dangerous magnitude free; I would expect it back down near 2.29, with WikiText-2 and LAMBADA perplexity
recovering from smoothing's 47/72 toward the low-44 / high-60s. The downstream accuracies should hold or
improve, since a numerically healthier softmax is strictly better-behaved. If `val_loss` improves but the
top logits are *still* not tightly bounded — if a single excited coordinate can spike even with the level
held on average — the diagnosis for the next rung is already there: a soft average penalty nudges the
level but guarantees no per-logit bound, so the next move is to stop nudging the level and structurally
bound the logit *values* themselves.
