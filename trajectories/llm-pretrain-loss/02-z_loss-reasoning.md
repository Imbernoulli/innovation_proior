Label smoothing told me something precise, and it told me in numbers. It did not just fail to help — it
landed *behind* plain cross-entropy on the metric I care about: `val_loss` 2.3377, with WikiText-2
perplexity at 47.13 and LAMBADA at 71.80, both noticeably worse than I would want, and the downstream
accuracies (arc_easy 54.04, hellaswag 33.63, piqa 63.71, winogrande 51.78) sitting in a
respectable-but-unremarkable band. Let me read those numbers with a little arithmetic before I move,
because they carry more than "it was worse." A `val_loss` of 2.3377 is a per-token cross-entropy, so the
FineWeb token-level perplexity is `exp(2.3377) = 10.36` — the model is effectively choosing among about
ten equally-likely next tokens. And the way I set that rung up, I expected light training-only smoothing
to land within a few hundredths of a nat of plain cross-entropy, above or below; it came in above,
behind, which is the sign that on a single-epoch run the half-nat of off-data bias I put into the
objective was not repaid by the regularization. Read that the right way. I trained against a softened
distribution and was graded against the true one, and the deeper lesson is structural: smoothing pulled
on the logit *gap* — the difference between the true-class logit and the rest, the part the softmax
actually sees — and the gap is the gauge-invariant part of the logits. It never touched the absolute
*level*. And on this run, in bfloat16, the level is exactly where I should expect trouble. So I am going
to stop attacking the target and go after the handle smoothing structurally cannot reach.

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
free gauge with no restoring force. Let me make the "no restoring force" quantitative rather than
rhetorical, because it is the whole reason this rung can work where smoothing could not. The gauge
direction in logit space is the all-ones vector `1`, and the force the optimizer feels along it is the
directional derivative `Σ_j ∂CE/∂l_j = Σ_j (p_j - 1[j=y]) = (Σ_j p_j) - 1 = 1 - 1 = 0`. Identically zero,
for every logit configuration — cross-entropy exerts *exactly no* force along the gauge, which is the
same shift-invariance seen through the gradient. That is why label smoothing did nothing here: smoothing
sharpens the *gaps*, it lives entirely in the gauge-invariant subspace, so it too contributes zero along
`1` and cannot supply the missing constraint on the level — the gauge is precisely the direction both are
blind to. The 2.3377 was smoothing pulling hard on a handle that was already constrained while the
genuinely free quantity drifted underneath.

Which way does the gauge actually drift in practice? I should be honest that shift-invariance alone does
not fix a direction — a truly forceless coordinate could wander either way — so the direction is an
empirical fact about these runs, not something I can derive from the loss. There is gradient flow into
the logits from many places, weight decay interacts with the tied embedding/unembedding, and with no
restoring force any push that inflates the logits is free as far as the loss is concerned; the observed
behavior is that it wanders *up*, the activations feeding the final softmax growing over the run, the
model getting more extreme in magnitude without the loss objecting. That is the slow gradient-norm
growth. And why is a drifting `log Z` dangerous *here*, in bfloat16 specifically? Because these large
numbers go straight into an exponential. bfloat16 keeps float32's eight exponent bits but only seven
mantissa bits against float32's twenty-three, so within any binade it rounds about `2^16 ≈ 65536` times
more coarsely, and because a fixed mantissa width spans each interval `[2^k, 2^{k+1})`, the *absolute*
roundoff grows with the magnitude of the number. Put a number on it: at magnitude 128, i.e. the binade
`[2^7, 2^8)`, the unit in the last place is `2^(7-7) = 2^0 = 1`, so consecutive representable bfloat16
values near 128 are spaced a full 1.0 apart. Now push a large, coarsely-rounded logit through `exp`: the
differential is `d(exp x) = exp(x)·dx`, so a small absolute error `dx` in the argument becomes a
*relative* error of size `dx` in the output; make the logits large and `dx` grows with them, so the
softmax outputs are corrupted by an amount that grows as the level drifts up. Concretely: ten logits at
128 and one at 128.5. The true softmax weight on the distinguished token is
`e^0.5/(10 + e^0.5) = 1.6487/11.6487 = 0.1416`. But 128.5 sits exactly halfway between the representable
128 and 129, and round-half-to-even sends it to 128 (whose last mantissa bit is zero), so in bfloat16 all
eleven logits are 128 and the weight becomes the uniform `1/11 = 0.0909` — a collapse from 0.1416 to
0.0909, a 36% swing manufactured entirely by roundoff, because the 0.5-nat gap that carried the
distinction fell below the resolution. The free gauge lets the logits drift large; large logits are
where the bfloat16 exponential becomes unfaithful; an unfaithful exponential corrupts the softmax and,
when it tips over, produces a spike. The slow growth and the sudden discontinuity are the same disease at
two timescales.

So I need a restoring force on the gauge, and cross-entropy cannot supply it by construction — I just
watched its gauge force come out to exactly zero. I have to add something that *does* care about the
level. Before I write a new term, though, I should ask whether something already in the frozen substrate
holds the level for me, because the cleanest fix is one I do not have to author. Weight decay is on
(`weight_decay=0.1`) and it pulls every parameter toward zero, including the unembedding matrix that
produces the logits — so does it not already discourage large logits? It does, but indirectly and with a
confound I cannot steer from the loss. Decay shrinks *weights*, not `log Z`; the logit level is a function
of the unembedding *and* the activation feeding it, and the level can drift up through growing activations
even while the unembedding norm is held down, so decay controls the wrong variable. Worse, the embedding
is tied — one matrix is both the input token lookup and the output projection — so decaying it to shrink
logits simultaneously shrinks the input representation, coupling two roles that want different scales, and
there is no per-role knob to separate them. And decay is a fixed pull toward zero-norm weights regardless
of where `log Z` actually sits, whereas what I want is a force that is *zero* when the level is healthy
and grows only when it drifts — a term that reads `log Z` itself, not the parameter norm. Weight decay is
a blunt instrument aimed a layer away from the quantity I care about; I have to add the term.

What do I want it to prefer? The logits not to drift large; equivalently `log Z` not to drift
large. The cleanest target is `log Z = 0`, i.e. `Z = 1`, i.e. `Σ_k exp(l_k) = 1` — the slice where the
raw logits behave like normalized log-probabilities. I do not want a hard projection onto that slice (a
constraint to enforce, fighting the optimizer, and one that would need a Lagrange multiplier or a
dual loop I have no room for); I want a soft nudge — a penalty minimized at `log Z = 0` and growing as
`log Z` moves away. What shape? Call `s = log Z`. A signed penalty `s` has no minimum, it would shove
`log Z → -∞`; it must be symmetric, since the representative with `Z` near 0 is no more meaningful than
the one with `Z` huge. `|s|` is symmetric and minimized at 0 but has a kink there, with gradient
`sign(s)` — a constant pull of fixed strength right up to the minimum, yanking even when `log Z` is
already tiny, exactly the fixed-strength misbehavior I would dislike in a hard `clamp` derivative. `s²`
is symmetric, minimized at 0, smooth everywhere including the minimum, convex, and — the property I want —
its gradient `2s` is *proportional to the violation*: gentle when `log Z` is near 0, stronger the further
it has drifted. That is exactly a restoring force; it leaves a well-behaved gauge alone and pulls hard
only on a runaway. So the penalty is `(log Z)²`. I will square the log-partition.

I should also justify the *quantity*, not just the shape, because the thing I literally care about is
`max_k l_k` — the single top logit that will actually saturate the bfloat16 exp — and `log Z` is only a
proxy for it. Why penalize `log Z` and not the max directly? Two reasons pushing the same way. The max is
non-smooth: `∂(max_k l_k)/∂l_j` is `1` on the single arg-max coordinate and `0` on every other, so a
max-penalty hands a gradient to exactly one logit per position and produces a kink whenever the argmax
switches — precisely the dead-coordinate, discontinuous-slope pathology I want to keep out of Adam's
per-coordinate statistics. `log Z` is the smooth softmax-weighted surrogate for the max:
`∂(log Z)/∂l_j = p_j` spreads the restoring force across coordinates in proportion to how much each
contributes to `Z`, so the largest logits are pushed hardest and nothing is ever severed. And the
log-sum-exp sandwich (which I lean on again just below) makes `log Z` a faithful stand-in for the max
anyway — they differ by at most `log V` — so holding `log Z` down holds the max down, smoothly, without my
ever having to name which coordinate is currently on top. Penalizing `log Z` is penalizing the max with a
differentiable handle. That is the quantity; `(log Z)²` is the shape.

Let me confirm the gradient actually does what I think, and reuse the gauge projection as the check. The
augmented per-position objective is `L = CE + λ·(log Z)² = (log Z - l_y) + λ·(log Z)²`. The key
derivative is `d(log Z)/d l_j = exp(l_j)/Σ_k exp(l_k) = p_j`. So `d(log Z - l_y)/d l_j = p_j - 1[j=y]`,
the familiar predicted-minus-one-hot, and `d/d l_j[λ·(log Z)²] = 2λ·log Z·p_j`. Stacked:
`dL/d l_j = (p_j - 1[j=y]) + 2λ·log Z·p_j`. Now project onto the gauge again — sum over `j` — and watch
the two terms separate cleanly: the cross-entropy part gives `Σ_j (p_j - 1[j=y]) = 0` as before, and the
penalty part gives `Σ_j 2λ·log Z·p_j = 2λ·log Z·(Σ_j p_j) = 2λ·log Z`. So the total force the optimizer
feels along the free direction is *exactly* `2λ·log Z`, supplied entirely by the penalty, zero of it from
cross-entropy. That is the whole design in one line: on the one direction cross-entropy leaves completely
unconstrained, the z-loss is the sole force, and it points back toward `log Z = 0` with a strength
proportional to how far the level has drifted. Read the per-logit term the same way: when `log Z > 0`
(the logits have drifted up, the dangerous case) `2λ·log Z·p_j` is positive on every coordinate, weighted
by `p_j`, so descent subtracts it — pushing down hardest on the logits that contribute most to `Z`,
shrinking `log Z` back toward 0; when `log Z < 0` it reverses and pushes those same high-probability
coordinates back up.

Does pinning the gauge actually bound the magnitudes that feed the exp? I want a guarantee, not a hope.
Log-sum-exp is sandwiched: `Z ≥ exp(max_k l_k)` gives `log Z ≥ max_k l_k`, and `Z ≤ V·exp(max_k l_k)`
gives `log Z ≤ max_k l_k + log V`. So `max_k l_k ≤ log Z ≤ max_k l_k + log V` — `log Z` and the largest
logit differ by at most `log V`, which for `V ≈ 50{,}257` is `log V = 10.82` nats. If the penalty holds
`log Z` near 0, the largest logit cannot become large and positive; it lives no more than about 10.8 nats
below `log Z` and never above it. That caps the top logits — the ones that dominate the exponential —
which is the difference between the exp seeing numbers near the normal log-probability scale and seeing
128. And `log Z = 0`, i.e. `Σ exp(l_k) = 1`, doubles as making the raw logits behave like honest
normalized log-probabilities rather than log-probabilities-plus-junk. The gauge-fixing and the
make-the-logits-meaningful goals are the same move.

One worry I should close before I trust this, because a restoring force toward `log Z = 0` sounds like it
could handcuff the model: can it still say "this token is almost certain" once I have pinned the level?
Pinning `log Z = 0` means `Σ_k exp(l_k) = 1`, so the raw logits *are* the log-probabilities,
`l_k = log p_k`, all of them `≤ 0`. To express `p_y = 0.95` the model sets `l_y = log 0.95 = -0.05` and
spreads the remaining `0.05` of mass over the other tokens near `log(0.05/(V-1)) ≈ log(9.95·10⁻⁷) =
-13.8`. That is a perfectly representable configuration — a spread of about `13.8` nats between the winner
and the tail — so holding `log Z` at 0 costs the model no expressive power at all; it just fixes the
additive gauge so that "confident" is realized by the tail going very negative rather than by the winner
running very positive. The penalty removes a redundant degree of freedom, not a useful one. And that
`13.8`-nat spread is exactly the confidence scale label smoothing's own optimum implied — its optimal
`p_y = 1-ε ≈ 0.95` against a tail of `ε/(V-1)` gives the same `log((1-ε)(V-1)/ε) ≈ 13.8` nats of log-odds
— which is reassuring: the level fix and the target fix agree on *how much* log-odds a healthy prediction
needs. They disagree only on whether to also let the whole logit stack float upward off that scale, and
floating upward is precisely the part that hurts in bfloat16.

This is also where I can see, cleanly, why this is the right lever and label smoothing was the wrong one
— not as a guess now but confirmed by the 2.3377. Smoothing changes the *target distribution*, fitting
the model to a softened version of the data; that lowers true likelihood (which is what I was graded on,
and it cost 2.3377 with perplexities 47.13/71.80) and acts on the gap, the gauge-invariant part with zero
gauge force. The z-loss changes neither — it leaves the target the data and touches only the free level
through `log Z`, so it sits on top of plain cross-entropy without becoming label smoothing in disguise.
Different handle, and the one a numerical drift actually has. A hard logit clamp is worse: it clamps an
already-corrupted value after the roundoff has happened and adds a kink the optimizer must route around;
the penalty acts smoothly *during* optimization to discourage the large logits from ever forming.
Gradient-norm clipping is further downstream still, reacting after the bad step is computed. The penalty
is the only one of the three that addresses the cause rather than the symptom.

Now `λ`. Two jobs fix its scale by an order-of-magnitude argument, not a guess. It must be small enough
that the penalty is a gentle regularizer and cross-entropy stays essentially maximum likelihood, or I am
back to changing what is modeled — the very thing that cost smoothing its 2.3377. And it must be large
enough that the restoring force keeps `log Z` near 0 against the drift. Here the gauge-projection result
does real work: because cross-entropy contributes *exactly zero* force along the level, I do not need
`λ` to out-muscle cross-entropy on that direction — there is nothing to out-muscle, the penalty is the
only force there, so even a tiny coefficient controls the gauge as long as it beats the incidental drift.
Cross-entropy per token is order one-to-ten nats early, settling to a couple (the 2.3377 I just measured
is `exp` of ten-way perplexity); a healthy `log Z` is order one, so `(log Z)²` is order one. At
`λ = 1e-4` the penalty contributes order `1e-4` against a CE of order 2.3 — a ratio around `4e-5`,
utterly invisible to the modeling objective — yet its gauge force `2λ·log Z = 2e-4·log Z` applies a
steady pull that accumulates over thousands of steps to hold the level down, and it is uncontested there.
Much larger and it would start crushing `log Z` and competing with CE for the gap-carrying logits; much
smaller and it loses to the drift. `1e-4` is the canonical large-vocabulary coefficient, and unlike
smoothing's `ε` it needs no training-vs-eval split: it does not distort the target, so I can leave it on
through the whole run and the evaluated cross-entropy is unaffected by it — the penalty term is not even
added into the reported number, and the model it produces is graded on honest likelihood.

Two implementation details or the whole thing is subtly wrong. First, reduction: cross-entropy ignores
the `-1` packed-boundary positions and averages over valid ones; the penalty has to be averaged over
*exactly* those same positions, both because the ignored positions' logits are untrained garbage whose
`log Z` is meaningless, and because a different denominator would change the effective `λ` away from the
clean ratio I just reasoned about — if I divided by all `B·T` positions instead of the valid ones, the
realized coefficient would be scaled by the valid fraction and my order-of-magnitude argument would be
off by exactly that factor. So mask to `targets != -1` and take the mean of `(log Z)²` over those.
Second, compute `log Z` with the max-subtracting stabilized `torch.logsumexp` — I would be embarrassed to
introduce an overflow while fixing a numerical drift — which subtracts `max_k l_k` before exponentiating,
so it never exponentiates a positive argument and is exact. The whole edit is then plain cross-entropy
(the library handles the ignore index and averaging) plus `1e-4` times the masked-mean squared
log-partition. The full scaffold function is in the answer.

So the delta from step 1: where smoothing softened the target and pulled on the gauge-invariant gap that
carries zero level force, I leave the target the data and add the one term that touches the free level — a
small squared-log-partition penalty whose proportional restoring force `2λ·log Z` is the *only* force on
the gauge, holds `log Z` near 0, caps the top logits via the log-sum-exp sandwich to within ~11 nats of
`log Z`, and keeps the bfloat16 exp in its faithful regime. Reading smoothing's shape, here is what I
expect, falsifiably. The primary `val_loss` should move *below* smoothing's 2.3377 and below plain
cross-entropy — because, unlike smoothing, this neither distorts the evaluated objective nor leaves the
dangerous magnitude free; I would expect a recovery of a few hundredths of a nat off smoothing's mark, and
by the same `perplexity = exp(loss)` map, that recovery should drag WikiText-2 and LAMBADA perplexity down
by a few percent each off smoothing's 47.13 / 71.80. The
downstream accuracies (arc_easy, hellaswag, piqa, winogrande) should hold or nudge up, since a
numerically healthier softmax is strictly better-behaved and I am no longer paying smoothing's off-data
bias — though at these eval-set sizes single-seed accuracy moves of a point or so are within noise, so I
will read them as a band rather than a ranking. If `val_loss` improves but the top logits are *still* not
tightly bounded — if a single excited coordinate can spike even with the level held on *average*, since
`2λ·log Z` is a force on the batch-mean `log Z` and not a per-position ceiling — the diagnosis for the
next rung is already there: a soft average penalty nudges the level but guarantees no per-logit bound, so
the next move is to stop nudging the level and structurally bound the logit *values* themselves.
