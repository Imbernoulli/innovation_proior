Label smoothing told me something precise, and in numbers: it landed *behind* plain cross-entropy on the
metric I care about, `val_loss` 2.3377, WikiText-2 perplexity 47.13 and LAMBADA 71.80, with downstream
accuracies (arc_easy 54.04, hellaswag 33.63, piqa 63.71, winogrande 51.78) in a respectable but
unremarkable band. A `val_loss` of 2.3377 is `exp(2.3377) = 10.36` in token perplexity — the model
effectively choosing among about ten equally-likely next tokens. I expected smoothing to land within a few
hundredths of a nat of plain cross-entropy, either side; it came in behind, which is the sign that on a
single-epoch run the half-nat of off-data bias was not repaid by the regularization. That confirms what I
flagged going in: smoothing pulled on the gauge-invariant *gap* and never touched the absolute *level* —
and in bfloat16 the level is exactly where the trouble is. So I stop attacking the target and go after the
handle smoothing structurally cannot reach.

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

Now the crux. Cross-entropy depends on the logits only through their differences: a uniform shift
`l → l + c·1` moves `l_y → l_y + c` and `log Z → log Z + c`, and inside `CE = log Z - l_y` the two cancel,
so a whole one-parameter family `l + c·1` produces identical loss while `log Z` ranges over the entire
real line. The level is a free gauge with no restoring force. Make that quantitative, because it is the
whole reason this rung can work where smoothing could not: the force along the gauge direction `1` is the
directional derivative `Σ_j ∂CE/∂l_j = Σ_j (p_j - 1[j=y]) = 1 - 1 = 0`, identically zero for every logit
configuration. Smoothing lives entirely in the gauge-invariant gap subspace, so it too contributes zero
along `1` and cannot supply the missing level constraint — the 2.3377 was smoothing pulling on a handle
already constrained while the genuinely free quantity drifted underneath.

Which way does the gauge actually drift in practice? I should be honest that shift-invariance alone does
not fix a direction — a truly forceless coordinate could wander either way — so the direction is an
empirical fact about these runs, not something I can derive from the loss. There is gradient flow into
the logits from many places, weight decay interacts with the tied embedding/unembedding, and with no
restoring force any push that inflates the logits is free as far as the loss is concerned; the observed
behavior is that it wanders *up*, the activations feeding the final softmax growing over the run, the
model getting more extreme in magnitude without the loss objecting. That is the slow gradient-norm
growth. And why is a drifting `log Z` dangerous *here*, in bfloat16 specifically? Because these large
numbers go straight into an exponential. bfloat16 keeps float32's exponent range but only seven mantissa
bits, and because a fixed mantissa width spans each binade `[2^k, 2^{k+1})`, the *absolute* roundoff grows
with the magnitude: at magnitude 128 the spacing between representable values is a full `1.0`. Push a
coarsely-rounded logit through `exp` and `d(exp x) = exp(x)·dx` turns that absolute error into a *relative*
error on the softmax weight that grows as the level drifts up. Concretely: ten logits at 128 and one at
128.5. The true softmax weight on the distinguished token is `e^0.5/(10 + e^0.5) = 0.1416`, but 128.5 sits
halfway between representable 128 and 129 and rounds to 128, so in bfloat16 all eleven logits are equal and
the weight collapses to `1/11 = 0.0909` — a 36% swing manufactured entirely by roundoff, the 0.5-nat gap
having fallen below the resolution. The free gauge lets the logits drift large, large logits make the
exponential unfaithful, and an unfaithful exponential corrupts the softmax and eventually spikes. The slow
growth and the sudden discontinuity are the same disease at two timescales.

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

The gradient of the augmented objective `L = CE + λ·(log Z)²` confirms the design. Using
`d(log Z)/d l_j = p_j`, the per-logit gradient is `dL/d l_j = (p_j - 1[j=y]) + 2λ·log Z·p_j`. Projecting
onto the gauge — sum over `j` — the two terms separate: cross-entropy gives `Σ_j (p_j - 1[j=y]) = 0` as
before, and the penalty gives `Σ_j 2λ·log Z·p_j = 2λ·log Z`. So the force along the one direction
cross-entropy leaves unconstrained is *exactly* `2λ·log Z`, supplied entirely by the penalty, pointing
back toward `log Z = 0` with a strength proportional to the drift. Per-coordinate, when `log Z > 0` the
term `2λ·log Z·p_j` is positive everywhere, so descent pushes down hardest on the logits contributing most
to `Z`; when `log Z < 0` it reverses.

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
-13.8`. That is a perfectly representable configuration — a `13.8`-nat spread between winner and tail — so holding
`log Z` at 0 costs no expressive power; it just fixes the additive gauge so "confident" is realized by the
tail going very negative rather than the winner running very positive. The penalty removes a redundant
degree of freedom, not a useful one. And that spread is the same `13.8` nats smoothing's own optimum
implied: the level fix and the target fix agree on *how much* log-odds a healthy prediction needs, and
disagree only on whether to let the whole stack float upward off that scale — which is precisely the part
that hurts in bfloat16.

Against the other level-side options: a hard logit clamp clamps an already-corrupted value after the
roundoff has happened and adds a kink the optimizer must route around, and gradient-norm clipping (already
on, and the spikes come anyway) reacts after the bad step is computed. The `(log Z)²` penalty is the only
one that acts smoothly *during* optimization to discourage the large logits from ever forming — the cause
rather than the symptom.

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

Two implementation details or the whole thing is subtly wrong. Reduction: the penalty has to be averaged
over *exactly* the valid (`targets != -1`) positions cross-entropy uses — the ignored positions' `log Z`
is meaningless garbage, and a different denominator would rescale the effective `λ` by the valid fraction,
off the clean ratio I just reasoned about. And compute `log Z` with the max-subtracting stabilized
`torch.logsumexp` so I do not introduce an overflow while fixing a numerical drift. The whole edit is then
plain cross-entropy plus `1e-4` times the masked-mean squared log-partition; the full scaffold is in the
answer.

So where smoothing softened the target and pulled on the zero-force gap, I leave the target the data and
add the one term that touches the free level. What I expect, falsifiably: `val_loss` should move *below*
smoothing's 2.3377 and below plain cross-entropy, since this neither distorts the evaluated objective nor
leaves the dangerous magnitude free — a recovery of a few hundredths of a nat, dragging WikiText-2 and
LAMBADA down a few percent each by the `perplexity = exp(loss)` map. Downstream accuracy should hold or
nudge up (a healthier softmax, no off-data bias), though single-seed moves of a point are within noise.
The residual risk is already visible: `2λ·log Z` is a force on the batch-*mean* `log Z`, not a per-position
ceiling, so a single excited coordinate can still spike while the average sits near zero — which says the
next move is to stop nudging the level and structurally bound the logit *values* themselves.
