Least confidence did roughly what the boundary argument said it would, and the places it *under*-delivered
are the tell. On seed 42 it beat random everywhere — letter 0.867 vs 0.835, spambase 0.948 vs 0.935,
splice 0.840 vs 0.771 — so chasing the model's least-confident points is genuinely better than drawing
blindly when the model is good. But look at the means, where the noise across runs shows through. On
**letter** the least-confidence *mean* accuracy is 0.7955 and auc 0.668 — *below* random's 0.816 / 0.724.
That inversion is exactly the redundant-batch and bad-early-model risk I flagged: with 26 classes and a
thin budget, the `n` least-confident rows cluster on one stretch of the boundary, the early model defines
"least confident" badly, and the budget gets steered into an unrepresentative corner — so on a bad seed
the curve rises slower than random's representative draw, dragging the mean under. On **spambase** least
confidence held up well (mean 0.927 / 0.908 vs random's 0.899 / 0.893), and on **splice** it gained (mean
0.814 / 0.744 vs 0.789 / 0.736). So the diagnosis is sharp: raw top-class uncertainty is the right
instinct but it is brittle, and the brittleness has a specific cause I can name. `1 − max_y p(y|x)` is a
single deterministic forward pass through one network; it cannot distinguish "I am unsure because I have
not seen enough data here" (which a label fixes) from "I am unsure because this point is genuinely
ambiguous / the label is intrinsically noisy" (which no label ever fixes). It chases both, and the second
kind is wasted budget — on letter, the confusable-but-irreducible boundary between similar glyphs eats
queries that buy nothing. So the next rung has to *separate* the uncertainty a label can resolve from the
uncertainty it cannot.

Let me state what I actually want to query, cleanly, and let the form tell me what to compute. The honest
meaning of "this label teaches me the most" is "this label most shrinks my uncertainty about the model
parameters." If I had a posterior `p(θ|D)` over the network weights, the myopic-best query is the one
whose label I expect to drop the parameter entropy by the most:

  argmax_x  H[θ | D] − E_{y ∼ p(y|x,D)} [ H[θ | y, x, D] ].

The expectation over `y` is unavoidable — I don't have the label yet, so I average over what it might be,
weighted by the current predictive belief. This is the right object. The trouble is it lives entirely in
*parameter* space: `H[θ|D]` is the entropy of a posterior over a few thousand network weights, an
intractable high-dimensional blob, and the expectation costs a full re-inference per imagined label
(`O(N_x N_y)` posterior updates). I have a criterion I literally cannot afford to compute, and I should
not patch it — I should look at its *form*.

`H[θ|D] − E_y[H[θ|y,x,D]]` is entropy minus expected-entropy-after-conditioning, which is, by definition,
a mutual information: the conditional MI between the parameters and the unseen label, `I[θ, y | x, D]`.
This is an identity, not an approximation. And mutual information is *symmetric*: `I(A;B) = H(A) − H(A|B)
= H(B) − H(B|A)`. I have been writing it as the entropy of `θ` minus expected entropy of `θ` given `y`. It
is equally the entropy of `y` minus expected entropy of `y` given `θ`:

  I[θ, y | x, D]  =  H[y | x, D]  −  E_{θ ∼ p(θ|D)} [ H[y | x, θ] ].

Stop and read what just happened. The left side is the intractable parameter-space object. The right side
has both entropies in *output* space — over the `n_classes` labels, finite and trivial even when `θ` is
huge — and the `θ` is conditioned only on `D`, not on any hypothetical `y`. There is no "pretend I saw
label `y` and re-infer" anywhere: I do inference once, on the data I already have, and then for every
candidate `x` compute two output-space entropies. The `O(N_x N_y)` re-inference collapses to `O(1)`, and
the entropy of `θ` never appears. The symmetry of MI turned the intractable criterion into a tractable one
without changing the number by one bit.

Now read the right-hand side as a recipe and notice it says exactly the thing least confidence couldn't.
I want `x` that maximizes `H[y|x,D] − E_θ[H[y|x,θ]]`. First term big: my *marginal* prediction — averaged
over all settings of `θ` in the posterior — is very uncertain about the label. Second term small: but each
*individual* `θ` is confident. Put them together: I want the `x` where the parameter settings are each
individually sure of the answer yet *sure of different answers*. That is disagreement — but disagreement
that *keeps* the confidence, because the second term is built out of each setting's full entropy, not a
hard vote. This is precisely the distinction least confidence missed. A point that is noisy-but-known —
where every `θ` agrees "this one is a genuine coin flip" — has high `H[y|x,D]` but *also* high
`E_θ[H[y|x,θ]]` (each `θ` is itself uncertain), so the difference is small and I correctly skip it. A point
where the model is genuinely torn — each `θ` confident, but they split — has high `H[y|x,D]` and *low*
`E_θ[H[y|x,θ]]`, so the difference is large and I query it. The subtraction is exactly the operation that
separates epistemic uncertainty (which a label can fix) from aleatoric uncertainty (which it can't).
Least confidence is essentially just the first term — the marginal uncertainty — which is why it chases
the irreducible boundary noise that dragged letter's mean under random; this criterion is the first term
*minus* the second, and the second term is the correction least confidence was missing. Sanity-check the
containment: if the observation noise is zero, every `θ` predicts deterministically, `H[y|x,θ] = 0`, the
expectation vanishes, and the criterion reduces to plain marginal entropy — the uncertainty-sampling
family. The extra `−E_θ[H[y|x,θ]]` term is exactly what matters when that conditional uncertainty varies
with the input, as it does on a multi-class problem like letter. So this isn't a fourth competitor; it
*contains* uncertainty sampling as the noise-free special case and adds the noise correction on top.

Now make it concrete in *this* scaffold, which is where the derivation has to actually land. For the small
neural classifier `self.clf` the harness trains, the weight posterior `p(θ|D)` is hopelessly intractable
and there is no probit-convolution closed form to exploit — so I use the *model-agnostic* form,
`H[y|x,D] − E_θ[H[y|x,θ]]`, which needs only two things: posterior samples of `θ`, and the predictive
softmax `p(y|x,θ)` for each sample. The harness hands me exactly this through
`self.predict_prob_dropout_split(X, Y, n_drop)`: it leaves dropout switched on at prediction time and runs
the network `n_drop` times, each pass zeroing a different random subset of units, so each pass is
effectively a different network drawn from an approximate posterior over the weights, returning a tensor
`[n_drop, len(X), n_classes]`. So `n_drop` dropout passes stand in for `n_drop` posterior samples
`θ¹,…,θᵀ`, and I never need the posterior in closed form, never its entropy — I just sample it and read off
softmaxes. (Note what the harness does *not* give me, and what I therefore drop from the full Bayesian-
disagreement story: there is no Gaussian-process path, no probit/squared-exponential closed form, no
nuisance-parameter marginalization, no preference-learning kernel — those need a GP posterior the scaffold
has no notion of. The only posterior I can sample here is the dropout one, so the only estimator I can run
is the Monte-Carlo one, and that is the rung.)

Plug the samples into the two terms with simple Monte-Carlo estimates. The expectation over the posterior
becomes an average over the passes. The marginal predictive `p(y|x,D) = E_θ[p(y|x,θ)]` becomes the mean
softmax across passes, `p̄ = (1/T) Σ_t pᵗ`. So the first term is the entropy of the mean prediction,
`H[p̄] = −Σ_c p̄_c log p̄_c`, high when the passes *together* are unsure; and the second term is the mean of
the per-pass entropies, `(1/T) Σ_t H[pᵗ] = −(1/T) Σ_{t,c} pᵗ_c log pᵗ_c`, high when each individual pass is
unsure. The disagreement score is their difference,
`I(x) ≈ H[p̄] − (1/T) Σ_t H[pᵗ]`. The intuition survives the move intact: high exactly when each pass is
individually confident but the passes confidently disagree — the dropout networks vote different ways
while each is sure of itself. That is the epistemic part, with the aleatoric part subtracted off, just as
the symmetry promised. And because it is entropy-of-an-average minus average-of-entropies, the difference
is non-negative by Jensen's inequality (entropy is concave): `I(x) ≥ 0` always, the right sign for an MI.

One implementation point to get right against the literal scaffold, because the sign and sort direction
are where this silently breaks. I want to rank the pool by `I(x)` and keep the most-informative `n`. It is
cleaner to compute the *negative*, `U = (mean per-pass entropy) − (entropy of the mean) = −I(x)`, then
sort `U` ascending and take the first `n`: the `n` smallest `U` are the `n` largest `I`. A handful of
dropout passes suffices to estimate two entropies; `n_drop = 10` is the default — enough samples to
estimate them, cheap enough to run over the whole pool. So the rung-3 edit: where least confidence took a
single `predict_prob` and the per-row max, I take `predict_prob_dropout_split` with `n_drop=10`, form `p̄`
as the mean over passes, compute `entropy1 = H[p̄]` (total) and `entropy2 = mean_t H[pᵗ]` (aleatoric), set
`U = entropy2 − entropy1`, sort ascending, return the first `n`. The distilled rule and the literal
scaffold fill are in the answer.

Now the falsifiable expectations against least confidence's actual numbers. The cleanest prediction is
**letter**, the dataset where least confidence's mean *inverted under random* (0.7955 / 0.668 vs random's
0.816 / 0.724) precisely because raw top-class uncertainty chased irreducible 26-class boundary noise.
BALD's whole reason for existing is the `−E_θ[H[y|x,θ]]` term that subtracts that noise off, so I expect it
to repair the letter regression — at minimum back above random, ideally above least confidence's seed-42
0.867. If BALD does *not* fix letter, then either the dropout posterior is too poor a surrogate for the
weight posterior on this small net, or the redundant-batch problem (which BALD does *not* address — it
still scores each point in isolation) dominates the noise problem here. On **spambase**, where least
confidence was already strong (mean 0.927 / 0.908) and the binary boundary carries little irreducible
noise, the epistemic/aleatoric split has little to separate, so I expect BALD to roughly match least
confidence, maybe a hair behind because ten noisy dropout passes are a coarser uncertainty estimate than
one clean softmax max — I would not be shocked to see spambase auc dip slightly under 0.908. On **splice**
I expect a small gain over least confidence's 0.814 / 0.744. And the structural caveat I am explicitly
carrying forward: BALD separates resolvable from irreducible uncertainty *per point*, but like least
confidence it has no term that looks at the *other chosen points* — the `n` highest-MI rows can still be
near-duplicates. So if the batch is large relative to the contested region, BALD will still buy redundant
labels, and the next rung will have to fuse uncertainty with diversity so the batch is informative *and*
spread out — which is exactly the failure I expect to remain after fixing the noise problem.
