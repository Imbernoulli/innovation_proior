Least confidence did roughly what the boundary argument said it would, and the places it *under*-delivered
are the tell. On seed 42 it beat random everywhere — letter 0.867 vs 0.835, spambase 0.948 vs 0.935,
splice 0.840 vs 0.771 — so chasing the model's least-confident points is genuinely better than drawing
blindly when the model is good. But look at the means, where the noise across runs shows through, and
difference them properly. On **letter** the least-confidence *mean* accuracy is 0.7955 and auc 0.668 —
*below* random's 0.816 / 0.724 by 0.0205 in accuracy and 0.056 in auc. And the seed-to-mean spread is the
alarm: least confidence's letter accuracy runs from 0.867 on seed 42 down to a 0.7955 mean, a gap of 0.0715,
roughly *four times* random's own letter accuracy spread of 0.0185. So least confidence didn't just fail to
improve letter on average — it made the run-to-run variance dramatically worse, which is the signature of a
rule that sometimes locks onto the right boundary and sometimes onto the wrong one. That inversion is exactly
the redundant-batch and bad-early-model risk I flagged: with 26 classes and a thin budget, the `n`
least-confident rows cluster on one stretch of the boundary, the early model defines "least confident" badly,
and the budget gets steered into an unrepresentative corner — so on a bad seed the curve rises slower than
random's representative draw, dragging the mean under while the lucky seed still shines. Back the bad seeds out
of the mean to see how catastrophic they are: if seed 42 is `0.867` and the three-seed mean is `0.7955`, the two
unlucky seeds average `(3 · 0.7955 − 0.867)/2 = (2.387 − 0.867)/2 ≈ 0.760` — nearly `0.11` below the lucky seed
and well under random's `0.816` mean. So it is not that least confidence is mildly worse on letter; it is that
the lucky seed is excellent and the unlucky ones are *catastrophic*, a bimodal outcome that screams "the early
model sometimes locks onto the wrong contested region and never recovers." A rule that subtracts off the
irreducible-noise points should narrow that spread, because much of what the bad seeds waste budget on is
precisely the `A`-type coin-flip glyphs I am about to formalize. On **spambase** least
confidence held up well (mean 0.927 / 0.908 vs random's 0.899 / 0.893), and on **splice** it gained (mean
0.814 / 0.744 vs 0.789 / 0.736) — the balanced and moderate datasets, where the early model locates a small
`R` reliably, are exactly where it paid off. So the diagnosis is sharp: raw top-class uncertainty is the right
instinct but it is brittle, and the brittleness has a specific cause I can name. `1 − max_y p(y|x)` is a
single deterministic forward pass through one network; it cannot distinguish "I am unsure because I have
not seen enough data here" (which a label fixes) from "I am unsure because this point is genuinely
ambiguous / the label is intrinsically noisy" (which no label ever fixes). It chases both, and the second
kind is wasted budget — on letter, the confusable-but-irreducible boundary between similar glyphs (an `l`
that overlaps an `i`, a rotated `n`/`u`) eats queries that buy nothing, and there are `~26²` such confusable
pairs to burn budget on. So the next rung has to *separate* the uncertainty a label can resolve from the
uncertainty it cannot.

Let me state what I actually want to query, cleanly, and let the form tell me what to compute. The honest
meaning of "this label teaches me the most" is "this label most shrinks my uncertainty about the model
parameters." If I had a posterior `p(θ|D)` over the network weights, the myopic-best query is the one
whose label I expect to drop the parameter entropy by the most:

  argmax_x  H[θ | D] − E_{y ∼ p(y|x,D)} [ H[θ | y, x, D] ].

The expectation over `y` is unavoidable — I don't have the label yet, so I average over what it might be,
weighted by the current predictive belief. This is the right object. The trouble is it lives entirely in
*parameter* space: `H[θ|D]` is the entropy of a posterior over a few thousand network weights, an
intractable high-dimensional blob, and the expectation costs a full re-inference per imagined label. Count
the cost honestly: for each of `N_x` candidate points I would have to imagine each of its `N_y` possible
labels, condition on it, and re-derive the parameter posterior to get `H[θ|y,x,D]` — that is `O(N_x · N_y)`
full posterior re-inferences *per round*, on top of an entropy over thousands of weights I cannot even write
down. I have a criterion I literally cannot afford to compute, and I should not patch it — I should look at
its *form*.

`H[θ|D] − E_y[H[θ|y,x,D]]` is entropy minus expected-entropy-after-conditioning, which is, by definition,
a mutual information: the conditional MI between the parameters and the unseen label, `I[θ, y | x, D]`.
This is an identity, not an approximation. And mutual information is *symmetric*: `I(A;B) = H(A) − H(A|B)
= H(B) − H(B|A)`. I have been writing it as the entropy of `θ` minus expected entropy of `θ` given `y`. It
is equally the entropy of `y` minus expected entropy of `y` given `θ`:

  I[θ, y | x, D]  =  H[y | x, D]  −  E_{θ ∼ p(θ|D)} [ H[y | x, θ] ].

Stop and read what just happened, and count the cost again on the right side. The left side is the
intractable parameter-space object — an entropy over thousands of weights, `O(N_x N_y)` re-inferences. The
right side has both entropies in *output* space — over the `n_classes` labels, so a sum over 2, 3, or 26
terms, trivial even when `θ` is huge — and the `θ` is conditioned only on `D`, not on any hypothetical `y`.
There is no "pretend I saw label `y` and re-infer" anywhere: I do inference once, on the data I already have,
and then for every candidate `x` compute two output-space entropies. The `O(N_x N_y)` re-inference collapses
to `O(N_x)` cheap entropy evaluations after a single inference, and the entropy of `θ` never appears. The
symmetry of MI turned the intractable criterion into a tractable one without changing the number by one bit.

Now read the right-hand side as a recipe and notice it says exactly the thing least confidence couldn't.
I want `x` that maximizes `H[y|x,D] − E_θ[H[y|x,θ]]`. First term big: my *marginal* prediction — averaged
over all settings of `θ` in the posterior — is very uncertain about the label. Second term small: but each
*individual* `θ` is confident. Put them together: I want the `x` where the parameter settings are each
individually sure of the answer yet *sure of different answers*. That is disagreement — but disagreement
that *keeps* the confidence, because the second term is built out of each setting's full entropy, not a
hard vote. This is precisely the distinction least confidence missed. Let me trace it on three concrete
points, using two posterior samples each, to make sure the subtraction really does what I claim and I am not
just narrating. Work in nats, `H(p) = −Σ_c p_c ln p_c`, on a binary label. Point `E` (epistemic): sample one
predicts `(0.9, 0.1)`, sample two predicts `(0.1, 0.9)` — each sample is sure, but they are sure of opposite
classes. The mean prediction is `p̄ = (0.5, 0.5)`, so `H[p̄] = ln 2 = 0.693`; each per-sample entropy is
`H(0.9, 0.1) = −0.9 ln 0.9 − 0.1 ln 0.1 = 0.095 + 0.230 = 0.325`, and their mean is `0.325`; the MI is
`0.693 − 0.325 = 0.368`, large — query it. Point `A` (aleatoric): both samples predict `(0.5, 0.5)` — the
model agrees the label is a genuine coin flip. Then `p̄ = (0.5, 0.5)`, `H[p̄] = 0.693`, mean per-sample
entropy `= 0.693`, and MI `= 0.693 − 0.693 = 0`, correctly skipped even though its marginal entropy is
maximal. Point `C` (confident-agreed): both samples predict `(0.95, 0.05)` — `p̄` entropy `0.199`, mean
per-sample entropy `0.199`, MI `= 0`, also skipped, as it should be. And to check the score is *graded*
and not just a three-way switch, point `B` (partial): sample one `(0.8, 0.2)`, sample two `(0.3, 0.7)` —
they lean opposite ways but neither is sharp. Then `p̄ = (0.55, 0.45)`, `H[p̄] = 0.688`; per-sample entropies
`H(0.8, 0.2) = 0.500` and `H(0.3, 0.7) = 0.611`, mean `0.556`; MI `= 0.688 − 0.556 = 0.132`, sitting between
`A`'s `0` and `E`'s `0.368`. The score rises smoothly with how much the confident samples disagree, which is
what I want from a ranking. Read the three together: least
confidence would score `A` and `E` *identically* (both have marginal top-mass `0.5`, so both score `1 − 0.5 =
0.5`) and would happily spend budget on `A`, the irreducible coin flip — which is exactly the wasted-budget
mechanism I measured on letter. This criterion scores `E` at `0.368` and `A` at `0`: the subtraction is the
operation that separates epistemic uncertainty (which a label can fix) from aleatoric uncertainty (which it
can't). Least confidence is essentially just the first term — the marginal uncertainty — which is why it
chases the irreducible boundary noise that dragged letter's mean under random; this criterion is the first term
*minus* the second, and the second term is the correction least confidence was missing. And the sign is
guaranteed: `H` is concave, so by Jensen `H[E_θ p] ≥ E_θ H[p]`, i.e. `H[p̄] ≥ mean per-sample entropy`, so
`I(x) ≥ 0` always — the three worked numbers (`0.368`, `0`, `0`) all respect it, the right sign for an MI.
Sanity-check the containment too: if the observation noise is zero, every `θ` predicts deterministically,
`H[y|x,θ] = 0`, the expectation vanishes, and the criterion reduces to plain marginal entropy — the
uncertainty-sampling family. So this isn't a fourth competitor; it *contains* uncertainty sampling as the
noise-free special case and adds the noise correction on top, which is exactly why I expect it to keep least
confidence's spambase/splice wins while repairing the letter regression that noise caused.

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
softmaxes. Note what the harness does *not* give me, and what I therefore drop from the full Bayesian-
disagreement story: there is no Gaussian-process path, no probit/squared-exponential closed form, no
nuisance-parameter marginalization, no preference-learning kernel — those need a GP posterior the scaffold
has no notion of. The only posterior I can sample here is the dropout one, so the only estimator I can run
is the Monte-Carlo one, and that is the rung.

I should be clear-eyed about *why* dropout passes are a defensible stand-in for posterior samples, because the
whole estimator leans on it. Training with dropout is, up to the choice of prior, fitting an approximate
variational posterior over the weights: each forward pass with dropout left on draws a different random mask,
which multiplies whole units by zero, and that is a draw of `θ` from that approximate posterior. It is crude —
the implied posterior is a specific structured family, not the true Bayesian one — but it is a *distribution
over functions* that concentrates where the data pin the weights down and stays spread where they don't, which
is exactly the property the `E_θ` in my criterion needs. I should note the honest failure mode this creates: if
dropout perturbs the network too gently, every pass produces nearly the same softmax, the samples never disagree,
`H[p̄] ≈ mean per-pass entropy`, and `I(x)` collapses toward zero *everywhere* — the estimator goes blind and I
am back to roughly ranking by marginal entropy. That is a real risk on a small net and I will watch letter for
it. Are there better posteriors I could reach for? A deep ensemble — train several independent networks and read
their disagreement — is a genuinely stronger posterior sample, but the harness retrains exactly *one* `self.clf`
per round and I cannot spawn a committee inside `query`. A Laplace approximation around the trained weights, or a
short SG-MCMC chain, would also give samples, but both need gradients and Hessians of the loss that the scaffold
does not expose to the acquisition rule. `predict_prob_dropout_split` is the one posterior-sampling primitive the
loop actually hands me, so it is the one I build on — not because it is the best posterior, but because it is the
only one reachable from inside this contract.

Plug the samples into the two terms with simple Monte-Carlo estimates. The expectation over the posterior
becomes an average over the passes. The marginal predictive `p(y|x,D) = E_θ[p(y|x,θ)]` becomes the mean
softmax across passes, `p̄ = (1/T) Σ_t pᵗ`. So the first term is the entropy of the mean prediction,
`H[p̄] = −Σ_c p̄_c log p̄_c`, high when the passes *together* are unsure; and the second term is the mean of
the per-pass entropies, `(1/T) Σ_t H[pᵗ] = −(1/T) Σ_{t,c} pᵗ_c log pᵗ_c`, high when each individual pass is
unsure. The disagreement score is their difference, `I(x) ≈ H[p̄] − (1/T) Σ_t H[pᵗ]` — precisely the
computation I just hand-traced on points `E`, `A`, `C`, now run over the whole pool with `T` passes instead
of two. How many passes do I need? `T` only has to estimate two class-entropies well enough to sort the pool,
not to nail either entropy precisely, and the Monte-Carlo error on each shrinks like `1/√T`; a handful of
passes gets the ordering stable, so `n_drop = 10` is the default — enough samples to estimate two entropies,
cheap enough to run over the whole pool in the one forward-pass budget the loop already spends.

One thing I should name honestly before I code it: this criterion is *myopic*. `H[θ|D] − E_y[H[θ|y,x,D]]`
scores the expected information from labeling `x` assuming I stop and re-infer immediately after — a one-step
lookahead. The truly optimal thing would be to reason about how a whole sequence of future queries interact,
which is a lookahead tree that branches on every possible label and is hopeless to compute. But the harness
structure makes the myopic compromise the right one rather than a lazy one: the loop retrains `self.clf` from
scratch after every round, so the model I score *next* round is a genuinely re-fit model, and the exploratory
feedback — this round's labels changing which points look informative next round — recovers much of what a
multi-step lookahead would have planned, without my having to plan it. Where myopia genuinely bites is *within*
a single batch: I score all `n` points against the *current* posterior, as if each were the only one added, so
the criterion has no idea that the 2nd-highest-MI point becomes redundant once the 1st is labeled. That is the
same batch-blindness least confidence had, wearing an information-theoretic coat, and no amount of getting the
per-point MI right will fix it — which is why I flag it now as the failure I expect to survive this rung.

One implementation point to get right against the literal scaffold, because the sign and sort direction
are where this silently breaks. I want to rank the pool by `I(x)` and keep the most-informative `n`. It is
cleaner to compute the *negative*, `U = (mean per-pass entropy) − (entropy of the mean) = −I(x)`, then
sort `U` ascending and take the first `n`: the `n` smallest `U` are the `n` largest `I` — and I can check the
direction against point `E`, whose `I = 0.368` gives `U = −0.368`, the most negative, so it sorts to the front
and gets queried, which is what I want. So the rung-3 edit: where least confidence took a single `predict_prob`
and the per-row max, I take `predict_prob_dropout_split` with `n_drop=10`, form `p̄` as the mean over passes,
compute `entropy1 = H[p̄]` (total) and `entropy2 = mean_t H[pᵗ]` (aleatoric), set `U = entropy2 − entropy1`,
sort ascending, return the first `n`. The distilled rule and the literal scaffold fill are in the answer.

Now the falsifiable expectations against least confidence's actual numbers. The cleanest prediction is
**letter**, the dataset where least confidence's mean *inverted under random* (0.7955 / 0.668 vs random's
0.816 / 0.724) precisely because raw top-class uncertainty chased irreducible 26-class boundary noise — the
`A`-type points in my trace. BALD's whole reason for existing is the `−E_θ[H[y|x,θ]]` term that subtracts
those points off, so I expect it to repair the letter regression — at minimum back above random's 0.816 /
0.724, ideally above least confidence's seed-42 0.867. If BALD does *not* fix letter, then either the dropout
posterior is too poor a surrogate for the weight posterior on this small net (the dropout samples all agree
because dropout perturbs too little, collapsing `I(x)` toward zero everywhere), or the redundant-batch problem
(which BALD does *not* address — it still scores each point in isolation) dominates the noise problem here. On
**spambase**, where least confidence was already strong (mean 0.927 / 0.908) and the binary boundary carries
little irreducible noise, the epistemic/aleatoric split has little to separate — the second term is near
constant across points, so `I(x)` reduces to roughly marginal entropy, and BALD should roughly *match* least
confidence, maybe a hair behind because ten noisy dropout passes are a coarser uncertainty estimate than one
clean softmax max — I would not be shocked to see spambase auc dip slightly under 0.908. On **splice** I expect
a small gain over least confidence's 0.814 / 0.744. And the structural caveat I am explicitly carrying forward:
BALD separates resolvable from irreducible uncertainty *per point*, but like least confidence it has no term
that looks at the *other chosen points* — the `n` highest-MI rows can still be near-duplicates, all sitting on
the same contested glyph pair. So if the batch is large relative to the contested region, BALD will still buy
redundant labels, and the next rung will have to fuse uncertainty with diversity so the batch is informative
*and* spread out — which is exactly the failure I expect to remain after fixing the noise problem.
