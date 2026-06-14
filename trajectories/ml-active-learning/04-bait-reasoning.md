BALD did what the noise-subtraction argument predicted on the hard dataset and confirmed the limit I was
carrying forward. On **letter** it repaired the regression: mean accuracy 0.8357 / auc 0.716, back above
random's 0.816 / 0.724 on accuracy and well above least confidence's collapsed 0.7955 / 0.668, with seed
42 up at 0.893 — the strongest letter-accuracy seed of any rung so far. That is exactly the
`−E_θ[H[y|x,θ]]` term earning its keep: subtracting the irreducible 26-class boundary noise that raw least
confidence was chasing. But the means also confirm the caveat. On **spambase** BALD came in at 0.905 /
0.892, *below* least confidence's 0.927 / 0.908 and barely above random — on a near-balanced binary
problem with little irreducible noise, the epistemic/aleatoric split has almost nothing to separate, and
ten coarse dropout passes are a noisier uncertainty estimate than one clean softmax max, so BALD gives
back some of least confidence's spambase edge. On **splice** it landed 0.795 / 0.737, roughly random-level
and slightly under least confidence. So BALD is the best rung *on letter* but no longer dominant
everywhere, and the reason is the structural hole I named: BALD scores each point in isolation. It separates
resolvable from irreducible uncertainty *per point*, but it has no term that looks at the other chosen
points, so the `n` highest-MI rows can still be near-duplicates clustered in one stretch of the boundary —
I pay for `n` labels and learn far fewer labels' worth. And I cannot fix that by tuning, because tuning a
tradeoff coefficient would itself burn labels: changing a knob queries a *different* set, whose labels I'd
have to buy. So the next rung must do two things at once — keep the per-point informativeness BALD found
*and* make the batch diverse — with no free coefficient to balance them, and ideally on a principled
footing rather than a heuristic blend.

Let me refuse to pick a design criterion by taste and instead derive the actual error I am trying to
minimize, then read off which scalar of the model's information it forces. Stop thinking of `self.clf` as a
function and think of it as a probability model: a softmax is `p(y|x,θ)`, fitting is maximum likelihood
with loss `ℓ = −log p(y|x,θ)`, and a whole body of theory about *how good an MLE is* becomes available.
The quantity that governs the MLE's error is the Fisher information `I(x;θ) = E_y[∇²ℓ(x,y;θ)]`, the
expected Hessian of the per-point loss — classically the MLE is asymptotically normal with covariance the
inverse Fisher, so the Fisher is the *precision* a labeled point buys me about `θ`. There is a structural
gift to check: does the Hessian depend on the label `y`, or only on `x` and `θ`? For multiclass logistic
regression, the first-derivative block for class `p` is `(π_p − 1[y=p]) x`, linear in the label indicator,
so its second derivative kills the label entirely, leaving `∇²ℓ = xxᵀ ⊗ (diag(π) − ππᵀ)` — label-
independent. That is the property that lets me even *talk* about a point's Fisher before paying for its
label, which is the whole game.

So I want to pick a labeling set `S` that minimizes the MLE error, the Fisher is the currency, and I need
a *scalar* of the matrix `Σ_{x∈S} I(x;θ)` to minimize. Experimental design hands me a menu — D-optimality
(maximize `det`), A-optimality (minimize `tr` of the inverse), others — and is annoyingly agnostic about
which. I won't guess; I'll derive the error and read off the scalar. Take Bayesian linear regression as
the case I can compute in closed form: the Bayes risk of the ridge estimate, measured in the pool second-
moment metric `Σ`, telescopes (every cross term cancels in pairs) down to `BayesRisk(S) = σ² tr(Λ_S⁻¹ Σ)`
with `Λ_S = Σ_{x∈S} xxᵀ + λσ² I`. Two things fall out that I did not put in. First, the criterion is a
*weighted trace*, not a determinant and not a bare `tr(Λ⁻¹)` — weighted by the pool second moment `Σ`. The
risk literally tells me which scalar to optimize, and it is A-optimality with `Σ` baked in. Second, the
right side contains *no labels*: the risk of labeling `S` depends only on the features in `S` and the
pool, so I can minimize it *before paying for a single label* — an oracle-free objective. And the
frequentist MLE analysis lands on the same functional, `tr(I_S(θ)⁻¹ I_U(θ)) / m`, with the per-point
Fisher in place of `xxᵀ` and the pool Fisher `I_U` in place of `Σ`. Two independent derivations converging
on `tr((Σ_{x∈S} I(x;θ))⁻¹ I_U(θ))` is the object to trust.

Why this beats the determinant — and therefore why it should beat BADGE, which I'm about to face as the
strongest rung — matters, so pin it down. If I tried D-optimality, `det(I(x;θ)⁻¹ I_U) = det(I(x;θ)⁻¹) ·
det(I_U)`, and `det(I_U)` is a constant in `x` that factors straight out: the determinant is structurally
*blind* to the pool Fisher. It only ever looks at each candidate's own Fisher, never at which directions
the pool actually weights. The trace does not collapse — `tr((Σ_S I)⁻¹ I_U)` genuinely couples the
selected Fisher to the pool Fisher, so it preferentially shores up the directions `I_U` says matter. And
`tr(M⁻¹)` is the sum of inverse eigenvalues, dominated by the *smallest* eigenvalues — A-optimality pours
effort into the directions of lowest information, the weak spots, exactly the ones a determinant (a
product, dominated by large eigenvalues) ignores. For active learning that is the right instinct: fix what
you're worst at, weighted by how much the pool cares. So the target objective is

  S* = argmin_{S⊂U, |S|≤B}  tr( (Σ_{x∈S} I(x;θ))⁻¹ I_U(θ) ).

Now reality, and this is where the *task's* implementation diverges hard from the textbook method, because
the harness imposes a CPU memory budget the full algorithm would blow through. Several things are broken.
First, `I(x;θ)` for the full network is enormous, so I restrict to the *last layer*: `I(x;θ^L) = V_x V_xᵀ`
where `V_x` is a `(d·k)×k` factor whose columns are the per-class last-layer loss gradients each scaled by
`√(class prob)` — and `V_x V_xᵀ` reproduces the exact pointwise Fisher `x^L x^Lᵀ ⊗ (diag(π) − ππᵀ)`. (Note
the contrast with BADGE, which uses *one* column — the gradient at the single hallucinated label — and does
*not* scale by `√p`; BADGE's embedding is a rank-one shadow of this rank-`k` Fisher, throwing away `k−1`
directions and the probability weighting. That is the per-point advantage of the full `V_x`.) The scaffold
advertises a `get_exp_grad_embedding` helper that returns exactly these per-class Fisher factors — but I
deliberately do *not* call it here, and that is the central adaptation: materializing the full
`[n_pool, k, d·k]` Fisher tensor and accumulating the `d·k × d·k` pool Fisher over the entire pool would
exhaust the 64 GB letter budget. So I rebuild the Fisher factors myself in *streaming batches* through a
`DataLoader` (the model's penultimate embeddings and softmax per batch, assembled into `V_x` on the fly),
accumulate the pool Fisher and the labeled "seed" Fisher batch-by-batch divided by their counts, and never
hold the whole tensor at once.

Second, `d·k` is still large — for letter, 26 classes times the penultimate width is thousands — so before
selection I **random-project** the Fisher factors down to a fixed `bait_proj_dim = 128` with a single
fixed Gaussian projection (seeded for reproducibility, scaled by `1/√128`). Johnson–Lindenstrauss says
random projection approximately preserves the inner products the trace objective is built from, so the
A-optimal geometry survives the squeeze while the `d·k × d·k` matrices I'd otherwise invert become
`128 × 128`. Third, even projected, scoring the *entire* unlabeled pool every greedy step is too slow, so I
run BAIT on an **entropy-filtered candidate pool** rather than the full unlabeled set: compute predictive
entropy for every unlabeled point in the same streaming pass, keep the top `max(4n, 512)` most-uncertain as
candidates, and let the Fisher selection diversify *within* that high-uncertainty shortlist. This is a
pragmatic fusion — uncertainty pre-filters, A-optimality diversifies — that keeps the expensive matrix
work on a few hundred points instead of tens of thousands. Fourth, the representation changes every round
as the harness retrains, so this is not the convex theory's one-shot SDP; I recompute the Fisher from the
current network and re-solve each round.

The selection itself is greedy, because exact combinatorial minimization is intractable, but greedy has a
caveat I must face: the trace functional is *not* submodular, so plain forward greedy can commit early to
points that become redundant once later ones are added and can't take them back. So I *oversample* — greedily
add `2n` points forward — then run a *backward* pass that greedily removes points one at a time, each time
deleting the one whose removal hurts the objective least, down to `n`. Forward casts a wide net; backward
prunes the ones that turned out redundant in company. The per-step argmin looks like a `128 × 128` inversion
per candidate per step, which I avoid with the low-rank structure: each `I(x) = V_x V_xᵀ` is rank `k`, so
the Woodbury identity turns `(M + V_x V_xᵀ)⁻¹` into an update needing only a `k × k` solve, and the cyclic
trace lets me precompute `M⁻¹ I_U M⁻¹` once per step and score every candidate with a small batched matmul.
One numerical adaptation the CPU port forces: because the projected, candidate-filtered Fisher can be
rank-deficient and ill-conditioned, I use the *pseudo*-inverse `torch.linalg.pinv` everywhere a plain
inverse appears in the textbook version, and `nan_to_num` the scores — the full-precision `torch.inverse`
would throw on singular matrices here. The seed/candidate normalization is the same as the derived risk:
scale the seed Fisher into `M_0` by `nLabeled/(nLabeled+n)` and each candidate `V_x` by `√(n/(nLabeled+n))`
so the new batch joins the labeled set in the correct proportion.

So the rung-4 edit, against the literal scaffold and honest about what the harness omits: where BADGE will
seed k-means++ over rank-one gradient embeddings, BAIT here streams the rank-`k` Fisher factors, projects
them to 128 dims, restricts to an entropy-filtered candidate shortlist, and runs the A-optimal greedy
(forward-oversample to `2n`, Woodbury+trace-rotation scoring with `pinv`, backward-prune to `n`). It is the
*same* A-optimal objective as the published method, but a CPU-tractable adaptation of it — projected,
candidate-filtered, pseudo-inverted — not the full-pool `get_exp_grad_embedding` version. The distilled rule
and the literal scaffold fill are in the answer.

Now the falsifiable expectations against BALD's actual numbers, and they are genuinely uncertain because
the adaptations cut both ways. On **spambase** and **splice** the full Fisher objective with the pool term
should beat the per-point-isolated BALD, because the A-optimality directly targets the diversity hole that
left BALD's batch redundant: I expect spambase back above BALD's 0.905 / 0.892 (toward least confidence's
0.927 / 0.908) and splice above BALD's 0.795 / 0.737. The sharpest worry is **letter** — the dataset where
BALD was strongest (0.8357 / 0.716, seed 42 up at 0.893) — because letter is exactly where the CPU
adaptations bite hardest: 26 classes means the largest `d·k`, so the random projection to 128 dims discards
the most Fisher geometry, and the streaming full-pool Fisher accumulation is the slowest, the one the time
budget can fail to finish. I expect BAIT to *underperform* BALD on letter, and possibly not complete every
seed there — if the leaderboard shows a missing or degraded letter result for BAIT, that is the projection-
and-time adaptation showing through, not the A-optimal objective failing. So the bar BAIT must clear is:
beat BALD on spambase and splice (where diversity is the binding constraint), at the cost of letter (where
the CPU port handicaps it). If instead BAIT loses on spambase/splice too, then the entropy-pre-filter plus
128-dim projection has degraded the Fisher geometry past usefulness, and the diversity must be recovered
more cheaply — which is exactly what the next rung does, replacing the heavy Fisher machinery with a single
gradient embedding whose k-means++ seeding gets diversity for free.
