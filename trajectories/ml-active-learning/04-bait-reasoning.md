BALD did what the noise-subtraction argument predicted on the hard dataset and confirmed the limit I was
carrying forward. On **letter** it repaired the regression: mean accuracy 0.8357 / auc 0.716, back above
random's 0.816 / 0.724 on accuracy and well above least confidence's collapsed 0.7955 / 0.668, with seed
42 up at 0.893 — the strongest letter-accuracy seed of any rule so far. That is exactly the
`−E_θ[H[y|x,θ]]` term earning its keep: subtracting the irreducible 26-class boundary noise that raw least
confidence was chasing. But the means also confirm the caveat, and I should difference them to see the size
of the giveback. On **spambase** BALD came in at 0.905 / 0.892, *below* least confidence's 0.927 / 0.908 —
a drop of `0.022` in accuracy and `0.016` in auc — and barely above random. On **splice** it landed 0.795 /
0.737, `0.019` under least confidence's accuracy and slightly under on auc, roughly random-level. So the
noise correction that rescued letter actively *cost* me on the two low-noise datasets, and the reason is
mechanical: where there is almost no aleatoric uncertainty to subtract, the `−E_θ[H]` term is near-constant
across points and adds nothing but estimator noise — ten coarse dropout passes are a shakier ranking signal
than one clean softmax max, so on spambase and splice BALD is paying for a correction it doesn't need. So
BALD is the best rule *on letter* but no longer dominant everywhere, and the reason it can't hold spambase
and splice is the structural hole I named: BALD scores each point in isolation. It separates resolvable from
irreducible uncertainty *per point*, but it has no term that looks at the other chosen points, so the `n`
highest-MI rows can still be near-duplicates clustered in one stretch of the boundary — I pay for `n` labels
and learn far fewer labels' worth. And I cannot fix that by tuning, because tuning a tradeoff coefficient
would itself burn labels: changing a knob queries a *different* set, whose labels I'd have to buy to even
evaluate the change, so a held-out sweep is not available to me the way it is in ordinary supervised tuning.
So the next step must do two things at once — keep the per-point informativeness BALD found *and* make the
batch diverse — with no free coefficient to balance them, and ideally on a principled footing rather than a
heuristic blend.

Let me refuse to pick a design criterion by taste and instead derive the actual error I am trying to
minimize, then read off which scalar of the model's information it forces. Stop thinking of `self.clf` as a
function and think of it as a probability model: a softmax is `p(y|x,θ)`, fitting is maximum likelihood
with loss `ℓ = −log p(y|x,θ)`, and a whole body of theory about *how good an MLE is* becomes available.
The quantity that governs the MLE's error is the Fisher information `I(x;θ) = E_y[∇²ℓ(x,y;θ)]`, the
expected Hessian of the per-point loss — classically the MLE is asymptotically normal with covariance the
inverse Fisher, so the Fisher is the *precision* a labeled point buys me about `θ`. There is a structural
gift to check, and it is the gift that makes the whole scheme possible: does the Hessian depend on the label
`y`, or only on `x` and `θ`? Work the multiclass logistic case. With `π = softmax(θ·features)`, the loss for
observed class `y` is `−log π_y`, and its gradient in the block for class `p` is `(π_p − 1[y=p]) · x`, linear
in the label indicator `1[y=p]`. Differentiate once more: the `−1[y=p]` term is constant in `θ`, so it dies,
and what survives is `∇²ℓ = (diag(π) − ππᵀ) ⊗ xxᵀ` — the label has vanished entirely. The Hessian of the
per-point loss is `y`-independent. That is the property that lets me even *talk* about a point's Fisher before
paying for its label: I can compute `I(x;θ)` from `x` and the current model alone, which is the whole game,
because the entire point of active learning is to rank points I have *not* yet labeled.

So I want to pick a labeling set `S` that minimizes the MLE error, the Fisher is the currency, and I need
a *scalar* of the matrix `Σ_{x∈S} I(x;θ)` to minimize. Experimental design hands me a menu — D-optimality
(maximize `det`), A-optimality (minimize `tr` of the inverse), E-optimality (the smallest eigenvalue), others
— and is annoyingly agnostic about which. I won't guess; I'll derive the error and read off the scalar. Take
Bayesian linear regression as the case I can compute in closed form: the Bayes risk of the ridge estimate,
measured in the pool second-moment metric `Σ`, telescopes — expand the estimator error, and every cross term
between distinct points cancels in expectation because the noise is independent and mean-zero — down to
`BayesRisk(S) = σ² tr(Λ_S⁻¹ Σ)` with `Λ_S = Σ_{x∈S} xxᵀ + λσ² I`. Two things fall out that I did not put in.
First, the criterion is a *weighted trace*, not a determinant and not a bare `tr(Λ⁻¹)` — weighted by the pool
second moment `Σ`. The risk literally tells me which scalar to optimize, and it is A-optimality with `Σ` baked
in. Second, the right side contains *no labels*: the risk of labeling `S` depends only on the features in `S`
and the pool, so I can minimize it *before paying for a single label* — an oracle-free objective, exactly the
label-independence the Fisher Hessian just guaranteed. And the frequentist MLE analysis lands on the same
functional, `tr(I_S(θ)⁻¹ I_U(θ)) / m`, with the per-point Fisher in place of `xxᵀ` and the pool Fisher `I_U`
in place of `Σ`. Two independent derivations converging on `tr((Σ_{x∈S} I(x;θ))⁻¹ I_U(θ))` is the object to
trust.

Why this beats the determinant — and therefore why it should out-diversify the alternatives — matters, so
pin it down with numbers, not adjectives. If I tried D-optimality, `det(I(x;θ)⁻¹ I_U) = det(I(x;θ)⁻¹) ·
det(I_U)`, and `det(I_U)` is a constant in `x` that factors straight out: the determinant is structurally
*blind* to the pool Fisher. It only ever looks at each candidate's own Fisher, never at which directions
the pool actually weights. The trace does not collapse — `tr((Σ_S I)⁻¹ I_U)` genuinely couples the
selected Fisher to the pool Fisher, so it preferentially shores up the directions `I_U` says matter. And the
*eigenvalue* behavior is the clincher. Suppose the accumulated selected Fisher `M` has eigenvalues
`(10, 1, 0.1)` along three directions. Then `tr(M⁻¹) = 1/10 + 1/1 + 1/0.1 = 0.1 + 1 + 10 = 11.1`, and note
that the single weakest direction (`λ = 0.1`) contributes `10` of that `11.1` — the trace-of-inverse is
*dominated by the smallest eigenvalue*, so A-optimality pours effort into exactly the directions of lowest
information, the weak spots. Meanwhile `det(M) = 10 · 1 · 0.1 = 1` is a product dominated by the *large*
eigenvalues; halving the weak `0.1` direction to `0.05` barely moves the determinant (`1 → 0.5`) but sends
`tr(M⁻¹)` from `11.1` to `21.1`, nearly doubling it. So the determinant shrugs at the weak direction the trace
frantically tries to fix. For active learning that is the right instinct: fix what you're worst at, weighted
by how much the pool cares — precisely the pool-aware diversity BALD's per-point score had no way to express.
So the target objective is

  S* = argmin_{S⊂U, |S|≤B}  tr( (Σ_{x∈S} I(x;θ))⁻¹ I_U(θ) ).

Now reality, and this is where the *task's* implementation diverges hard from the textbook method, because
the harness imposes a CPU memory budget the full algorithm would blow through, and I should do the arithmetic
that proves it rather than assert it. First, `I(x;θ)` for the full network is enormous, so I restrict to the
*last layer*: `I(x;θ^L) = V_x V_xᵀ` where `V_x` is a `(d·k)×k` factor whose columns are the per-class
last-layer loss gradients each scaled by `√(class prob)` — and `V_x V_xᵀ` reproduces the exact pointwise
Fisher `x^L x^Lᵀ ⊗ (diag(π) − ππᵀ)`. I could imagine cutting the per-point cost further by keeping only a
single column of `V_x` — the loss gradient at one hallucinated label, unscaled by `√p` — a rank-one shortcut
instead of this rank-`k` factor. But that shortcut is a rank-one shadow of the true Fisher, throwing away `k−1`
of its directions and the probability weighting, and A-optimality is precisely a criterion about covering *all*
the weak directions of the information matrix, so collapsing the per-point Fisher to rank one would blind the
objective in the directions it most needs to see. So I keep the full rank-`k` `V_x`. That is the per-point
fidelity of the full factor, and it is also its per-point cost.
Put letter's numbers on the cost. Letter has 26 classes, so `k = 26`, and the penultimate width `d` is order
a hundred, so `d·k` is order a few thousand — call it ~3000. The scaffold advertises a `get_exp_grad_embedding`
helper that returns exactly these per-class Fisher factors, but materializing the full `[n_pool, k, d·k]`
tensor over letter's pool (order 15,000 rows) is `15000 × 26 × 3000 ≈ 1.2×10⁹` floats, ~4.7 GB *per copy* in
float32, and the greedy selection touches several copies — that is what exhausts the CPU budget. So I
deliberately do *not* call `get_exp_grad_embedding`; instead I rebuild the Fisher factors myself in *streaming
batches* through a `DataLoader` (the model's penultimate embeddings and softmax per batch, assembled into `V_x`
on the fly), accumulate the `d·k × d·k` pool Fisher and the labeled "seed" Fisher batch-by-batch divided by
their counts, and never hold the whole tensor at once.

Second, even a single `d·k × d·k` matrix is order `3000² ≈ 9×10⁶` entries and I have to invert it repeatedly,
so before selection I **random-project** the Fisher factors down to a fixed `bait_proj_dim = 128` with a single
fixed Gaussian projection (seeded for reproducibility, scaled by `1/√128`). The projected matrices are `128 ×
128`, order `1.6×10⁴` entries — a ~500× shrink — and the inversion goes from choking to instant. The reason
this is allowed and not vandalism: the trace objective is built entirely out of *inner products* of the Fisher
factors (`tr((Σ V Vᵀ)⁻¹ I_U)` is a contraction of `⟨V_a, V_b⟩`-type quantities), and Johnson–Lindenstrauss
says a random Gaussian projection to `m` dimensions preserves all pairwise inner products to relative error
`~1/√m`, so `m = 128` keeps the A-optimal geometry to a few-percent distortion while making the matrices
tiny. Third, even projected, scoring the *entire* unlabeled pool every greedy step is too slow, so I run BAIT
on an **entropy-filtered candidate pool** rather than the full unlabeled set: compute predictive entropy for
every unlabeled point in the same streaming pass, keep the top `max(4n, 512)` most-uncertain as candidates, and
let the Fisher selection diversify *within* that high-uncertainty shortlist. This is a pragmatic fusion —
uncertainty pre-filters, A-optimality diversifies — that keeps the expensive matrix work on a few hundred
points instead of tens of thousands. Put numbers on the shortlist: for a batch of `n = 100`, the candidate pool
is `max(4·100, 512) = 512`, so out of letter's ~15,000 unlabeled rows the greedy only ever scores 512 — a ~30×
reduction in the matrix work, which is what buys tractability. But I should be honest that this pre-filter is
not free geometry: entropy and the A-optimal trace do not rank points the same way, so a point with only
*moderate* predictive entropy that would nonetheless plug a weak Fisher direction gets thrown out before the
A-optimal step ever sees it. On a binary or 3-class problem the shortlist of 512 is a large fraction of the
contested region, so little is lost; on letter, with 26 classes spreading uncertainty across many glyph pairs,
512 may not even cover all the confusable pairs, so the entropy filter and the projection stack their losses
exactly where the pool is hardest. That coupling is a second reason to expect letter, specifically, to be where
this CPU port is most fragile. Fourth, the representation changes every round as the harness retrains,
so this is not the convex theory's one-shot SDP; I recompute the Fisher from the current network and re-solve
each round.

The selection itself is greedy, because exact combinatorial minimization over subsets is intractable, but
greedy has a caveat I must face: the trace functional is *not* submodular, so plain forward greedy can commit
early to points that become redundant once later ones are added and can't take them back. Make the non-submodularity concrete so the fix is not superstition. Say the objective is best served by a pair
of points `{c, d}` that jointly cover two weak directions, but individually each of `c`, `d` looks mediocre
because alone it only half-covers one direction; meanwhile a point `a` looks great alone because it dumps all
its Fisher on the single weakest direction. Forward greedy grabs `a` first (best marginal gain), then is biased
toward points that complement `a` rather than toward the `{c, d}` pair it should have taken — and because the
functional is non-submodular, the marginal gains do not diminish in the orderly way that would let a greedy
argument bound the loss, so it can stay stuck with `a` in the batch. So I *oversample* — greedily add `2n`
points forward, wide enough that `c` and `d` both get pulled in alongside `a` — then run a *backward* pass that
greedily removes points one at a time, each time deleting the one whose removal hurts the objective least,
down to `n`; that pass is exactly what lets me drop `a` once `c` and `d` are both present and revealed to make
`a` redundant. Forward casts a wide net; backward prunes the ones that turned out redundant in company —
recovering much of what a one-directional greedy would have locked out. The per-step argmin looks like a `128 × 128` inversion
per candidate per step, which I avoid with the low-rank structure: each `I(x) = V_x V_xᵀ` is rank `k`, so
the Woodbury identity turns `(M + V_x V_xᵀ)⁻¹` into an update needing only a `k × k` solve — for letter a
`26 × 26` solve rather than a `128 × 128` inversion — and the cyclic property of the trace lets me precompute
`M⁻¹ I_U M⁻¹` once per step and score every candidate with a small batched matmul. Over `2n` forward steps and a
few hundred candidates each, that is the difference between the round finishing and the round timing out. One numerical adaptation
the CPU port forces: because the projected, candidate-filtered Fisher can be rank-deficient and
ill-conditioned (a `128 × 128` matrix built from a few hundred rank-26 factors need not be full rank), I use
the *pseudo*-inverse `torch.linalg.pinv` everywhere a plain inverse appears, and `nan_to_num` the scores — the
full-precision `torch.inverse` would throw on singular matrices here. The seed/candidate normalization is the
same as the derived risk: scale the seed Fisher into `M_0` by `nLabeled/(nLabeled+n)` and each candidate `V_x`
by `√(n/(nLabeled+n))` so the new batch joins the labeled set in the correct proportion.

So the step-4 edit, against the literal scaffold and honest about what the harness omits: BAIT here streams the
rank-`k` Fisher factors, projects them to 128 dims, restricts to an entropy-filtered candidate shortlist, and
runs the A-optimal greedy
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
adaptations bite hardest, and the arithmetic says why: 26 classes means the largest `d·k` (~3000), so the
random projection to 128 dims is the most aggressive squeeze — from ~3000 down to 128, a ~24× compression
where the `1/√m` JL distortion is riding on the largest ambient dimension — and the streaming full-pool Fisher
accumulation over letter's ~15,000 rows is the slowest, the one the time budget can fail to finish. I expect
BAIT to *underperform* BALD on letter, and possibly not complete every seed there — if the letter result comes
back missing or degraded, that is the projection-and-time adaptation showing through, not the A-optimal objective
failing. So the bar BAIT must clear is: beat BALD on spambase and splice (where
diversity is the binding constraint), at the cost of letter (where the CPU port handicaps it). If instead BAIT
loses on spambase/splice too, then the entropy-pre-filter plus 128-dim projection has degraded the Fisher
geometry past usefulness, and the whole fused-objective idea needs to be recovered by machinery light enough to
survive letter — no `d·k × d·k` matrix, no projection, no pseudo-inverse — which is the open problem this step
hands forward: keep the pool-aware diversity, pay only uncertainty-sampling cost.
