Random search told me exactly what I predicted, and the numbers split the way the theory said they would.
On final quality it is genuinely strong: XGBoost mean best −0.3939, SVM 0.9778, NN −3050 — competitive
optima everywhere, which is the low-effective-dimensionality lesson playing out, and the final scores are
tight across seeds (SVM identical to six figures on two of three seeds). Where it bleeds is the convergence
AUC, and it bleeds *with high variance*, exactly as the geometric-arrival argument said it would. Put the
coefficient of variation on each benchmark's AUC: XGBoost is 0.945 mean with a std of 0.020, a CV of only
0.021; SVM is 0.789 mean with a std of 0.160, a CV of 0.203; the NN is 0.772 mean with a std of 0.134, a CV
of 0.174. So the two benchmarks where the good config has to be *waited for* — SVM's kernel-gated slab, the
NN's narrow viable region — have AUC coefficients of variation roughly ten times XGBoost's, and SVM's is
the largest of all, driven by seed 123 collapsing to 0.5625 against the other seeds' 0.89 and 0.91. That is
the signature of statelessness and it matches the mechanism precisely: seed 123's draw order on SVM simply
did not stumble onto a good config until late, nothing in the strategy pulled it back toward where the early
evidence pointed, so the best-so-far curve stayed flat and the area under it collapsed. It is worth noting
I had guessed the NN would be the noisiest and it is SVM that is; but the prediction that mattered — AUC
weak and high-variance on the wait-limited benchmarks, final quality untouched — held on both. The
deficiency is not final quality; it is that random search throws away every loss it has paid for. So the
obvious next move is to *use the history* — to maintain a model of where the good configurations live and
concentrate draws there.

There are two families for "use the history," and I should say why I pick one over the other rather than
default into it. The first is a surrogate-plus-acquisition method: fit a probabilistic regressor of the
loss and maximize an acquisition to pick each next point. That is the sequential-model-based route in the
background, and it is the natural thing to reach for — but a Gaussian-process surrogate wants a metric on
the whole configuration vector, and my spaces have a categorical axis (SVM kernel, NN activation) with no
natural metric, and its Expected-Improvement exploration stakes everything on a predictive variance that a
sparse early sample can silently collapse. I want to be honest about which of these objections actually
binds: the textbook complaint about a GP is its O(n³) conditioning cost, but at n = 40 evaluations that is
40³ = 64,000 operations, utterly negligible next to a single train-and-score, so compute is *not* the reason
to defer it. The reasons that bite are structural — the categorical axis has no kernel metric, and the
variance-collapse failure is silent and happens exactly in the sparse-early-data regime I live in — and
those do not go away by spending more budget. I am not ready to take on that machinery at this rung; I want
to isolate a cleaner ingredient first. The second family is to adapt the *sampling distribution itself* —
learn, from the ranks of what I have evaluated, the shape of the region where good configs live, and draw
the next batch from that shape. That is the one continuous-optimization idea that uses history without ever
needing a calibrated loss model or a metric across the categorical, so it is the honest first step up from
stateless sampling, and it is what this rung builds.

Even within "adapt the sampling distribution" there is a real fork, and the arithmetic decides it. I could
adapt only the *scale* of each axis — a separable, diagonal Gaussian, an axis-independent
estimation-of-distribution method with n parameters — or I could adapt the full *shape*, a general
covariance with n(n+1)/2 parameters that can capture rotation and correlation between axes. The diagonal
model is cheap to fit but it is blind to exactly the structure I most expect here: these spaces are
ill-conditioned in *rotated* directions, not just anisotropic along the axes (a log learning rate and a log
capacity trade off against each other, so the viable region is a tilted ridge, not an axis-aligned box). A
full covariance is the only thing that fits a tilted ridge. But the full model has a cost I have to face
head-on: for the 6-D benchmarks that is n(n+1)/2 = 21 free covariance parameters, and my budget of 40–50
evaluations at full fidelity buys only a handful of generations to estimate them. I will do the generation
count below and it is going to be the whole story; for now the decision is that I take the full covariance
*because* the rungs above me will need to know whether continuous-shape adaptation is worth its cost, and
the only way to learn that is to run the version that can actually model the ridge and see whether the
budget can fund it. So: sample from a multivariate normal N(m, σ²C) over a normalized box, rank the
offspring by score, and use the ranks to manufacture a better mean and a better ellipsoid for the next
generation.

Let me be precise about the encoding, because it decides everything downstream. The loop hands me configs
over a mixed space — log-scaled floats, integers, one categorical per benchmark — but the engine works on a
vector in R^n. So I work in a normalized box: encode every knob to [0,1] (linear for plain ranges,
log-linear for `log_scale` knobs, and a categorical to its choice-index divided by the number of choices
minus one), search in that box, and decode back at the end. I should flag the cost of this up front,
because it is going to matter: the categorical knob gets flattened to a single scalar coordinate and rounded
on decode, so the optimizer sees a *continuous* relaxation of a discrete axis with no metric meaning. For
SVM, where the kernel is one of three coordinates out of three, that is a real distortion — a third of the
search dimension is a rounded fiction. I am betting that the continuous knobs dominate the loss and the
rounding noise on the categorical is tolerable; the alternative, a separate discrete handler, is more
machinery than this rung should carry, and the point of this rung is to test "adapt the continuous shape,"
cleanly.

Why a normal? Among all distributions on R^n with prescribed second moments it has maximum entropy, so it
commits to nothing beyond the covariance I have actually learned, and it privileges no direction a priori —
exactly what I want from a general-purpose sampler. And why obsess over the *shape* C rather than just
chasing the mean? Because near a good region the objective looks locally quadratic, f ≈ ½(x−x*)ᵀH(x−x*),
and the ideal sampling covariance there is C ∝ H⁻¹: in the coordinates C^{−1/2}(x−x*) the level sets become
spheres and the search is isotropic. Learning C is the gradient-free analogue of building the
inverse-Hessian preconditioner a quasi-Newton method maintains — except I have only ranks. The NN space is
the one that should reward this: 6-D with log-scaled widths, learning rate, and regularizer, almost
certainly ill-conditioned (a narrow viable learning-rate band, a much broader capacity band), so a method
that learns the anisotropy should, in principle, converge there where random search wandered — *if* it is
given enough generations to bend C into shape.

Now the actual update, the (μ/μ_w, λ)-CMA-ES the implementation fills in, derived in the order the pieces
become forced, with the numbers worked so I can see how much learning each generation actually buys. First
the population: λ = 4 + floor(3·ln n) offspring per generation, which is 4 + floor(3·1.099) = 7 for the 3-D
SVM and 4 + floor(3·1.792) = 9 for the 6-D XGBoost and NN, with μ = λ//2 parents (3 and 4). The rank-based
recombination weights w_i ∝ ln(μ+½) − ln i normalized to sum to one give, for μ = 4, the vector
(0.53, 0.29, 0.14, 0.04) — the best parent counts most, the weights decay smoothly. The variance-effective
selection mass μ_eff = 1/Σw_i² is the single number summarizing "how many parents are really contributing,"
and here it is 2.03 for SVM and 2.60 for XGBoost/NN. That number is small and it is going to haunt every
learning rate below, because it is the amount of selection information per generation, and every rate has to
be throttled by it or the estimate is noise. The new mean is the weighted recombination of the μ best,
m ← Σ_i w_i x_{i:λ}. That much is just "move toward the good points."

The covariance is where the care goes, and the design principle is *derandomization*: do not infer the
strategy parameters indirectly from which offspring won; read them off the steps actually taken. The rank-μ
update estimates a covariance from this generation's selected steps, C_μ = Σ_i w_i y_i y_iᵀ with
y_i = (x_{i:λ} − m_old)/σ — crucially referenced to the *old* mean the points were sampled around, not to
the selected mean, because referencing the selected mean would systematically *shrink* the variance along
the very directions selection just favored (it measures spread within the winners rather than the
displacement of the winners), which is exactly backwards. The rank-one update adds information across
generations through the evolution path p_c, an exponentially-smoothed sum of the realized mean-shifts:
p_c ← (1−c_c)p_c + h_σ·√(c_c(2−c_c)μ_eff)·(m−m_old)/σ, and C gains c1·p_c p_cᵀ. The path carries the sign
information the rank-μ term throws away: consecutive steps in the same direction mean a long correlated
ridge to stretch C along, whereas steps that zig-zag cancel in the path and signal an over-long step. The
covariance update is the convex blend C ← (1−c1−c_μ)C + c1·p_c p_cᵀ + c_μ·Σ_i w_i y_i y_iᵀ, with the
standard rates c1 = 2/((n+1.3)² + μ_eff) and c_μ = min(1−c1, 2(μ_eff−2+1/μ_eff)/((n+2)² + μ_eff)).

Now put numbers on those rates for the 6-D case and read what they mean, because this is the computation
that predicts the whole rung. c1 = 2/(53.29 + 2.60) = 0.0358 and c_μ = 2(0.985)/(64 + 2.60) = 0.0296, so
the covariance retains a fraction (1 − c1 − c_μ) = 0.935 of its old value every generation and admits only
about 6.5% new information per step. Over a run of five generations the identity I started C at persists
with weight 0.935^5 = 0.71 — so after the *entire* XGBoost budget, roughly seventy percent of the sampling
covariance is still the original isotropic I, essentially unbent. On the NN, with only 40/9 = 4.4
generations, it is worse; on SVM the low dimension makes c1 larger (0.0975) so 0.481 of C survives after
five generations, but SVM has only three axes and one of them is the rounded categorical fiction, so there
is little real shape to learn. The parameter count says the same thing from the other side: 21 covariance
parameters for the 6-D spaces against μ_eff ≈ 2.6 effective samples per generation over ~5 generations,
about 13 effective samples total — fewer samples than parameters, so C is formally underdetermined for the
whole run. CMA-ES is a method for *many* generations; it pays an up-front cost learning C and σ and the
payoff is asymptotic, over hundreds to thousands of evaluations. My budget buys four to six generations, and
the arithmetic says that is not enough to fill in the covariance before the budget is spent.

Let me hand-trace one axis of one generation to make both the "old-mean" rule and the "C barely bends"
claim concrete rather than asserted. Take the 6-D weights (0.53, 0.286, 0.143, 0.041), and suppose along
one coordinate the four selected parents landed at (0.70, 0.65, 0.60, 0.55) after being sampled around
m_old = 0.5 with σ = 0.3. The new mean along that axis is 0.53·0.70 + 0.286·0.65 + 0.143·0.60 + 0.041·0.55
= 0.665, a shift of +0.165 — a real, decisive step toward the good side. Now the rank-μ variance, referenced
to the *old* mean: y_i = (x_i − 0.5)/0.3 = (0.667, 0.5, 0.333, 0.167), and Σ w_i y_i² = 0.324, so the
selected steps report a variance of 0.324 along this axis. Blend that into C with the rates above:
(1 − c1 − c_μ)·1.0 + c_μ·0.324 = 0.935 + 0.0096 = 0.944, so even along an axis where selection moved
consistently and decisively, the sampling variance drops from 1.0 to only 0.944 in this generation — a 5.6%
contraction. That is the "barely bends" fact at the single-step, single-axis level, and it compounds to the
0.71 survival of the identity over five generations I computed above. The trace also shows why the old-mean
reference is not a technicality: if I had instead referenced the *new* mean 0.665, the same selected points
give y_i = (0.116, −0.051, −0.218, −0.384) and Σ w_i y_i² = 0.021 — fifteen times smaller. Referencing the
selected mean would report the variance along the very axis selection just favored as 0.021 instead of
0.324, collapsing C toward a needle exactly where the search is making progress, which is precisely
backwards. The factor of fifteen is the size of the mistake the old-mean convention avoids.

The step size σ rides a separate path, and keeping it separate from C is the whole trick of CSA (cumulative
step-size adaptation). The conjugate evolution path p_σ ← (1−c_σ)p_σ + √(c_σ(2−c_σ)μ_eff)·C^{−1/2}(m−m_old)/σ
accumulates the mean-shift in the *isotropic* frame (the C^{−1/2} whitening removes the shape so only length
and correlation across generations survive). Then σ ← σ·exp((c_σ/d_σ)(‖p_σ‖/E‖N(0,I)‖ − 1)): if successive
selected steps line up, the path is longer than a random walk of independent steps would be, so I am making
consistent progress and should lengthen σ; if they anticorrelate (overshoot, zig-zag), the path is short and
σ shrinks. Here c_σ = (μ_eff+2)/(n+μ_eff+5) = 0.338 and the damping d_σ = 1.34 for the 6-D spaces, so the
per-generation multiplier on σ is exp(0.338/1.34 · (‖p_σ‖/χ_n − 1)) = exp(0.25·(…)), which even under
maximally consistent selection changes σ by at most about a factor exp(±0.25) ≈ 1.28 per generation — so
over five generations σ can move by at most roughly 1.28^5 ≈ 3.5× up or 0.29× down. That bound matters
because of where σ *starts*. I initialize σ₀ = 0.3 in a unit box with m₀ = 0.5·𝟙, so the first generation's
one-standard-deviation spread already reaches [0.2, 0.8] on every axis and its two-sigma spread reaches
[−0.1, 1.1], clipped to the box edges — the opening generation is nearly as spread out as uniform sampling.
The early best-so-far curve therefore sits flat while the distribution is still contracting, and the σ bound
says it can contract by at most a factor of a few over the whole run, so on the shortest-budget benchmark
there may simply not be time to go from wide exploration to tight exploitation. The convergence AUC
integrates exactly that early flatness.

Two properties of this machinery are worth checking as sanity rather than trusting. First, the h_σ flag in
the p_c update is the coupling between the two paths: when ‖p_σ‖ is anomalously large (a big correlated
move), it stalls the rank-one accumulation for one step so a single outsized step does not blow up C.
Second — and this is a genuine degenerate-case check, not a reassurance — CSA is unbiased under random
selection: with no real selection signal the expected conjugate path length is exactly E‖N(0,I)‖ = χ_n, so
the exponent averages to zero and E[ln σ] is stationary. That means σ does not drift when I am learning
nothing, which is the safety property I want; if it failed, σ would random-walk away from a sensible scale
during the many low-information early steps this budget is full of. After the covariance update I symmetrize
C and floor its eigenvalues to keep it positive-definite (eigenvalues floored at 1e-20, σ clipped to
[1e-10, 1]), since the rank-deficient rank-μ term plus finite arithmetic can otherwise push an eigenvalue
negative and the next eigendecomposition would fail.

There is a subtler budget cost hidden in the generational structure, and it sharpens the amortization
problem. The very first generation is sampled from N(0.5·𝟙, 0.3²·I) with no adaptation at all — it is, by
construction, essentially isotropic random sampling, just with a smaller-than-uniform spread. So CMA-ES
spends its opening λ evaluations (9 on the NN, 9 on XGBoost, 7 on SVM) reproducing what random search does,
and only *after* that first generation is scored does any covariance or step-size adaptation begin. On the
NN that leaves 40 − 9 = 31 evaluations, or about 3.4 further generations, of actual learning — so the
effective adaptation budget is not "4.4 generations" but closer to three, because the first one is a sunk
random sweep. Against random search, then, CMA-ES starts from behind: it pays a full generation to match
the baseline before it can try to beat it, and the arithmetic on c1, c_μ, and the σ contraction bound says
the three generations it has left cannot recover that head start under this budget.

One implementation detail interacts badly with the wide σ₀ and is worth naming, because it further weakens
the covariance estimate. Each sampled point is clipped, `x = np.clip(x, 0, 1)`, so the effective sampling
distribution is a *truncated* normal, not a normal, and with σ₀ = 0.3 and a mean at 0.5 the two-sigma
spread already reaches past the box edges — a real fraction of each early generation piles up exactly on the
boundary. That pile-up biases the rank-μ estimate: the y_i for clipped points understate the step CMA
actually intended, so the covariance CMA reads off the winners is a distorted, edge-flattened version of the
one it sampled from. It is a second-order effect, but it runs the same direction as everything else here —
it makes the little bit of shape the budget could have taught even less trustworthy in the early
generations, which are the only generations this budget has.

In the scaffold this becomes a generational `suggest`: on first call, initialize m = 0.5·𝟙, C = I, σ = 0.3,
sample a whole population, and queue the decoded configs; each later call matches the returned trial back to
its pending candidate by `np.allclose` on the encoded vector, fills in its score, and when the whole
generation is scored, runs the update and resamples. Every evaluation is full fidelity — CMA-ES is a
single-fidelity optimizer, it has no notion of cheap noisy partial evaluations, so it spends the budget one
generation at a time. That single-fidelity commitment means `total_evals` should come back pinned to the
budget again (50/40/40), the same tell as random search — a useful cross-check that the generational
book-keeping is not silently spending extra evaluations. The distilled module is in the answer.

So the falsifiable expectation against random search's numbers is specific and, for a "stronger" method,
uncomfortable. Final best scores should stay in the same competitive band — XGBoost near −0.40, SVM ~0.978,
NN ~3020–3060 — because even four or five CMA generations land on a decent optimum, and the encoding does
not lose the fat targets random search found. But the convergence AUC should on average *degrade* relative
to random search's 0.946 / 0.789 / 0.772, because the arithmetic says the covariance barely bends and σ
opens wide and cannot contract fast enough to front-load the curve within the budget. I expect the damage
concentrated where the budget is shortest and the anisotropy thinnest — the low-D SVM and XGBoost, where
there is little rotated structure for C to earn its keep and the wide σ₀ opening is pure waste — with the
worst single seed cratering when a generation samples wide and scores nothing competitive before σ
contracts (an AUC seed dropping toward 0.2 would be that failure, not a bug). The one place covariance
learning has any chance to pay is the ill-conditioned NN, but with only 4.4 generations it is a race the
arithmetic says is close. If I see CMA-ES post average AUCs *below* random search's with an isolated
cratered seed, that is the diagnosis confirmed: a pure continuous-shape optimizer is the wrong tool under a
few-dozen-evaluation budget because it cannot amortize its model-learning cost — every evaluation costs a
full unit and there are too few of them. And that points straight at the next rung: the fix is not a better
continuous model but *cheaper evaluations* — spend the budget on many low-fidelity looks, keep only the
survivors at high fidelity — so the strategy gets far more than four to six effective generations out of the
same cost.
