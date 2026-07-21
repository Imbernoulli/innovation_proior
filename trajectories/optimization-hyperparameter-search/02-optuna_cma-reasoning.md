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

A normal is the maximum-entropy choice for prescribed second moments, so it commits to nothing beyond the
covariance I have learned. And the reason to obsess over the *shape* C rather than just chase the mean:
near a good region the objective looks locally quadratic, f ≈ ½(x−x*)ᵀH(x−x*), and the ideal sampling
covariance there is C ∝ H⁻¹, which whitens the level sets to spheres. Learning C is the gradient-free
analogue of the inverse-Hessian preconditioner a quasi-Newton method maintains, from ranks only. The NN
should reward this most — 6-D, almost certainly ill-conditioned (a narrow viable learning-rate band, a much
broader capacity band) — *if* it is given enough generations to bend C into shape.

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

Those rates for the 6-D case predict the whole rung. c1 = 2/(53.29 + 2.60) = 0.0358 and
c_μ = 2(0.985)/(64 + 2.60) = 0.0296, so the covariance retains (1 − c1 − c_μ) = 0.935 of its old value every
generation and admits only ~6.5% new information per step. Over five generations the identity I started C at
persists with weight 0.935^5 = 0.71 — so after the *entire* XGBoost budget, roughly seventy percent of the
sampling covariance is still the original isotropic I, essentially unbent; the NN's 40/9 = 4.4 generations is
worse, and SVM's larger c1 (0.0975, leaving 0.481 after five generations) buys little because two of its three
axes are the rounded categorical fiction. The parameter count says the same: 21 covariance parameters against
μ_eff ≈ 2.6 effective samples per generation over ~5 generations — about 13 effective samples, fewer than
parameters, so C is formally underdetermined for the whole run. CMA-ES pays an up-front cost learning C and σ
with an asymptotic payoff over hundreds of evaluations; four to six generations cannot fill in the covariance
before the budget is spent.

The old-mean reference is the one convention worth pinning down with a number, because it is not a
technicality. Take one axis where the four selected parents landed at (0.70, 0.65, 0.60, 0.55) after being
sampled around m_old = 0.5 with σ = 0.3. Referenced to the old mean, y_i = (x_i − 0.5)/0.3 =
(0.667, 0.5, 0.333, 0.167) and Σ w_i y_i² = 0.324 — the displacement of the winners. Referenced instead to
the *new* mean 0.665, the same points give y_i = (0.116, −0.051, −0.218, −0.384) and Σ w_i y_i² = 0.021,
fifteen times smaller — the spread *within* the winners. Using the selected mean would collapse C toward a
needle along the very axis selection just favored, which is exactly backwards; the factor of fifteen is the
size of the mistake the old-mean convention avoids.

The step size σ rides a separate path, and keeping it separate from C is the whole trick of CSA (cumulative
step-size adaptation). The conjugate evolution path p_σ ← (1−c_σ)p_σ + √(c_σ(2−c_σ)μ_eff)·C^{−1/2}(m−m_old)/σ
accumulates the mean-shift in the *isotropic* frame (the C^{−1/2} whitening removes the shape so only length
and correlation across generations survive). Then σ ← σ·exp((c_σ/d_σ)(‖p_σ‖/E‖N(0,I)‖ − 1)): consistent selected steps make the path longer than a
random walk and lengthen σ, zig-zagging steps shorten it and shrink σ. With c_σ = 0.338 and damping
d_σ = 1.34 for the 6-D spaces, the per-generation multiplier is exp(0.25·(…)), so even under maximally
consistent selection σ changes by at most ≈1.28 per generation — over five generations at most ≈3.5× up or
0.29× down. That bound matters because of where σ starts: at σ₀ = 0.3 in a unit box with m₀ = 0.5·𝟙, the
first generation's one-σ spread already reaches [0.2, 0.8] on every axis, nearly as wide as uniform sampling.
The early best-so-far curve sits flat while the distribution contracts, and the σ bound says it can contract
by at most a factor of a few over the whole run — so on the shortest-budget benchmark there may simply not be
time to go from wide exploration to tight exploitation, and the convergence AUC integrates exactly that
flatness.

The h_σ flag in the p_c update couples the two paths: when ‖p_σ‖ is anomalously large it stalls the
rank-one accumulation for one step so a single outsized step does not blow up C. The safety property that
matters under this budget is that CSA is unbiased under random selection — with no selection signal the
expected conjugate path length is exactly χ_n, so E[ln σ] is stationary and σ does not random-walk away
from a sensible scale during the many low-information early steps. After the covariance update I symmetrize
C and floor its eigenvalues (at 1e-20, σ clipped to [1e-10, 1]), since the rank-deficient rank-μ term plus
finite arithmetic can otherwise push an eigenvalue negative and the next eigendecomposition would fail.

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

A second detail runs the same direction: each point is clipped, `x = np.clip(x, 0, 1)`, so the effective
distribution is a *truncated* normal, and with σ₀ = 0.3 and mean 0.5 the two-sigma spread reaches past the
box edges and a real fraction of each early generation piles up on the boundary. The y_i for clipped points
understate the step CMA intended, so the covariance read off the winners is an edge-flattened distortion of
the one sampled from — a second-order effect, but it makes the little shape the budget could teach even less
trustworthy in exactly the early generations that are all this budget has.

In the scaffold this becomes a generational `suggest`: on first call initialize m = 0.5·𝟙, C = I, σ = 0.3,
sample a population, queue the decoded configs; each later call matches the returned trial back by
`np.allclose` on the encoded vector, fills in its score, and runs the update and resample once the generation
is scored. Every evaluation is full fidelity — CMA-ES has no notion of cheap partial evaluations — so
`total_evals` should again pin to the budget (50/40/40), a cross-check that the generational bookkeeping is
not silently overspending. The distilled module is in the answer.

So the falsifiable expectation against random search's numbers is, for a "stronger" method, uncomfortable.
Final best scores should stay in the same competitive band random search found, because even four or five
CMA generations land on a decent optimum and the encoding does not lose the fat targets. But the convergence
AUC should on average *degrade* relative to random search's 0.946 / 0.789 / 0.772, because the covariance
barely bends and σ opens wide and cannot contract fast enough to front-load the curve within the budget. I
expect the damage concentrated where the budget is shortest and the anisotropy thinnest — the low-D SVM and
XGBoost, where there is little rotated structure for C to earn its keep — with a single seed liable to
crater when a generation samples wide and scores nothing competitive before σ contracts. The one place
covariance learning has any chance to pay is the ill-conditioned NN, but with only 4.4 generations the
arithmetic says it is close. If CMA-ES posts average AUCs *below* random search's with an isolated cratered
seed, the diagnosis is confirmed: a pure continuous-shape optimizer cannot amortize its model-learning cost
under a few-dozen-evaluation budget. The fix is not a better continuous model but *cheaper evaluations* —
many low-fidelity looks, survivors kept at high fidelity — so the strategy gets far more than four to six
effective generations out of the same cost.
