Random search told me exactly what I predicted, and the numbers split the way the theory said they would.
On final quality it is genuinely strong: XGBoost mean best −0.3939, SVM 0.9778, NN −3050 — competitive
optima everywhere, which is the low-effective-dimensionality lesson playing out. Where it bleeds is the
convergence AUC, and it bleeds *with high variance*: SVM AUC swings 0.5625 → 0.9132 → 0.8899 across seeds
for a mean of only 0.7885, and the NN AUC is the same story, 0.6648 / 0.6907 / 0.9619 for 0.7725. That is
the signature of statelessness — seed 123's draw order on SVM simply did not stumble onto a good config
until late, and nothing in the strategy pulled it back toward where the early evidence pointed, so the
best-so-far curve stayed flat for many evaluations and the area under it collapsed. The deficiency is not
final quality; it is that random search throws away every loss it has paid for. So the obvious next move is
to *use the history* — to maintain a model of where the good configurations live and concentrate draws
there, and the most principled way to do that for the continuous knobs is to adapt the shape of the
sampling distribution itself.

Let me be precise about what I am optimizing, because the encoding decides everything here. The loop hands
me configs over a mixed space — log-scaled floats, integers, one categorical per benchmark — but the engine
I want to build works on a vector in R^n. So I will work in a normalized box: encode every knob to [0,1]
(linear for plain ranges, log-linear for `log_scale` knobs, and a categorical to its choice-index divided
by the number of choices minus one), search in that box, and decode back at the end. I should flag the cost
of this up front, because it is going to matter: the categorical knob (SVM `kernel`, NN `activation`) gets
flattened to a single scalar coordinate and rounded back on decode, so the optimizer sees a *continuous*
relaxation of a discrete axis with no metric meaning. For SVM, where the kernel choice is one of three
coordinates out of three, that is a real distortion. I am betting that the continuous knobs dominate the
loss and the rounding noise on the categorical is tolerable; the alternative — a separate discrete handler —
is more machinery than this rung should carry, and the point of this rung is to test "adapt the continuous
shape," cleanly.

So: sample from a multivariate normal `N(m, σ²C)` over the box, rank the offspring by score, and use the
ranks to manufacture a better mean and a better ellipsoid for the next generation. Why a normal? Among all
distributions on R^n with prescribed second moments it has maximum entropy, so it commits to nothing beyond
the covariance I have actually learned, and it privileges no direction a priori — exactly what I want from a
general-purpose sampler. And why obsess over the *shape* `C` rather than just chasing the mean? Because near
a good region the objective looks locally quadratic, `f ≈ ½(x−x*)ᵀH(x−x*)`, and the ideal sampling
covariance there is `C ∝ H⁻¹`: in the coordinates `C^{-1/2}(x−x*)` the level sets become spheres and the
search is isotropic. Learning `C` is the gradient-free analogue of building the inverse-Hessian
preconditioner a quasi-Newton method maintains — except I have only ranks. The NN space is the one that
should reward this: it is 6-D with log-scaled widths, learning rate, and regularizer, almost certainly
ill-conditioned (a narrow viable learning-rate band, a much broader capacity band), and a method that learns
the anisotropy should converge there where random search wandered.

Now the actual update, which is the (μ/μ_w, λ)-CMA-ES the implementation fills in, derived in the order the
pieces become forced. First the population: `λ = 4 + floor(3·ln n)` offspring per generation (5–8 here for
n = 3–6), `μ = λ//2` parents, with rank-based recombination weights `w_i ∝ ln(μ+½) − ln i` normalized to sum
to one, so the best parent counts most and the weights decay smoothly. The variance-effective selection mass
`μ_eff = 1/Σw_i²` is the single number that summarizes "how many parents are really contributing," and it
shows up in every learning rate below — that is not a coincidence, it is the amount of selection information
per generation, and every rate has to be throttled by it or the estimate is noise. The new mean is the
weighted recombination of the μ best, `m ← Σ_i w_i x_{i:λ}`. That much is just "move toward the good
points."

The covariance is where the care goes, and the design principle is *derandomization*: do not infer the
strategy parameters indirectly from which offspring won; read them off the steps actually taken. Two
ingredients implement this. The rank-μ update estimates a covariance from this generation's selected steps,
`C_μ = Σ_i w_i y_i y_iᵀ` with `y_i = (x_{i:λ} − m_old)/σ` — crucially referenced to the *old* mean the points
were sampled around, not to the selected mean, because referencing the selected mean would systematically
*shrink* the variance along the very directions selection just favored (it measures spread *within* the
winners rather than the displacement of the winners), which is exactly backwards. The rank-one update adds
information across generations through the evolution path `p_c`, an exponentially-smoothed sum of the
realized mean-shifts: `p_c ← (1−c_c)p_c + h_σ·√(c_c(2−c_c)μ_eff)·(m−m_old)/σ`, and `C` gains `c1·p_c p_cᵀ`.
The point of the path is sign information the rank-μ term throws away: consecutive steps in the same
direction mean a long correlated ridge to stretch `C` along, whereas steps that zig-zag cancel in the path
and signal an over-long step. The covariance update is then the convex blend `C ← (1−c1−c_μ)C + c1·p_c p_cᵀ +
c_μ·Σ_i w_i y_i y_iᵀ`, with the standard rates `c1 = 2/((n+1.3)² + μ_eff)` and `c_μ = min(1−c1,
2(μ_eff−2+1/μ_eff)/((n+2)² + μ_eff))` — both small, both scaled by `μ_eff` and shrinking with dimension,
because I can only move `C` as fast as the selection signal justifies. After the update I symmetrize `C` and
floor its eigenvalues to keep it positive-definite, since the rank-deficient rank-μ term plus finite
arithmetic can otherwise push an eigenvalue negative and the next eigendecomposition would fail.

The step size σ rides a separate path, and keeping it separate from `C` is the whole trick of CSA
(cumulative step-size adaptation). The conjugate evolution path `p_σ ← (1−c_σ)p_σ + √(c_σ(2−c_σ)μ_eff)·
C^{-1/2}(m−m_old)/σ` accumulates the mean-shift in the *isotropic* frame (the `C^{-1/2}` whitening removes
the shape so only length and correlation across generations survive). Then `σ ← σ·exp((c_σ/d_σ)(‖p_σ‖/E‖N(0,
I)‖ − 1))`: if successive selected steps line up, the path is longer than a random walk of independent steps
would be, so I am making consistent progress and should lengthen σ; if they anticorrelate (overshoot,
zig-zag), the path is short and σ shrinks. The damping `d_σ` and the rate `c_σ = (μ_eff+2)/(n+μ_eff+5)`
control how fast σ may move. This σ path is also why CSA is unbiased under random selection: with no real
selection signal the expected path length is exactly `E‖N(0,I)‖`, so `E[ln σ]` is stationary — σ does not
drift when I am learning nothing, a safety property, not a cosmetic one. The `h_σ` flag in the `p_c` update
is the coupling: when `‖p_σ‖` is anomalously large (a big correlated move), it stalls the rank-one
accumulation for one step so a single outsized step does not blow up `C`.

In the scaffold this becomes a generational `suggest`: on first call, initialize `m = 0.5·𝟙`, `C = I`,
`σ = 0.3`, sample a whole population, and queue the decoded configs; each later call matches the returned
trial back to its pending candidate by `np.allclose` on the encoded vector, fills in its score, and when the
whole generation is scored, runs the update and resamples. Every evaluation is full fidelity — CMA-ES is a
single-fidelity optimizer, it has no notion of cheap noisy partial evaluations, so it spends the budget one
generation at a time. The distilled module is in the answer.

Here is where I have to be honest about the regime, because it is going to bite. CMA-ES is a method for
*many* generations: it pays an up-front cost learning `C` and `σ`, and the payoff is asymptotic, on
ill-conditioned landscapes, over hundreds to thousands of evaluations. My budget is 40–50. With λ ≈ 5–8 that
is only 5–10 generations — barely enough for the evolution paths to accumulate, let alone for `C` to bend
into the right ellipsoid. Worse, the σ₀ = 0.3 start in a unit box is large, so the first generations sample
widely and the early best-so-far curve can sit flat while the distribution is still contracting — and the
convergence AUC integrates exactly that early flatness. I therefore expect this rung to *lose* to random
search on convergence AUC, not beat it, despite being the more sophisticated method: random search's i.i.d.
draws get lucky early sometimes, whereas CMA-ES deterministically spends its first generations exploring
before it exploits. The categorical relaxation compounds the risk on SVM, where a third of the space is the
rounded kernel axis and the covariance machinery is modeling a discrete choice as if it were continuous.

So the falsifiable expectation against random search's numbers is specific and, for a "stronger" method,
uncomfortable. Final best scores should stay in the same competitive band — XGBoost near −0.40, SVM ~0.978,
NN ~3020–3060 — because even a few CMA generations land on a decent optimum. But the convergence AUC should
*degrade* relative to random search's 0.946 / 0.789 / 0.772, possibly badly on the benchmark where σ₀ is
most mismatched and the budget is shortest, with the worst single seed cratering (a generation that samples
wide and scores nothing before σ contracts). If I see CMA-ES post AUCs *below* random search's — a SVM mean
dropping toward 0.6, an isolated seed near 0.2 — that is not a bug; that is the diagnosis confirmed: a pure
continuous-shape optimizer is the wrong tool under a few-dozen-evaluation budget, because it cannot amortize
its model-learning cost. And that points straight at the next rung: the fix is not a better continuous model
but *cheaper evaluations* — spend the budget on many low-fidelity looks, keep only the survivors at high
fidelity — so that the strategy gets far more than 5–10 effective trials out of the same cost.
