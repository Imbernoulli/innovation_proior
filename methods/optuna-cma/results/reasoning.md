Let me start from what actually hurts. I have a black box `f: R^n -> R` to minimize, I can only read function values, and every value costs me a simulation. So the only thing I am allowed to spend is function evaluations, and the only sane currency for "did I do well" is how many of them I needed to reach a target. The landscapes that motivate this are the nasty ones: ill-conditioned, so the favorable directions are orders of magnitude narrower than the unfavorable ones; non-separable, so I cannot decompose the problem into `n` one-dimensional searches and just sweep each coordinate; non-convex and sometimes rugged or noisy, so anything that wants a clean gradient or Hessian — BFGS, conjugate gradients — is out. I have ranks of sampled points and nothing else.

What should I sample from? I want a randomized search, and the most honest choice of distribution is a multivariate normal `N(m, C)`: among all distributions on `R^n` with prescribed (co)variances it has the largest entropy, so it commits to nothing beyond the second moments I have actually learned, and it distinguishes no coordinate direction, which is what I want from a general-purpose sampler that has not yet learned anything about the problem's orientation. A normal distribution is completely described by its mean `m` and covariance `C`, and `C` factorizes: with the eigendecomposition `C = B D^2 B^T`, where `B` has orthonormal columns (the principal axes) and `D` is diagonal (the axis lengths), I can sample by drawing `z ~ N(0, I)`, scaling by `D`, rotating by `B`, and shifting: `x = m + B D z ~ N(m, C)`. The one-`sigma` contour is the ellipsoid `(x-m)^T C^{-1} (x-m) = const`, principal axes given by the eigenvectors of `C`, axis lengths by `sigma` times square-roots of the eigenvalues. So the whole game is: out of ranked samples, manufacture a good mean and a good ellipsoid.

Why do I care so much about the ellipsoid and not just the mean? Picture the simplest nontrivial model the optimizer will ever face near a minimum, a convex quadratic `f(x) = 1/2 (x-x*)^T H (x-x*)` with positive-definite Hessian `H`. Suppose I sample with covariance `C = H^{-1}` and substitute `u = C^{-1/2}(x-x*) = H^{1/2}(x-x*)`. Then `f = 1/2 (x-x*)^T H (x-x*) = 1/2 u^T H^{-1/2} H H^{-1/2} u = 1/2 u^T u = 1/2 ||u||^2`, so in `u`-coordinates the level sets of `f` are exact spheres and the sampling distribution `C^{-1/2} N(0,C) = N(0,I)` is isotropic — every direction equally good, the search trivially isotropic. The cancellation is exact: `H^{-1/2} H H^{-1/2} = H^{-1/2} H^{1/2} H^{1/2} H^{-1/2} = I`. So on the quadratic the *ideal* search covariance is the inverse Hessian, up to scale. That is a clean target: learning a good `C` is the black-box analogue of building the inverse-Hessian preconditioner a quasi-Newton method maintains from gradients — except I have no gradients, only ranks. The condition number of `C`, largest over smallest eigenvalue, is how anisotropic my distribution is, and matching it to the conditioning of `f` is the whole difference between a search that grinds to a halt on an ill-conditioned valley and one that doesn't. I will hold onto this `C -> H^{-1}` target as the thing to check at the end: whatever update I build, on a quadratic it had better drive the learned `C` toward `H^{-1}` up to scale, or it is not doing the one thing I designed it to do.

Now, how have people set the shape of the sampling distribution without gradients? The evolution-strategy tradition climbs a ladder. Level one: isotropic Gaussian, one free knob, the global step size `sigma`, contours are spheres. Level two: diagonal covariance, `n` per-coordinate step sizes, axis-parallel ellipsoids. Level three: arbitrary zero-mean normal, `(n^2+n)/2` parameters, arbitrarily oriented ellipsoids, can produce any normal. Climbing the ladder is exactly buying the ability to fit ill-conditioned non-separable problems. But there is a catch I should pin down immediately, because it is going to dictate the design: level two privileges the coordinate axes, so it is not rotation-invariant — rotate the problem and the diagonal model no longer aligns with anything. Only level three, *if* its adaptation is formulated independently of the coordinate system, restores rotation invariance. And invariance is not a luxury here. If my method is invariant to rotation of the search space, then whatever I prove or measure on a nice separable problem transfers verbatim to an arbitrarily rotated, non-separable version of it; one experiment generalizes to a whole orbit of problems. Likewise, if I use only the *ranking* of `f`-values and never their magnitudes, I am automatically invariant to any strictly monotone rescaling of `f`. So two of my design criteria — rank-only and rotation invariance — I will treat as hard constraints, not nice-to-haves. Rank-only I can enforce trivially by never reading a magnitude. Rotation invariance is the demanding one: it rules out level two (a diagonal model is tied to the axes) and, within level three, rules out any update rule that refers to a coordinate frame. Every adaptation step I write down will have to pass through these two filters.

So how do I set those `(n^2+n)/2` parameters from ranks alone? The classical answer is mutative strategy parameter control: attach the strategy parameter to each individual, mutate *it*, generate the object point with the mutated value, and let selection on the object point decide which strategy parameter survives. For a single global step size that is

  sigma_k = sigma * exp(xi_k),   x_k = x + sigma_k z_k,   z_k ~ N(0,I),

pick the best offspring, repeat. The hope is that a strategy-parameter value that *produced* a selected step is a good value going forward. Let me actually stress-test this, because I am about to inherit its failure modes if I am not careful. The selection of a strategy-parameter value is *indirect*: a value of `sigma_k` is judged only through the single object point it happened to spawn, and two different `sigma_k` differ in their probability of being selected by a small, noisy margin — the better one is only slightly more likely to be selected, "slightly" being swamped by the randomness of which `z_k` it got paired with. So the selection signal on the strategy-parameter level is heavily disturbed. Worse, maximizing selection probability is not the same as maximizing progress rate, and the well-documented consequence is that this scheme tends to drive `sigma` *too small*. And the change rate I can realize per generation is bounded by the finite amount of selection information; when I have `n` or `(n^2+n)/2` strategy parameters to set, the per-parameter change rate has to drop, so adaptation slows, and the only knob that decouples change rate from mutation strength is the parent number `mu` — which forces the population to scale roughly linearly with the number of strategy parameters. Fitting a full covariance this way needs huge populations and burns my whole evaluation budget on estimating internal parameters. Wall. The indirection is the disease: I am inferring the strategy parameter from the noisy fact of which object point won, instead of reading it off the steps I actually took.

The cure has a name in this tradition — derandomization — and the principle is exactly to remove that indirection. The first level of it just decouples the realized change rate of `sigma` from the mutation strength used to probe it, by inserting a damping: `sigma_k = sigma * exp(xi_k / d)` with `d >= 1`, so I can probe with a large `xi` (for a clean selection difference) but only let `sigma` move a little. Useful, but still mutative — I am still mutating `sigma` and selecting indirectly. The decisive move is the second level: stop mutating the strategy parameter at all, and use the *realized* step to update it directly. After I take a step `sigma z_k`, I already know `z_k`; its length `||z_k||` tells me whether the step I just committed to was long or short compared to what an unselected step would have been. So

  sigma <- sigma * exp( ( ||z_k|| - E||N(0,I)|| ) / d ).

If the selected step was longer than a random one would have been, increase `sigma`; if shorter, decrease it. No strategy-parameter mutation, no indirect selection — the realized mutation vector *is* the signal. Notice the three principles this crystallizes, and I will hold myself to all three for whatever I build: (1) change the distribution so the probability of reproducing the *selected* step goes up; (2) explicitly control the change rate of the strategy parameters (that is what `d` is for); (3) keep the strategy parameters *stationary under random selection* — if `f` returns pure noise so selection is meaningless, the expected value of (the log of) each strategy parameter must be unchanged generation to generation, or I will systematically drift even when I am learning nothing. The third is subtle and I almost missed it: an adaptation rule that is biased to grow `sigma` under random selection will diverge whenever selection pressure is low; one biased to shrink will converge prematurely. Unbiasedness on the log scale is a safety property, not a cosmetic one.

Good. Now generalize from "the realized step length updates one scalar `sigma`" to "the realized steps update the whole covariance." Two motivations push me here at once. First, the right thing to adapt is a general linear encoding of the search space, and adapting all variances and covariances of a zero-mean normal is *exactly* adapting a general linear transformation — any linear map of an `N(0,I)` mutation is a zero-mean normal, and any zero-mean normal is some linear map of `N(0,I)` — so the covariance is the natural and complete object. Second, the problems I care about are non-separable, and a non-separable problem cannot be cracked by adapting only `n` individual step sizes; I need the off-diagonal correlations.

Let me think about how to learn `C` the most direct way I can, before getting clever: just *estimate* it from one generation. I sample `lambda` points `x_1, ..., x_lambda` from `N(m, C)` (take `sigma = 1` for now), rank them, keep the `mu` best. How should I estimate a covariance from the good ones? My instinct is the textbook empirical covariance — but stop, there is a choice of reference mean hiding here that I should examine before writing anything down. Option A, the ordinary empirical covariance, references the mean of the *selected points themselves*:

  C_emp = (1/(mu-1)) sum_i (x_{i:lam} - mean_of_selected)(x_{i:lam} - mean_of_selected)^T.

Option B references the *old* distribution mean `m`, the true center the points were sampled around:

  C_mu = sum_i w_i (x_{i:lam} - m)(x_{i:lam} - m)^T,

with weights `w_i` summing to one (I will allow weights rather than a flat average — more on that shortly). Both are unbiased estimators of `C` if selection is random. But under *real* selection they say different things, and I should figure out which one I want by reasoning about what each measures, not by reaching for the familiar formula. `C_emp` measures the spread *within the surviving cluster*. `C_mu` measures the spread of the *selected steps* `x_{i:lam} - m` — the displacements from where I was to where the good points are.

Here is why that distinction matters, and I want a number rather than a story, so let me put a search on a slope and actually compute it. Take `f` locally linear with the downhill direction along coordinate 0, sample `lambda = 20` points from `N(0, I)`, keep the `mu = 10` with the largest projection on the slope, and average over many generations. I measure the variance each estimator assigns along the slope (which was `1` before selection) and along a neutral coordinate (which selection should leave at `1`). The numbers:

  along the slope:   C_emp keeps 0.343   |  C_mu keeps 0.999
  neutral coord:     C_emp keeps 0.899   |  C_mu keeps 0.999

`C_emp` collapses the slope direction to a third of its width — selection chopped the uphill tail, so the surviving cluster is narrow, and `C_emp` (this is the EDA / cross-entropy / EMNA estimator) reports that narrowness and shrinks `C` along *exactly* the direction I want to move; iterate it and the distribution collapses onto the slope and converges prematurely, worst with the small populations an expensive `f` forces on me. Worse, `C_emp` even shrinks the *neutral* coordinate to 0.90, a spurious artifact of referencing the moving cluster mean. `C_mu` keeps the slope variance at essentially `1.0`, and the decomposition shows why: the selected mean displaces by `0.767` along the slope, contributing `0.767^2 = 0.656` of variance about the *old* mean, and the within-cluster spread is the same `0.343` that `C_emp` saw — and `0.656 + 0.343 = 0.999`. Referencing the old mean recaptures the displacement *as variance*, exactly cancelling the within-cluster narrowing, so the productive direction is preserved rather than thrown away. (And on the neutral coordinate `C_mu` stays at `1.0`, doing nothing, as it must.) On top of that, sampling from `C_mu` next time tends to reproduce these displaced steps — principle (1), make the selected step more likely — and the cumulation machinery I add later will amplify that displacement instead of merely preserving it. The reference mean is the difference between a method that keeps the productive direction open and a method that strangles itself. So I take `C_mu`, with the *old* mean as reference.

About those weights: instead of a flat `1/mu` over the best `mu`, let me allow a decreasing weight vector `w_1 >= w_2 >= ... >= w_mu > 0` summing to one. Truncation (drop the worst `lambda - mu`) is already a selection; differential weights are a finer-grained selection that trusts the better-ranked steps more. The cost of weighting is that I am averaging fewer effectively-independent samples. Let me make that precise, because the number is going to reappear everywhere. If I average `mu` independent samples with weights `w_i`, the variance of the weighted mean is reduced by a factor I will call `mu_eff`,

  mu_eff = ( sum_i w_i )^2 / sum_i w_i^2 = 1 / sum_i w_i^2

(using `sum_i w_i = 1`), the variance-effective selection mass. For flat weights `w_i = 1/mu` this is `(mu * 1/mu)^2 / (mu * 1/mu^2) = 1/(1/mu) = mu` exactly; for skewed weights Cauchy-Schwarz gives `(sum w_i)^2 <= mu sum w_i^2`, i.e. `mu_eff <= mu`, with equality only when all weights are equal — so in general `1 <= mu_eff <= mu`. Let me put a real number on it for the defaults I will land on. With the log-decreasing raw weights `w'_i = ln((lambda+1)/2) - ln i` over the positive half, at `n = 10` (so `lambda = 4 + floor(3 ln 10) = 10`, `mu = 5`) the normalized positive weights come out `[0.456, 0.271, 0.162, 0.085, 0.026]`, and `1/sum w_i^2 = 3.17`. So out of `mu = 5` retained parents the weighting effectively uses about `3.2` independent samples' worth of information — the skew toward the best ranks costs me roughly a third of the nominal count. At `n = 40` (`lambda = 15`, `mu = 7`) the same computation gives `mu_eff = 4.54` out of `7`. The ratio `mu_eff/mu ~ 0.6` is stable across `n`, which I will lean on later. So `mu_eff` is "how many independent samples' worth of information" my weighting actually uses, and it will be the single most important derived quantity in calibrating every learning rate, because every estimate I form from the selected steps has variance set by `mu_eff`, not by `mu`.

Now, can I rely on a single generation's `C_mu`? No — and here is the wall. To get `C_mu` accurate enough that its condition number on the sphere is below, say, ten, I need `mu_eff` on the order of `10n`. That means a population of order `n` or more, which on an expensive `f` is exactly the budget I do not have, and which also defeats the point: I wanted *fast* search with small populations so that, for a fixed evaluation budget, I get *many* generations and thus many small reliable updates rather than a few large noisy ones. A one-shot per-generation estimate is unusable for small `mu_eff`. Wall.

The fix mirrors what I do everywhere I have a noisy per-step estimate: do not trust one generation, accumulate. Average the per-generation estimates `C_mu^{(i)}` over generations. A flat average over all past generations would work after enough of them, but it weights ancient generations — sampled from a long-gone distribution — as heavily as the current one, which is wrong because the geometry has moved. So use exponential smoothing, the same forgetting trick that turns a never-forgetting sum into a windowed estimate. With a learning rate `0 < c_mu <= 1`,

  C <- (1 - c_mu) C + c_mu C_mu = (1 - c_mu) C + c_mu sum_i w_i y_{i:lam} y_{i:lam}^T,

where `y_{i:lam} = (x_{i:lam} - m) / sigma` is the selected step in `sigma`-units (I have put `sigma` back). This is the rank-`mu` update: each generation it adds a rank-`min(mu,n)` correction built from all `mu` selected steps, and it decays the old `C` toward the new evidence. Unrolling it, `C` is an exponentially weighted average of all past per-generation estimates, so after about `1/c_mu` generations roughly 63% of its content has been refreshed — `1/c_mu` is the backward time horizon. What should `c_mu` be? Faster (larger `c_mu`) learns sooner but from noisier single-generation evidence; the noise floor is set by `mu_eff`, and the matrix lives in `~ n^2` dimensions, so a stable first-order choice is `c_mu ~ mu_eff / n^2`. That is the right shape: more reliable evidence (larger `mu_eff`) earns a faster rate; more parameters to fill (larger `n^2`) demands a slower one. Now small populations are fine, because I no longer need one generation to be reliable — I need many generations to accumulate, and small `lambda` *gives* me more generations per evaluation budget. The wall is gone, and notice the trade I made: I traded the impossible demand "estimate `C` from one generation" for the cheap demand "nudge `C` toward this generation's evidence."

There is a second viewpoint on this same update that I want to work out, because looking at the rank-`mu` term one direction at a time may show me something the all-at-once estimate hides. Forget estimating a whole matrix; think about building a covariance one direction at a time. If I have vectors `y_1, ..., y_k` then the random vector `sum_j N_j(0,1) y_j`, with independent standard normals, is distributed `N(0, sum_j y_j y_j^T)` — adding "line distributions" along the `y_j` builds up a covariance as a sum of outer products. A single line distribution `N(0,1) y` is `N(0, y y^T)`, the rank-one covariance that generates the vector `y` with maximum likelihood among all zero-mean normals. So if I take the simplest possible rank-`mu` update — `mu = 1`, a single selected step `y` per generation —

  C <- (1 - c_1) C + c_1 y y^T,

I am folding into `C` the maximum-likelihood rank-one term for the step I just found good, which directly raises the probability of producing that step again. Iterate it: `C^{(1)}` reproduces `y_1` more readily than `C^{(0)} = I` did, `C^{(2)}` reproduces `y_2` more readily than `C^{(1)}`, and the distribution progressively aligns itself with the cloud of selected steps. When the search distribution matches the distribution of selected steps, under random selection no further change happens in expectation — which is the stationarity I demanded. So rank-one and rank-`mu` are the same idea seen from two ends: estimate the whole thing from many steps at once, or fold in one step at a time. Good — consistent.

But staring at the rank-one term `y y^T`, I notice it throws away something I should not be throwing away. The outer product is even in `y`: `y y^T = (-y)(-y)^T`. The *sign* of the step is invisible to it. Yet the sign carries real information: if my last several mean-steps all pointed the same way, that is a strong signal there is a long favorable axis there, and I would like to stretch `C` along it aggressively; if they alternated, the directional information cancels and I should not. The per-generation outer products cannot see this because each is sign-blind. So before I square, let me *accumulate the signed steps* across generations and feed the accumulated direction into the rank-one update. Call this accumulated vector the evolution path `p_c`: a running, sign-aware sum of the recent mean-displacements. Exponentially smoothed, starting from zero,

  p_c <- (1 - c_c) p_c + (normalization) * (m_new - m_old) / (c_m sigma).

What normalization? I want a calibration tied to my stationarity principle: under random selection, where consecutive mean-steps are independent zero-mean `N(0,C)/sqrt(mu_eff)` vectors, the path `p_c` should itself stay distributed `N(0,C)` — neither growing nor shrinking on average — so that "the path is longer/shorter than `N(0,C)`-typical" is a meaningful, unbiased selection signal. Let me solve for the constant. The mean-step `m_new - m_old` over `c_m sigma` is the weighted average of `mu` selected `y`-steps, and a weighted average of independent `N(0,C)` vectors with weights summing to one has covariance `(sum w_i^2) C = C / mu_eff`; so `sqrt(mu_eff) * (m_new - m_old)/(c_m sigma) ~ N(0,C)` under random selection. Now I want constants `a, b` with `p_c <- a p_c + b * sqrt(mu_eff) (m_new - m_old)/(c_m sigma)` such that if `p_c ~ N(0,C)` going in, it comes out `~ N(0,C)`: that needs `a^2 + b^2 = 1` (variances of independent normals add). Taking `a = 1 - c_c` for the smoothing, I need `b = sqrt(1 - (1-c_c)^2) = sqrt(c_c(2-c_c))`. Folding in the `sqrt(mu_eff)`, the normalization is `sqrt(c_c (2-c_c) mu_eff)`.

I do not fully trust an algebra-only argument here, because there are two places I could have dropped a factor: the `sqrt(mu_eff)` from recombination, and the `a^2+b^2=1` balance. So I simulate the recurrence directly. Working in whitened coordinates (`C = I`), I feed in random-selection mean-steps `~ N(0, I/mu_eff)`, run `p <- (1-c) p + sqrt(c(2-c) mu_eff) * step` with `c = 0.3`, `mu_eff = 4`, `n = 8`, and measure the stationary covariance of `p` after burn-in. It comes out `diag = [1.00, 1.00, 1.01, 1.00, 1.00, 0.99, 1.01, 1.00]` — flat at `1`, so `p ~ N(0, I)` as intended, and `E||p|| = 2.744` against the target `E||N(0,I)|| = 2.742`. Now the control: if I had *forgotten* the `sqrt(mu_eff)` factor and used `b = sqrt(c(2-c))` (the naive "it's just exponential smoothing of unit-variance steps" guess), the simulated stationary `diag` comes out `[0.248, 0.248, ...]`, i.e. exactly `1/mu_eff = 0.25`. That confirms the `sqrt(mu_eff)` is load-bearing: without it the path is systematically a factor `sqrt(mu_eff)` too short, the length test reads "always too short," and `sigma` would shrink under pure noise — a stationarity violation. The factor is not cosmetic; it is what makes the recombination-shrunk mean-step into a full-variance path. So the normalization is `sqrt(c_c (2-c_c) mu_eff)`:

  p_c <- (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) * (m_new - m_old) / (c_m sigma),

and the rank-one covariance update uses the path:

  C <- (1 - c_1) C + c_1 p_c p_c^T.

Check the corner: `c_c = 1`, `mu_eff = 1` gives `p_c = (x_{1:lam} - m)/sigma`, a single step, and the path-based rank-one reduces to the plain single-step rank-one — consistent again. Why would the path help, beyond restoring the sign? Because correlation between consecutive steps should get *amplified* in the sum. If steps are perfectly aligned unit vectors along a genuine long axis, the path reaches its fixed point `p* = (1-c_c) p* + sqrt(c_c(2-c_c))`, i.e. `p* = sqrt(c_c(2-c_c))/c_c = sqrt((2-c_c)/c_c)`. Let me put numbers on that ratio of path-length to single-step-length: `c_c = 0.25 -> 2.65`, `c_c = 0.1 -> 4.36`, `c_c = 0.05 -> 6.24`, against `1/sqrt(c_c) = 2.0, 3.16, 4.47`. So the inflation is `sqrt((2-c_c)/c_c)`, asymptotically `1/sqrt(c_c)` for small `c_c`, and at the rates I will use it is a factor of several. Anti-correlated steps would deflate it by the inverse. The path represents a persistent axis far more accurately than any single noisy step, and the inflation acts like a free boost to the effective learning rate along that axis.

The claim I actually care about is operational: with `1/c_c` between `sqrt(n)` and `n`, a single long axis — a cigar `f(x) = x_0^2 + 10^6 sum_{i>=1} x_i^2` — should be learnable in `O(n)` evaluations even though the raw rate `c_1 ~ 2/n^2` looks far too slow. I do not want to take that on faith, so let me run the assembled algorithm (once I have it all) on the cigar and count. The result, at three dimensions:

  n=3:  solved (f < 1e-9) in   924 evals (132 generations)
  n=5:  solved              in  1896 evals (237 generations)
  n=8:  solved              in  3480 evals (348 generations)

The evaluation count goes roughly `924 -> 1896 -> 3480` as `n` goes `3 -> 5 -> 8`; per-dimension that is about `308, 379, 435` evals — growing slowly, consistent with the near-linear scaling the path is supposed to buy, and nowhere near the `n^2` that the bare `c_1` would suggest. And the diagnostic I really want: after solving, the longest eigen-axis of the learned `C` has `|component on coordinate 0| = 1.000` in every case, with `cond(C) ~ 10^6` — `C` has aligned its long axis with the cigar's long axis and matched its condition number. The path did the work the small learning rate alone could not. So I set `c_c ~ 4/n` (a time horizon `~ n/4`, in the right band), and `c_1 ~ 2/n^2` for the rank-one rate.

Now I have two covariance updates that capture different things. Rank-`mu` uses the whole population each generation — great when `mu_eff` is large, because there is a lot of within-generation evidence. Rank-one uses correlations *between* generations via the path — great when `mu_eff` is small, because then within-generation evidence is thin and the cross-generation signal carries the load. There is no reason to choose; in the positive-weight form I combine them, each with its own rate:

  C <- (1 - c_1 - c_mu) C + c_1 p_c p_c^T + c_mu sum_{i=1}^{mu} w_i y_{i:lam} y_{i:lam}^T.

The leading coefficient keeps the total weight at one so the update is, in expectation, stationary; with `c_1 = 0` it is pure rank-`mu`, with `c_mu = 0` pure rank-one. This is the core covariance adaptation.

But `C` alone does not control the overall scale of my steps, and I have two separate reasons to handle scale on its own. First, the optimal overall step length depends on `mu` / `mu_eff` (more parents means I can afford to probe further out and still land well after recombination shrinks the realized step by `~ sqrt(mu_eff)`), and the covariance update simply cannot represent that dependence — it adapts shape and per-direction scale, not the single global magnitude in the way the optimum demands. Second, and more sharply: the largest *reliable* learning rate for `C` is `~ mu_eff/n^2` or `~ 2/n^2`, far too slow to change the overall step length at the rate the sphere demands (where the optimal `sigma` must shrink by a factor of order `exp(1/4)` every `n` evaluations); whenever `mu_eff << n`, covariance adaptation cannot keep up with the needed change in scale. So I need a fast, separate `sigma` controller — and I already know the derandomized principle to use: read the scale off the realized steps, not by mutating `sigma`.

How to read it off the *path*, to exploit the same sign-amplification trick? Use the length of an evolution path as the goodness measure for the overall step length. The intuition is clean: if successive steps point the same way, the path is long, and I am being inefficient — I could have covered the same ground in fewer, longer steps, so `sigma` should *increase*. If successive steps cancel, the path is short — I am overshooting and stepping back and forth, so `sigma` should *decrease*. The neutral, desired situation is steps roughly perpendicular in expectation, i.e. uncorrelated, which is exactly what holds under random selection. So the rule is: compare the realized path length to its expected length under random selection, and move `sigma` to push it toward neutral.

There is one defect I must fix before this works as a *length* measure, though. The path `p_c` I built for the covariance is distributed `N(0,C)` under random selection, so its *expected length depends on its direction* — it is long along the long axes of `C` and short along the short ones. If I judge "long vs short" by `||p_c||`, I will confound "the search is aligned" with "the search happens to be moving along a long axis of `C`," and `sigma` will react to the shape of `C` rather than to the alignment of the steps. I need a path whose length is direction-independent. So build a *conjugate* path, the same accumulation but with each incoming step *whitened* by `C^{-1/2}`:

  p_sigma <- (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * C^{-1/2} (m_new - m_old) / (c_m sigma).

The whitening `C^{-1/2} = B D^{-1} B^T` is exactly the transform that turns `N(0,C)` into `N(0,I)`: read right to left, `B^T` rotates the principal axes of `C` onto the coordinate axes, `D^{-1}` rescales every axis to unit length, `B` rotates back so the principal axes are not permanently rotated and consecutive whitened steps remain comparable. After whitening, under random selection `p_sigma ~ N(0, I)` regardless of `C`, so its expected length is the constant `E||N(0,I)||`, call it `chi_n`, with no directional dependence. Now I have a clean length test. (I should keep straight that `p_c`, used for the *covariance*, must *not* be whitened — there I want the step in the natural coordinates so `p_c p_c^T` stretches `C` along the actual selected direction; only the *step-size* path `p_sigma` is whitened, because there I want a pure length comparison.)

The update for `sigma` is then a multiplicative push of `||p_sigma||` toward `chi_n`:

  sigma <- sigma * exp( (c_sigma / d_sigma) * ( ||p_sigma|| / chi_n - 1 ) ).

If `||p_sigma|| > chi_n` (path longer than random-selection-typical, steps aligned) the exponent is positive and `sigma` grows; if shorter, `sigma` shrinks; if equal, `sigma` holds. The damping `d_sigma ~ 1` is the explicit change-rate control from derandomization principle (2) — it lets me probe with informative steps but bound how fast `sigma` moves, which matters because `sigma` is a single scalar driving the entire scale and I cannot afford it to oscillate. And check stationarity, principle (3): under random selection `||p_sigma||/chi_n - 1` has mean zero (since `E||p_sigma|| = chi_n`), so `E[ln sigma_new | sigma] = ln sigma` — the update is unbiased *on the log scale*. It is *not* unbiased for `sigma` itself — by Jensen the expectation of `sigma` drifts slightly upward — but unbiasedness of `ln sigma` is the right invariant: `sigma` lives on a multiplicative scale, and a small upward bias on noisy problems is the safe direction (better to keep moving than to collapse). Good, this is the cumulative step-size adaptation, and it is independent of the covariance machinery — it would work on its own as a path-length step-size controller.

What is `chi_n` exactly? It is `E||N(0,I)||` in `n` dimensions, which is `sqrt(2) * Gamma((n+1)/2) / Gamma(n/2)`. (Quick sanity check that I have the right object: Monte-Carlo `E||N(0,I)||` over 200k draws gives `2.128` at `n=5` and `4.415` at `n=20`, matching the Gamma formula's `2.128` and `4.417`. Good, it is the mean norm, not the median or the mode.) I do not want to call the Gamma function every generation; the standard asymptotic expansion is

  chi_n = sqrt(n) * (1 - 1/(4n) + 1/(21 n^2) + ...).

Before trusting it I should check the error at the *small* `n` where the asymptotic is most strained, not just large `n`. Tabulating exact-vs-approx: `n=2` gives `1.2533` vs `1.2543` (rel err `7.7e-4`), `n=3` rel err `7.0e-4`, `n=5` rel err `3.9e-4`, `n=10` rel err `1.3e-4`, `n=20` rel err `3.7e-5`. So even at `n=2` the closed form is within `0.08%`, and it tightens fast — well below the noise in `||p_sigma||` itself. That is plenty; I will use the closed form.

Let me reconsider the mean update with all this in hand, because I glossed it. The new mean is the weighted recombination of the selected points,

  m <- m + c_m * sigma * sum_{i=1}^{mu} w_i y_{i:lam},   y_{i:lam} = (x_{i:lam} - m)/sigma,

with a mean learning rate `c_m`, default `1`, so the mean just moves to the weighted average of the `mu` best (the `-m` cancels). Why weighted truncation rather than, say, taking only the single best (`mu = 1`) or all of them? The single best is high-variance — one lucky sample yanks the mean. All of them includes the bad uphill points and pulls the mean backward. The weighted average of the top `mu ~ lambda/2` with decreasing weights is the bias-variance sweet spot: it averages out the per-sample noise (variance reduced by `mu_eff`) while still being a genuine selection (the worst half discarded, the better-ranked trusted more). The raw weights I will use over all ranks are log-decreasing,

  w'_i = ln((lambda + 1)/2) - ln i,   i = 1, ..., lambda.

The positive half is normalized to sum to one and drives recombination. But the ranking tells me more than "these are good": it also tells me which directions were bad. If I simply throw away the worst half, I miss a chance to *remove* variance in directions that consistently produce poor samples. So for the covariance update I can keep the full `lambda`-weight vector and let the worst ranks carry negative weights. That immediately raises a danger: a negative outer product can destroy positive definiteness if it subtracts too much along one direction. I need two safeguards. First, scale the total negative mass by

  alpha_minus = min(1 + c_1/c_mu, 1 + 2 mu_eff_minus/(mu_eff + 2), (1 - c_1 - c_mu)/(n c_mu)),

where `mu_eff_minus` is the variance-effective mass of the raw negative tail. The first bound aims to make `c_1 + c_mu sum_i w_i` vanish so the covariance has no net decay; the second keeps the negative mass commensurate with the information in the negative tail; the third guarantees enough remaining variance for positive definiteness. Second, in each generation rescale each negative term by its realized Mahalanobis length:

  w_i^o = w_i                       if w_i >= 0,
          w_i n/(||C^{-1/2} y_i||^2 + eps)   if w_i < 0.

Now a bad direction can shrink `C`, but a single long bad vector cannot subtract an uncontrolled amount. With these active weights the covariance accounting is no longer `sum w = 1`; the leading coefficient must be

  1 + c_1 delta_h - c_1 - c_mu sum_{i=1}^{lambda} w_i,

and the rank-`mu` term is really the full weighted sum `sum_i w_i^o y_i y_i^T`. In the positive-only special case, `sum_i w_i = 1`, the negative tail is zero, and this collapses back to the simpler formula above.

Now the defaults, which I refuse to pull from a hat — they have to fall out of the structure. The population: `lambda = 4 + floor(3 ln n)`, a slow logarithmic growth, because I argued small populations are *good* (more generations per budget) and only enough offspring are needed to get a usable selection signal; the logarithm is the empirically robust compromise. The positive parents are the ranks with positive raw log weights, which for this profile is `mu = floor(lambda/2)`. From the positive weights I compute `mu_eff = (sum w_i)^2 / sum w_i^2`; from the negative raw tail I compute `mu_eff_minus` for the active-weight bound. The step-size cumulation rate `c_sigma = (mu_eff + 2)/(n + mu_eff + 5)` — order `1/n` for a time horizon between `sqrt(n)` and `n`, rising with `mu_eff` because more reliable mean-steps justify faster cumulation. The damping `d_sigma = 1 + 2 max(0, sqrt((mu_eff-1)/(n+1)) - 1) + c_sigma`, which is `~ 1` for small `mu_eff` and grows only when `mu_eff` is large relative to `n` (then the mean-steps are big and `sigma` must be damped harder to avoid overshoot). The covariance cumulation rate `c_c = (4 + mu_eff/n)/(n + 4 + 2 mu_eff/n)`, again `~ 1/n`, the right band for learning a single axis in `O(n)`. The rank-one rate `c_1 = 2 / ((n+1.3)^2 + mu_eff)` — essentially `2/n^2`, the largest rate at which a rank-one update stays stable, with the `+mu_eff` and the `1.3` offset as low-dimensional corrections. The rank-`mu` rate `c_mu = min(1 - c_1, 2 (mu_eff - 2 + 1/mu_eff)/((n+2)^2 + mu_eff))`, with an implementation-level epsilon in the cap, is `~ mu_eff/n^2` for large `n` exactly as I derived, capped so the covariance update never over-decays `C`. The `mu_eff - 2 + 1/mu_eff` numerator vanishes at `mu_eff = 1` (no rank-`mu` learning when there is effectively one parent — correct, there is no within-generation spread to estimate) and grows toward `mu_eff` for large `mu_eff`. Every constant traces to "fast enough to learn, slow enough to stay reliable, scaled by `mu_eff` and `n`."

One more guard I should add before writing it down, because the path-based rank-one update can misbehave when the step size is far too small. If `sigma` is initialized way below the scale of the problem, the search makes a near-linear run of long aligned steps, the path `p_sigma` gets very long, `sigma` ramps up fast — and meanwhile `p_c p_c^T` would inject a huge spurious axis into `C` from those transient aligned steps. So stall the `p_c` accumulation whenever `||p_sigma||` is anomalously large. Concretely, a Heaviside switch `h_sigma = 1` only if

  ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2(g+1)})  <  (1.4 + 2/(n+1)) chi_n,

else `0`, where the denominator de-biases the path length for the early generations: starting from `p_sigma = 0` the exponential sum has not saturated yet, so the raw `||p_sigma||` is systematically short at small `g`. Let me check that the proposed denominator is exactly the right correction. Each component of `p_sigma` at generation `g` is a sum of independent terms `sqrt(c(2-c)mu_eff)(1-c)^k * step_k` with `Var(step) = 1/mu_eff`, so its variance is `c(2-c) sum_{k=0}^{g} (1-c)^{2k} = c(2-c) (1 - (1-c)^{2(g+1)})/(1 - (1-c)^2) = 1 - (1-c)^{2(g+1)}` (the `c(2-c)` cancels the `1-(1-c)^2` denominator). So per-component variance should equal `1 - (1-c)^{2(g+1)}`, and dividing `||p_sigma||` by `sqrt` of that restores unit variance. Simulating the recurrence with `c=0.3`, `mu_eff=4`, and measuring the per-component variance generation by generation: `g=0` measures `0.510` vs predicted `0.510`; `g=1` `0.760` vs `0.760`; `g=3` `0.942` vs `0.942`; `g=7` `0.997` vs `0.997` — exact agreement, and it saturates to `1` by about `g=8`. So the de-bias is precisely the geometric-sum transient, the same bias-correction one needs whenever an exponential accumulator starts from zero. When `h_sigma = 0`, `p_c` is not updated with the new step, and the covariance update compensates with `delta_h = (1 - h_sigma)c_c(2-c_c)` inside the leading coefficient `1 + c_1 delta_h - c_1 - c_mu sum_i w_i`. It is a small correction, on most generations `h_sigma = 1` and it does nothing, but it prevents the one nasty failure mode of a badly-initialized `sigma`.

Let me sanity-check the whole thing against the invariances I demanded, since those were hard constraints. Rank-only: every place I used `f` was through the ordering `f(x_{1:lam}) <= ... <= f(x_{lam:lam})`, never a magnitude — so monotone-transform invariance holds by construction; I can read the code and confirm no line touches an `f`-value except `sorted(..., key=lambda s: s[1])`.

Rotation is the one I should be careful about, because I want to state it at the right strength. The update is written in terms of the geometric vectors `y_{i:lam}` and `p_c` and the matrix `C`, and `sigma`'s path is whitened by `C^{-1/2}` defined from `C`'s eigendecomposition — no privileged coordinate system appears, so the *idealized* update commutes with rotation: replacing `f` by `f∘R^{-1}` and `C^{(0)}` by `R C^{(0)} R^T` should leave the `f`-trajectory unchanged. My first instinct was to claim the trajectory is *pathwise identical* under a shared random stream. Let me test that. I run the axis-aligned ellipsoid `f = 1/2 sum 10^{6(i/(n-1))} x_i^2` and a randomly rotated copy, feeding the *same* standard normals `z` into both, with the rotated run started at `R m_0` and `C_0 = R I R^T`. The two `f(mean)` trajectories *diverge* — by the third generation they are completely different. The reason, when I trace it, is `eigh`: `numpy.linalg.eigh(R C R^T)` does not return `R @ eigvecs(C)` with matching column signs and ordering, so `y = B D z` is not the `R`-image of the base run's `y` even though the covariances are `R`-related. The eigenvector gauge breaks the naive coupling. So the pathwise claim is *false for the concrete sampler*, and I should not assert it.

What is true is invariance *in distribution*. I run 40 seeds of each and compare the convergence distributions: axis-aligned ellipsoid reaches median `log10 f(mean) = -0.73` (IQR `-1.87..0.18`), the rotated copy `-0.48` (IQR `-1.19..0.12`) — statistically indistinguishable, heavily overlapping. So rotation invariance holds in the sense that matters (the algorithm has no preferred orientation; performance on a problem and on its rotated copy is the same distribution), even though a single trajectory is not the literal `R`-image of another because of the eigendecomposition gauge. That is the honest statement, and it is enough: it is exactly what lets a result on a separable test transfer to its non-separable rotation.

And the inverse-Hessian target I set at the start — does the assembled algorithm actually hit it? I run it on the ill-conditioned ellipsoid (condition number `10^6` across `n=4`) to convergence and compare the learned `C` to `H^{-1}`, both normalized by trace:

  normalized diag(C):     [0.9857, 0.0143, 1.00e-4, 0.0e0]
  normalized diag(H^-1):  [0.9900, 0.0099, 1.00e-4, 0.0e0]
  ||C_norm - H^{-1}_norm|| = 0.046

`C` has tracked `H^{-1}` across six orders of magnitude in the eigenvalues; the search distribution learned the inverse Hessian from ranks alone, turning the ellipsoid into (almost) a sphere. (The cigar run earlier showed the same thing from the eigenvector side: longest `C`-axis aligned with the long problem axis, `cond(C) ~ 10^6`.) That is the quasi-Newton analogue I was aiming at, with no derivatives — and it is checked, not asserted.

Now let me write the generation as the algorithm I would actually ship, sampling with the eigendecomposition, the mean recombination, the whitened path and the `sigma` update, the unwhitened path with the Heaviside guard, and the combined covariance update — this fills the single empty slot in the ask/tell harness, the rule that turns one ranked generation into the next `(m, sigma, C)`:

```python
import math
import numpy as np

_EPS = 1e-8
_SIGMA_MAX = 1e32


class CMA:
    """(mu/mu_w, lambda)-CMA-ES: adapt the mean, the overall step size sigma,
    and the full covariance C of the search distribution N(m, sigma^2 C) from
    ranked samples, using only the ranking of f-values."""

    def __init__(self, mean, sigma, population_size=None, seed=None):
        n = len(mean)
        self.dim = n
        self.mean = np.asarray(mean, dtype=float).copy()
        self.sigma = float(sigma)
        self.C = np.eye(n)
        self.p_c = np.zeros(n)            # evolution path for C (sign-aware, unwhitened)
        self.p_sigma = np.zeros(n)        # conjugate path for sigma (whitened -> N(0,I))
        self.g = 0
        self.rng = np.random.RandomState(seed)
        self.B = None
        self.D = None

        # population, full log-weight vector, and positive-weight effective mass
        self.population_size = population_size or 4 + math.floor(3 * math.log(n))
        self.mu = self.population_size // 2
        weights_prime = np.array([
            math.log((self.population_size + 1) / 2) - math.log(i + 1)
            for i in range(self.population_size)
        ])
        me = (np.sum(weights_prime[: self.mu]) ** 2) / np.sum(weights_prime[: self.mu] ** 2)
        me_minus = (np.sum(weights_prime[self.mu :]) ** 2) / np.sum(weights_prime[self.mu :] ** 2)
        self.mu_eff = me

        alpha_cov = 2.0
        self.c_1 = alpha_cov / ((n + 1.3) ** 2 + me)
        self.c_mu = min(
            1 - self.c_1 - 1e-8,
            alpha_cov * (me - 2 + 1 / me) / ((n + 2) ** 2 + alpha_cov * me / 2),
        )
        min_alpha = min(
            1 + self.c_1 / self.c_mu,
            1 + 2 * me_minus / (me + 2),
            (1 - self.c_1 - self.c_mu) / (n * self.c_mu),
        )
        positive_sum = np.sum(weights_prime[weights_prime > 0])
        negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))
        self.weights = np.where(
            weights_prime >= 0,
            weights_prime / positive_sum,
            min_alpha * weights_prime / negative_sum,
        )

        # learning rates / damping, all derived from n and mu_eff
        self.c_m = 1.0
        self.c_sigma = (me + 2) / (n + me + 5)
        self.d_sigma = 1 + 2 * max(0.0, math.sqrt((me - 1) / (n + 1)) - 1) + self.c_sigma
        self.c_c = (4 + me / n) / (n + 4 + 2 * me / n)
        self.chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))   # E||N(0,I)||

    def _eigen_decomposition(self):
        if self.B is not None and self.D is not None:
            return self.B, self.D
        self.C = (self.C + self.C.T) / 2
        D2, B = np.linalg.eigh(self.C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self.C = B @ np.diag(D ** 2) @ B.T
        self.B, self.D = B, D
        return B, D

    def ask(self):
        B, D = self._eigen_decomposition()
        z = self.rng.randn(self.dim)                            # z ~ N(0, I)
        y = B @ (D * z)                                          # y = B D z ~ N(0, C)
        return self.mean + self.sigma * y                       # x ~ N(m, sigma^2 C)

    def tell(self, solutions):
        assert len(solutions) == self.population_size
        n = self.dim
        self.g += 1
        solutions = sorted(solutions, key=lambda s: s[1])       # rank by f, best first
        x = np.array([s[0] for s in solutions])
        y = (x - self.mean) / self.sigma                        # selected steps in sigma-units

        B, D = self._eigen_decomposition()
        self.B, self.D = None, None                             # C changes below; cache expires
        C_invsqrt = B @ np.diag(1.0 / D) @ B.T                  # C^{-1/2} = B D^{-1} B^T

        # mean: weighted recombination of the mu best (reference = old mean)
        y_w = np.sum(y[: self.mu].T * self.weights[: self.mu], axis=1)
        self.mean = self.mean + self.c_m * self.sigma * y_w

        # step-size: conjugate (whitened) path -> length test against chi_n
        self.p_sigma = ((1 - self.c_sigma) * self.p_sigma
                        + math.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
                        * (C_invsqrt @ y_w))
        norm_ps = np.linalg.norm(self.p_sigma)
        self.sigma *= math.exp((self.c_sigma / self.d_sigma) * (norm_ps / self.chi_n - 1))
        self.sigma = min(self.sigma, _SIGMA_MAX)

        # covariance: sign-aware (unwhitened) path for rank-one; Heaviside guard
        h_sigma = (norm_ps / math.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.g + 1)))
                   < (1.4 + 2 / (n + 1)) * self.chi_n)
        self.p_c = ((1 - self.c_c) * self.p_c
                    + h_sigma * math.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w)
        delta_h = (1 - h_sigma) * self.c_c * (2 - self.c_c)

        # Active covariance weights: negative weights are length-normalized in C^{-1/2} coordinates.
        w_io = self.weights * np.where(
            self.weights >= 0,
            1.0,
            n / (np.linalg.norm(C_invsqrt @ y.T, axis=0) ** 2 + _EPS),
        )

        rank_one = np.outer(self.p_c, self.p_c)                 # c_1 term: aligned-axis stretch
        rank_mu = np.einsum('i,ij,ik->jk', w_io, y, y)          # c_mu term, all lambda ranks
        old_weight = 1 + self.c_1 * delta_h - self.c_1 - self.c_mu * np.sum(self.weights)
        self.C = (old_weight * self.C
                  + self.c_1 * rank_one
                  + self.c_mu * rank_mu)
```

Let me trace the causal chain back to make sure every piece is load-bearing. I started stuck: black-box, expensive, ill-conditioned, non-separable, rank-only — and I needed a sampler that learns the problem's geometry from ranks. The maximum-entropy choice was a Gaussian, and the ideal covariance was the inverse Hessian, so the prize was learning `C`. Mutative self-adaptation could learn it in principle but selected strategy parameters indirectly through a noisy signal and needed huge populations — wall — so I took the derandomized principle of reading the adaptation off the *realized* steps directly. Estimating `C` from one generation forced the reference-mean question, and referencing the *old* mean (selected-steps estimator) rather than the selected-points' own mean was the hinge that grows variance in the productive direction instead of collapsing it. One generation was too unreliable for small populations — wall — so I accumulated with exponential smoothing into the rank-`mu` update, which let small populations win by giving more generations. The sign-blindness of the outer product — wall — pushed me to accumulate signed steps into an evolution path and use it for a rank-one update, which also amplifies persistent axes and learns a cigar in `O(n)`. The worst ranks carry information too, but negative covariance weights need bounded total mass and Mahalanobis length scaling so they shrink bad directions without breaking positive definiteness. The covariance still could not set the overall scale fast enough — wall — so I added cumulative step-size control: a *conjugate*, whitened path whose length is direction-independent, compared against `E||N(0,I)|| = chi_n`, driving `sigma` multiplicatively with a damping for explicit change-rate control, and unbiased on the log scale for stationarity. The Heaviside guard catches the badly-initialized-`sigma` transient, and the coefficient `1 + c_1 delta_h - c_1 - c_mu sum_i w_i` keeps the covariance accounting correct for both positive-only and active weights. Every learning rate scaled by `mu_eff` and `n` from "fast enough to learn, slow enough to stay reliable." And the things I demanded at the outset I checked rather than assumed: rank-only holds because no line reads an `f`-magnitude; rotation invariance holds *in distribution* (the `eigh` gauge breaks pathwise coupling, but axis-aligned and rotated ellipsoids converge to the same distribution over 40 seeds); the cigar is learned in roughly linear evaluations with `C`'s long axis snapping onto the problem's; and on the ill-conditioned ellipsoid the normalized `C` matches `H^{-1}` to within `0.05` across six orders of magnitude. Quasi-Newton behavior on a black box, from ranks alone — and now verified, not just argued.
