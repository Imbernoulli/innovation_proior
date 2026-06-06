OK, let me start from where it actually hurts. I have a function `y(x)` over a box of design variables — say a handful to a dozen of them — and I want its global minimum. The catch that changes everything: one evaluation of `y` is *expensive*. It's an automotive crash simulation that takes the better part of a day, or a finite-element / circuit code that takes minutes to hours per run. So the budget isn't CPU-seconds spent thinking; it's the number of times I'm allowed to call `y`. Tens of calls, maybe. Not thousands. And the code is a black box: no gradient comes out of it, the surface is probably nonlinear and quite possibly multimodal, and I have no analytic form — if I did, I wouldn't have paid to build the expensive simulator.

So the real objective behind the objective is: locate the global min in as few evaluations as possible, and — this would be lovely — have some self-contained signal telling me when there's nothing left to gain, so I can stop. A method that burns hundreds of calls, or that quietly settles into a local minimum and declares victory, is useless here.

Let me try the obvious things first and watch them break, because the failure modes are what point at the answer.

Grid search? In `k` dimensions a grid with `m` points per axis is `m^k` evaluations. For `k=6` and even a coarse `m=5` that's 15625 calls of a 20-hour simulation. Dead on arrival. Random search has the same curse with worse constants — it spreads the budget uniformly and ignores everything it learns. A multistart local optimizer (gradient descent from many seeds) is wrong on two counts: there's no gradient, and even with finite differences each local run alone eats more evaluations than my whole budget, and on a multimodal surface it just finds whatever basin its seed fell into. Genetic algorithms / branch-and-bound *on the true function* — same story, they're designed for cheap functions and assume you can afford thousands of evaluations. None of these "use" the expensive data; they keep re-querying.

So the thing I keep circling back to: every one of those calls is gold, and I'm throwing the information away. What I should do is *build a model of `y` from the calls I've made* and let that cheap model decide where to spend the next expensive call. Fit a surface, then reason about the surface for free.

The classical version of that is a response surface (RSM, Box–Hunter–Hunter 1978): assume `y(x_i) = Σ_h β_h f_h(x_i) + ε_i` with the `ε_i` independent zero-mean noise of variance σ², least-squares the β's, and optimize the fitted polynomial. Let me actually try to use this and see where it cracks. First crack: I have to *pick* the regressors `f_h`. A quadratic? But the whole reason the expensive code exists is that I don't know the functional form — if it's wigglier than quadratic, a quadratic systematically misses, and a flexible basis with many terms needs many evaluations to pin down its many coefficients. Second crack, and this one is deeper: the independent-noise assumption. My code is *deterministic*. Run it twice at the same `x`, get the identical number. There is no measurement noise. So what is `ε(x)`? It's the part of `y(x)` my regressors failed to capture — the left-out terms in `x`. And those left-out terms are a *continuous function of `x`*. Stare at that for a second. If `ε(x)` is continuous, then `ε(x_i)` and `ε(x_j)` for two nearby points are *nearly equal*. The residuals are correlated — strongly when the points are close, weakly when far apart — and treating them as independent draws is just structurally false for a deterministic code. That's not a nuisance; that's the whole opening. The regression's own logic (independent errors) is wrong precisely in a way I can exploit.

And because the errors are taken as independent noise, the fitted RSM doesn't pass through my data points and it can't tell me where it's ignorant. It gives a point prediction and a global σ², not a *local* uncertainty. But "where am I ignorant" is exactly the quantity I'll need to decide where to explore. So plain regression fails me twice: false correlation model, no honest local uncertainty.

Let me follow the correlation thread, because it feels load-bearing. If the residual is a correlated continuous field, then there's a body of theory built for exactly that: kriging, out of geostatistics (Matheron 1963, *Principles of geostatistics*; Krige; surveyed by Cressie). There, an unknown spatial field (ore grade across a deposit) is modeled as one realization of a random process whose correlation decays with distance, and you derive the best linear unbiased predictor at any unsampled location *together with its mean-squared prediction error*. Two things jump out. One: kriging *interpolates* — the predictor reproduces the data exactly at sampled points, which is precisely right for a deterministic code (once I've evaluated `y(x_i)`, I *know* it, there should be zero uncertainty there). Two: it hands me a prediction *and* a prediction-error at every point. That second output is the thing RSM couldn't give me. A surrogate that emits both `μ(x)` and `σ(x)` — the mean to exploit, the spread to know where to explore.

Sacks, Welch, Mitchell & Wynn (1989) carried this over to deterministic computer experiments (DACE — Design and Analysis of Computer Experiments). Let me build that surrogate concretely, because I'll be deriving everything on top of it.

Model the function as a constant mean plus a correlated Gaussian random field:

  y(x_i) = μ + ε(x_i),  ε(x) zero-mean Gaussian, Var = σ².

Why a *constant* mean μ and not a regression trend `Σ β_h f_h`? Because I'm about to put all the modeling power into the correlation structure, and that turns out to be expressive enough that I don't need to guess basis functions — and dropping the regressors means fewer parameters to fit on a tiny sample, which is exactly the currency I'm short on. Good. Now the correlation. I want `Corr → 1` for near points and `→ 0` for far points, and I want it to encode two facts about each coordinate separately: how *active* that variable is (does the function change a lot as I move along it?) and how *smooth* it is. Write the correlation as decaying in a weighted distance:

  d(x_i, x_j) = Σ_h θ_h |x_{ih} − x_{jh}|^{p_h},  θ_h ≥ 0,  p_h ∈ [1,2],
  Corr(ε(x_i), ε(x_j)) = exp(−d(x_i, x_j)).

`exp(−d)` is 1 when the points coincide (d=0) and decays to 0 as they separate — the shape I wanted. The `θ_h` is the *activity* of variable `h`: large `θ_h` means the correlation falls off fast along coordinate `h`, i.e. the function wiggles fast there and is sensitive to it; tiny `θ_h` means that variable barely matters. The exponent `p_h` is *smoothness*: `p_h = 2` gives a locally smooth (parabolic-at-the-origin) field, `p_h` near 1 a rougher, more jagged one. Letting `p_h` range over [1,2] interpolates between rough and smooth per coordinate. That's `2k+2` parameters in all — μ, σ², the k θ's, the k p's.

Now fit them. The data `y = (y(x_1),...,y(x_n))` is a draw from a multivariate normal with mean `1·μ` (the all-ones vector times μ) and covariance `σ² R`, where `R` is the n×n correlation matrix with `R_{ij} = Corr(ε(x_i), ε(x_j))`. So the likelihood is

  (2π)^{-n/2} (σ²)^{-n/2} |R|^{-1/2} exp[ −(y − 1μ)' R⁻¹ (y − 1μ) / (2σ²) ].

Here's a nice economy: for *fixed* correlation parameters (the θ's and p's, which fix R), I can solve for μ and σ² in closed form. Take the log-likelihood, differentiate w.r.t. μ: the only μ-dependence is in the quadratic form `(y−1μ)'R⁻¹(y−1μ)`, whose derivative set to zero gives the generalized-least-squares mean

  μ̂ = (1' R⁻¹ y) / (1' R⁻¹ 1).

Differentiate w.r.t. σ² and set to zero:

  σ̂² = (y − 1μ̂)' R⁻¹ (y − 1μ̂) / n.

Substitute both back and what's left is a *concentrated* log-likelihood in just the θ's and p's, which I maximize numerically. So instead of fitting many regression coefficients I'm fitting a handful of correlation parameters — far cheaper in evaluations.

Now the payoff: prediction at a new point `x*`. Let `r` be the n-vector of correlations between `x*` and the sampled points, `r_i = Corr(ε(x*), ε(x_i))`. The best linear unbiased predictor is

  ŷ(x*) = μ̂ + r' R⁻¹ (y − 1μ̂).

Let me sanity-check the interpolation claim, because it's the whole reason I switched to kriging. Put `x* = x_i`, one of the sampled points. Then `r` equals the i-th column of `R`, so `r' R⁻¹ = e_i'` (the i-th unit row vector, since `R⁻¹` times the i-th column of `R` is `e_i`). Thus `ŷ(x_i) = μ̂ + e_i'(y − 1μ̂) = μ̂ + (y_i − μ̂) = y_i`. It reproduces the data exactly. Good — deterministic-consistent.

And the uncertainty. The mean-squared error of this predictor is

  s²(x*) = σ²[ 1 − r' R⁻¹ r + (1 − 1' R⁻¹ r)² / (1' R⁻¹ 1) ].

Check it at a sampled point `x* = x_i`: there `r' R⁻¹ r = r' e_i = r_i = R_{ii} = 1`, and `1' R⁻¹ r = 1' e_i = 1`, so the bracket is `1 − 1 + (1−1)²/(...) = 0`. `s² = 0` at every sampled point — exactly zero uncertainty where I've evaluated, as a deterministic function demands. Far from all data, `r → 0`, so the bracket → `1 − 0 + (1)²/(1'R⁻¹1)` ≈ 1 and `s² ≈ σ²` — full prior uncertainty out in the unexplored wilderness. The RMSE `s(x*) = √(s²)` is my honest local error bar. This is precisely the `μ(x)` AND `σ(x)` surrogate the problem was crying out for.

(There's a tidier way to see the predictor too. Pretend I've evaluated a pseudo-observation `(x*, y*)` and write down the augmented likelihood for the n+1 points. The only `y*`-dependent part of the augmented quadratic form works out to `(1/(1 − r'R⁻¹r))(y*−μ̂)² − (2 r'R⁻¹(y−1μ̂)/(1−r'R⁻¹r))(y*−μ̂) + const`. Maximize over `y*` — set the derivative to zero: `(2/(1−r'R⁻¹r))(y*−μ̂) − 2 r'R⁻¹(y−1μ̂)/(1−r'R⁻¹r) = 0`, so `y* = μ̂ + r'R⁻¹(y−1μ̂)`, the same predictor. The interpretation is cute: the prediction is the value that, if I'd observed it, would be *most consistent* with the correlated data I already have.)

Right. I have a cheap surrogate that interpolates and reports its own ignorance. Now the actual question: **where do I sample next?** This is where it would be easy to do something dumb, so let me reason it out rather than guess a rule.

Dumb idea #1, pure exploitation: fit the surrogate, jump to the `x` that minimizes `ŷ(x)`, evaluate there, refit, repeat. Walk through what happens on a multimodal surface. The surrogate interpolates my data, so its minimum sits at — or very near — the lowest *data point* I happen to have, i.e. in whatever basin I've already sampled into. I jump there, evaluate, the new point pins the surrogate down locally, and the next minimum is again right beside it. I spiral into the nearest local minimum and never learn that a deeper basin sits across the box where I've never looked. The minimum of a fitted surface is a *local* minimum of the data. Pure exploitation is not global optimization — full stop. That's the motivating failure: it's exactly the trap that makes naive surrogate-chasing useless.

Dumb idea #2, pure exploration: sample wherever `s(x)` is largest — the point I know least about. That fills in the map, sure, but it spends my entire microscopic budget charting empty corners of the box and essentially never zeroes in on the optimum. Also useless.

So neither pole works, and the truth is in between: I need a single figure of merit that is large where `ŷ` is low (worth exploiting) *and* large where `s` is high (worth exploring), and that trades the two off **on its own**, without me hand-tuning some exploit/explore knob per problem. The trade-off has to be intrinsic to the criterion, or I'm back to babysitting.

What has the literature tried for the "where next" rule? Kushner (1964) modeled a 1-D objective as a Wiener process and sampled to maximize the *probability of improvement*: `P(Y(x) < f_min)`, the chance the new value beats the current best `f_min`. That's a real step — it uses the surrogate's distribution at `x`, not just its mean. But let me see what PI actually rewards. It counts *whether* I improve, not *by how much*. So a point sitting just barely below the incumbent, with tiny `s`, where improvement is nearly certain but minuscule, scores almost 1 — while a point with a long lower tail that *could* improve by a lot but only with moderate probability scores less. PI hugs the incumbent and is biased toward small, near-certain exploitation gains. To make it explore you have to inflate the target (`P(Y < f_min − ξ)` with a hand-set ξ) — back to a per-problem knob. And it was posed in 1-D on a Wiener model. The right instinct is there; the criterion is just measuring the wrong thing.

The fix is staring at me from PI's own flaw. If "*whether* you improve" loses the magnitude, then score by the *amount* of improvement. Define the improvement random variable

  I(x) = max(f_min − Y(x), 0),

where `Y(x) ~ Normal(ŷ(x), s²(x))` is my surrogate's belief about the unknown value at `x`. `I` is the shortfall below the incumbent if `x` turns out better, and 0 if it doesn't — it's literally how much I'd gain. Now weight every possible improvement by its probability density and add them up: take the expectation. This is the *expected* improvement,

  E[I(x)] = E[ max(f_min − Y(x), 0) ],

and it's the natural object — Mockus, Tiesis & Zilinskas (1978) posed global optimization in exactly this random-function, expected-gain framework. The reason this is the right criterion and PI isn't: PI integrates the density over the improving region (probability mass only); EI integrates `(improvement) × (density)`, so it credits both *how likely* an improvement is and *how big* it would be. That's the exploit/explore balance built straight into one number. What was missing before was a surrogate concrete enough to actually evaluate `E[I]` on, and a way to maximize the (nasty, multimodal) criterion reliably — and I now have the DACE surrogate for the first half.

Let me get `E[I]` into closed form, because if I can I'll be able to compute it and optimize it cheaply. Let me abbreviate `μ ≡ ŷ(x)`, `s ≡ s(x)`. `Y ~ Normal(μ, s²)`. Improvement is positive only when `Y < f_min`, so

  E[I] = ∫_{−∞}^{f_min} (f_min − y) · (1/s) φ((y − μ)/s) dy,

where φ is the standard normal density. Substitute `u = (y − μ)/s`, so `y = μ + s u`, `dy = s du`, and the upper limit `y = f_min` maps to `u = z` with

  z ≡ (f_min − μ)/s.

Then `f_min − y = f_min − μ − s u = s z − s u = s(z − u)` (using `f_min − μ = s z`), and the `1/s` and the `s du` cancel:

  E[I] = ∫_{−∞}^{z} s(z − u) φ(u) du = s[ z ∫_{−∞}^{z} φ(u) du − ∫_{−∞}^{z} u φ(u) du ].

The first integral is `Φ(z)`, the standard normal CDF. For the second, the key fact is `d/du[φ(u)] = −u φ(u)`, so `∫ u φ(u) du = −φ(u)`, giving `∫_{−∞}^{z} u φ(u) du = −φ(z) − (−φ(−∞)) = −φ(z)` (since `φ(−∞)=0`). Therefore

  E[I] = s[ z Φ(z) − (−φ(z)) ] = s z Φ(z) + s φ(z).

And `s z = s · (f_min − μ)/s = f_min − μ`, so

  **E[I(x)] = (f_min − μ) Φ((f_min − μ)/s) + s φ((f_min − μ)/s).**

Note it's `s` out front of the second term, not `s²` — easy to slip there, but the substitution makes it `s·φ`. Beautiful. And look at what the two terms *are*:

  • `(f_min − μ) Φ(z)` — the **exploitation** term. It's large when the predicted mean `μ` sits well below the incumbent `f_min` (the model thinks there's real value here), and `Φ(z)` is the probability that we actually land below `f_min`.

  • `s φ(z)` — the **exploration** term. It's large when `s` is large (the model is uncertain here), capturing the upside that even if `μ` isn't promising, a wide error bar means there's a fat chance the true value surprises us downward.

That's the whole exploit/explore tension resolved inside one scalar, with no knob to set. A point with low predicted mean scores via the first term; a point I know nothing about scores via the second; a point that's both gets both. The criterion *automatically* balances them. This is exactly what neither pure-exploit (first term only, effectively) nor pure-explore (second term only) could do.

Now let me confirm the behavior that makes it a *global* search and not a local one. At a sampled point `s = 0`. Then `z = (f_min − μ)/s → ±∞`, but `s Φ` and `s φ` both → 0 (the `s` prefactor kills it; `Φ` is bounded and `φ → 0` faster than any blowup). So `E[I] = 0` at every point I've already evaluated — the criterion never wastes a call re-sampling a known point. Between sampled points `E[I] > 0`. And crucially, EI is itself *highly multimodal*: it has a bump in every under-explored region as well as near promising low-`μ` regions. On a one-dimensional example with five points, EI shows two peaks — one near a promising basin, one out in an unexplored stretch; the taller peak wins this round, but once I sample there and `s` collapses locally, the *other* peak becomes the max and the search is driven across the box. That emergent globe-spanning behavior is the exploration term doing its job over iterations. So maximizing EI, evaluating, refitting, and repeating gives me global search for free — the thing pure exploitation could never deliver.

One worry: EI being multimodal *and* having big flat near-zero plateaus (most of the box, late in the run) means a naive multistart-from-random-seeds maximizer could miss the true max of EI — and missing it means I might sample a suboptimal point or, worse, falsely think EI is small everywhere and stop early. I'd like to maximize EI to *guaranteed* optimality. Can I? EI is in closed form, so maybe I can exploit its structure with branch-and-bound: recursively split the box into sub-boxes, compute an *upper bound* on EI over each sub-box, and prune any sub-box whose upper bound is below the best EI found so far. For that I need a cheap, valid upper bound on EI over a rectangle.

Here's where a clean fact helps. Differentiate `E[I]` w.r.t. `μ` and w.r.t. `s` and watch the terms cancel. Write `g(μ,s) = (f_min−μ)Φ(z) + s φ(z)` with `z = (f_min−μ)/s`.

For `∂g/∂μ`: `∂z/∂μ = −1/s`. Then
  `∂g/∂μ = −Φ(z) + (f_min−μ)φ(z)·(−1/s) + s φ'(z)·(−1/s)`.
Use `φ'(z) = −z φ(z)` and `(f_min−μ)/s = z`:
  `= −Φ(z) − z φ(z) + s·(−z φ(z))·(−1/s) = −Φ(z) − z φ(z) + z φ(z) = −Φ(z) < 0.`
So EI is monotonically *decreasing* in `μ` — lower predicted mean, more expected improvement. Makes sense.

For `∂g/∂s`: `∂z/∂s = −(f_min−μ)/s² = −z/s`. Then
  `∂g/∂s = (f_min−μ)φ(z)·(−z/s) + φ(z) + s φ'(z)·(−z/s)`.
Again `(f_min−μ) = s z` and `φ'(z) = −z φ(z)`:
  `= s z · φ(z)·(−z/s) + φ(z) + s·(−z φ(z))·(−z/s) = −z² φ(z) + φ(z) + z² φ(z) = φ(z) > 0.`
So EI is monotonically *increasing* in `s` — more uncertainty, more expected improvement. Both signs match the intuition, and both derivatives collapsed to a single clean term, which is exactly the structure I need: because EI is increasing in `s` and decreasing in `μ`, to upper-bound EI over a sub-box it suffices to find a *lower* bound `y_L` on `ŷ(x)` and an *upper* bound `s_U` on `s(x)` over that box, and plug `μ = y_L`, `s = s_U` into the closed form. (Lower-bounding `ŷ` and upper-bounding `s` over a box is itself doable from the kriging formulas — bound `r'R⁻¹(...)` componentwise — but the point I needed is just that the monotonicity makes a valid box-bound trivial to assemble.) That gives branch-and-bound a legitimate upper bound, so I can maximize EI to guaranteed global optimality despite its multimodality. (In practice a dense multistart / random-sample-then-local-polish does fine and is what I'll code, but the guarantee is what justifies trusting the criterion.)

Now I can assemble the loop. The pieces:

1. **Initial design.** Before I can fit the correlation parameters θ, p at all, I need a spread of points across the box — and I want them to cover the box's low-dimensional projections well, without clustering (clustered points make the correlation matrix nearly singular and waste evaluations). A Latin hypercube design (McKay, Conover & Beckman 1979) does exactly this: it's space-filling and its 1-D and 2-D projections are near-uniform. Evaluate `y` on these initial points — the only expensive calls up front.

2. **Fit the surrogate** by maximizing the concentrated likelihood over θ, p (μ̂, σ̂² closed-form inside). Then sanity-check the fit with cross-validated standardized residuals — leave one point out, predict it, divide the error by the predicted RMSE; a well-fit model keeps these below about 3 in magnitude. If they're bad, the response may need a transformation (model `log y` or `−1/y` instead of `y`) so the smooth-Gaussian-field assumption holds better.

3. **Sequential loop.** Maximize EI over the box (branch-and-bound for the guarantee, or dense multistart in practice). If the maximum EI has fallen below ~1% of the current best `|f_min|`, stop — and this is the self-contained stopping rule I wanted at the very start: EI is the model's *own* estimate of the remaining gain from sampling, so "EI is everywhere tiny" *means* "the model expects nothing more to be had." (On a log-transformed response, stop when EI on the log scale is below ~0.01 absolute, ≈1% relative.) Otherwise, evaluate `y` at the argmax of EI — one expensive call — append it, refit the surrogate, and repeat.

One more practical snag I should preempt: `R` can go nearly singular two ways. If the function is very smooth, neighboring columns of `R` are almost identical (everything correlates with everything) and `R` is ill-conditioned; and late in the run, as EI keeps pulling points into the same promising basin, near-duplicate points create near-duplicate columns. The fix is to solve the linear systems through an SVD of `R`, zeroing out tiny singular values rather than naively inverting (a small jitter / "nugget" on the diagonal does the same job in modern code). With that, the loop is stable.

Let me now write it against a real GP toolkit. Modern Gaussian-process libraries give me the DACE surrogate directly: scikit-learn's `GaussianProcessRegressor` with a Matérn kernel and per-dimension (ARD) length scales is the same animal — the Matérn smoothness parameter ν plays the role of the exponent `p` (smoothness), the ARD length scales are the reciprocal analog of the activity weights `θ_h`, and a `WhiteKernel`/noise term is the modern stand-in for the SVD nugget that keeps `R` invertible. skopt's `gp_minimize` wires up exactly the loop above: build the GP, in each iteration fit it, maximize the acquisition, evaluate, tell the model, refit.

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


def latin_hypercube(n_points, bounds, rng):
    # Space-filling start: one sample per stratum on each axis, shuffled
    # per dimension so the 1-D projections are near-uniform and points
    # don't cluster (clustering -> near-singular R, wasted evals).
    bounds = np.asarray(bounds, float)
    k = len(bounds)
    cut = np.linspace(0.0, 1.0, n_points + 1)
    u = rng.uniform(size=(n_points, k))
    pts = cut[:n_points, None] + u * (cut[1] - cut[0])
    for j in range(k):                       # decorrelate the columns
        pts[:, j] = rng.permutation(pts[:, j])
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + pts * (hi - lo)


class CorrelatedSurrogate:
    """Kriging/DACE surrogate: interpolates the data and reports an honest
    local error bar (s=0 at sampled points, growing away from them)."""

    def __init__(self):
        # constant mean (ConstantKernel) + correlated field (Matern, ARD
        # length scales = per-variable 'activity'; nu = smoothness ~ DACE p)
        # + a small noise term standing in for the SVD nugget on R.
        kernel = (ConstantKernel(1.0) *
                  Matern(length_scale=np.ones(1), nu=2.5) +
                  WhiteKernel(noise_level=1e-8))
        self.gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=10)

    def fit(self, X, y):
        # fit correlation parameters by maximum (marginal) likelihood;
        # mean/variance close in closed form inside the GP.
        self.gp.kernel.k1.k2.length_scale = np.ones(np.asarray(X).shape[1])
        self.gp.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X, return_std=True):
        return self.gp.predict(np.atleast_2d(X), return_std=return_std)


def expected_improvement(X, surrogate, f_min, xi=0.01):
    # The figure of merit that balances exploiting low mu against exploring
    # high sigma -- one scalar, no hand-set trade-off knob.
    mu, std = surrogate.predict(X, return_std=True)
    mu = np.atleast_1d(mu); std = np.atleast_1d(std)
    ei = np.zeros_like(mu)
    mask = std > 0                                   # EI = 0 where s = 0
    improve = f_min - xi - mu[mask]                  # f_min - mu (xi: margin)
    z = improve / std[mask]                          # z = (f_min - mu)/s
    ei[mask] = improve * norm.cdf(z) + std[mask] * norm.pdf(z)
    #          \_ exploitation: mu below f_min  \_ exploration: large s
    return ei


def maximize_ei(surrogate, bounds, f_min, rng, n_restarts=20, n_raw=10000):
    # EI is multimodal with big flat ~0 plateaus, so seed densely (the safe
    # stand-in for the branch-and-bound global guarantee) and polish.
    bounds = np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    raw = lo + rng.uniform(size=(n_raw, len(bounds))) * (hi - lo)
    vals = expected_improvement(raw, surrogate, f_min)
    seeds = raw[np.argsort(vals)[-n_restarts:]]
    best_x, best_ei = raw[vals.argmax()], vals.max()
    for x0 in seeds:                                 # local polish of -EI
        res = minimize(lambda x: -expected_improvement(x, surrogate, f_min)[0],
                       x0, bounds=list(map(tuple, bounds)), method="L-BFGS-B")
        if -res.fun > best_ei:
            best_x, best_ei = res.x, -res.fun
    return best_x, best_ei


def efficient_global_optimization(objective, bounds, n_init=10, max_evals=40,
                                  ei_tol_frac=0.01, seed=0):
    rng = np.random.default_rng(seed)
    X = latin_hypercube(n_init, bounds, rng)
    y = np.array([objective(x) for x in X])          # the only expensive calls
    surrogate = CorrelatedSurrogate()
    for _ in range(max_evals - n_init):
        surrogate.fit(X, y)
        f_min = y.min()
        x_next, ei = maximize_ei(surrogate, bounds, f_min, rng)
        # stopping rule from EI itself: the model's own estimate of the
        # remaining gain has fallen below 1% of the current best.
        if ei < ei_tol_frac * max(abs(f_min), 1e-12):
            break
        y_next = objective(x_next)                   # one expensive call
        X = np.vstack([X, x_next]); y = np.append(y, y_next)
    i = y.argmin()
    return X[i], y[i]
```

So the whole chain, in one breath: the objective is an expensive, gradientless, possibly-multimodal black box with a budget of tens of calls, which kills grid/random/local/true-function methods because they re-query instead of *learning*. Plain regression can't model a deterministic code's correlated residuals and gives no honest local uncertainty — but kriging/DACE does both, handing me an interpolating surrogate with a mean ŷ to exploit and an error bar s to explore. Chasing min-ŷ alone falls into a local minimum; chasing max-s alone wastes the budget; so I want one criterion that fuses them with no tuning knob. Probability-of-improvement only counts *whether* I improve, biasing toward the incumbent — so I score by the *expected amount* of improvement instead, `E[max(f_min−Y,0)]`, which integrates to the closed form `(f_min−ŷ)Φ(z) + s φ(z)` with `z=(f_min−ŷ)/s`, whose two terms *are* exploitation and exploration, balanced automatically. EI is zero at sampled points and multimodal, so it drives global search across iterations; it's monotone (decreasing in ŷ, increasing in s), so I can bound and maximize it to guaranteed optimality; and its own magnitude is a built-in stopping rule. Maximize EI, evaluate the real function once, refit the kriging surrogate, repeat — and land on a GP-with-Matérn / expected-improvement loop that is exactly the modern `gp_minimize`.
