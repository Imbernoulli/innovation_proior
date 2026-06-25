OK, let me start from where it actually hurts. I have a function `y(x)` over a box of design variables — say a handful to a dozen of them — and I want its global minimum. The catch that changes everything: one evaluation of `y` is *expensive*. It's an automotive crash simulation that takes the better part of a day, or a finite-element / circuit code that takes minutes to hours per run. So the budget isn't CPU-seconds spent thinking; it's the number of times I'm allowed to call `y`. Tens of calls, maybe. Not thousands. And the code is a black box: no gradient comes out of it, the surface is probably nonlinear and quite possibly multimodal, and I have no analytic form — if I did, I wouldn't have paid to build the expensive simulator.

So the real objective behind the objective is: locate the global min in as few evaluations as possible, and — this would be lovely — have some self-contained signal telling me when there's nothing left to gain, so I can stop. A method that burns hundreds of calls, or that quietly settles into a local minimum and declares victory, is useless here.

Let me put numbers on the off-the-shelf options before reaching for anything clever, because the budget arithmetic alone disqualifies most of them. Grid search: in `k` dimensions a grid with `m` points per axis is `m^k` evaluations. For `k=6` and even a coarse `m=5` that's `5^6 = 15625` calls of a 20-hour simulation — about 36 years of wall-clock. Dead on arrival. Random search has the same curse with worse constants — it spreads the budget uniformly and ignores everything it learns. A multistart local optimizer (gradient descent from many seeds) is wrong on two counts: there's no gradient, and even with finite differences each local run alone eats more evaluations than my whole budget, and on a multimodal surface it just finds whatever basin its seed fell into. Genetic algorithms / branch-and-bound *on the true function* — same story, they're designed for cheap functions and assume you can afford thousands of evaluations. The common thread in all of them is that they keep re-querying the expensive function and never reuse what those queries taught them.

That common thread is the lever. Every one of those calls is gold, and these methods throw the information away. The alternative is to *build a model of `y` from the calls I've made* and let that cheap model decide where to spend the next expensive call. Fit a surface, then reason about the surface for free.

The classical version of that is a response surface (RSM, Box–Hunter–Hunter 1978): assume `y(x_i) = Σ_h β_h f_h(x_i) + ε_i` with the `ε_i` independent zero-mean noise of variance σ², least-squares the β's, and optimize the fitted polynomial. Let me actually try to use this and see where it cracks. First crack: I have to *pick* the regressors `f_h`. A quadratic? But the whole reason the expensive code exists is that I don't know the functional form — if it's wigglier than quadratic, a quadratic systematically misses, and a flexible basis with many terms needs many evaluations to pin down its many coefficients. Second crack, and this one is deeper: the independent-noise assumption. My code is *deterministic*. Run it twice at the same `x`, get the identical number. There is no measurement noise. So what is `ε(x)`? It's the part of `y(x)` my regressors failed to capture — the left-out terms in `x`. And those left-out terms are a *continuous function of `x`*. Stare at that for a second. If `ε(x)` is continuous, then `ε(x_i)` and `ε(x_j)` for two nearby points are *nearly equal*. The residuals are correlated — strongly when the points are close, weakly when far apart — and treating them as independent draws is just structurally false for a deterministic code. The regression's own error model is wrong precisely where the function is deterministic, and the wrongness has a definite shape: correlation that decays with distance.

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

There is a useful economy: for *fixed* correlation parameters (the θ's and p's, which fix R), I can solve for μ and σ² in closed form. Take the log-likelihood, differentiate w.r.t. μ: the only μ-dependence is in the quadratic form `(y−1μ)'R⁻¹(y−1μ)`, whose derivative set to zero gives the generalized-least-squares mean

  μ̂ = (1' R⁻¹ y) / (1' R⁻¹ 1).

Differentiate w.r.t. σ² and set to zero:

  σ̂² = (y − 1μ̂)' R⁻¹ (y − 1μ̂) / n.

Substitute both back and what's left is a *concentrated* log-likelihood in just the θ's and p's, which I maximize numerically. So instead of fitting many regression coefficients I'm fitting a handful of correlation parameters — far cheaper in evaluations.

Now the payoff: prediction at a new point `x*`. Let `r` be the n-vector of correlations between `x*` and the sampled points, `r_i = Corr(ε(x*), ε(x_i))`. The best linear unbiased predictor is

  ŷ(x*) = μ̂ + r' R⁻¹ (y − 1μ̂).

I should check the interpolation claim rather than trust it, because it's the whole reason I switched to kriging. Put `x* = x_i`, one of the sampled points. Then `r` equals the i-th column of `R`, so `r' R⁻¹ = e_i'` (the i-th unit row vector, since `R⁻¹` times the i-th column of `R` is `e_i`). Thus `ŷ(x_i) = μ̂ + e_i'(y − 1μ̂) = μ̂ + (y_i − μ̂) = y_i`. It reproduces the data exactly. Good — deterministic-consistent.

And the uncertainty. The mean-squared error of this predictor is

  s²(x*) = σ²[ 1 − r' R⁻¹ r + (1 − 1' R⁻¹ r)² / (1' R⁻¹ 1) ].

Check it at a sampled point `x* = x_i`: there `r' R⁻¹ r = r' e_i = r_i = R_{ii} = 1`, and `1' R⁻¹ r = 1' e_i = 1`, so the bracket is `1 − 1 + (1−1)²/(...) = 0`. `s² = 0` at every sampled point — exactly zero uncertainty where I've evaluated, as a deterministic function demands. Far from all data, `r → 0`, so the bracket → `1 − 0 + (1)²/(1'R⁻¹1)` ≈ 1 and `s² ≈ σ²` — full prior uncertainty out in the unexplored wilderness. The RMSE `s(x*) = √(s²)` is my honest local error bar. So the surrogate does carry both pieces the problem asked for — a `μ(x)` to exploit and a `σ(x)` to explore — and the two boundary checks (zero error at data, full prior error far away) come out as they must.

(There's a tidier way to see the predictor too. Pretend I've evaluated a pseudo-observation `(x*, y*)` and write down the augmented likelihood for the n+1 points. The only `y*`-dependent part of the augmented quadratic form works out to `(1/(1 − r'R⁻¹r))(y*−μ̂)² − (2 r'R⁻¹(y−1μ̂)/(1−r'R⁻¹r))(y*−μ̂) + const`. Maximize over `y*` — set the derivative to zero: `(2/(1−r'R⁻¹r))(y*−μ̂) − 2 r'R⁻¹(y−1μ̂)/(1−r'R⁻¹r) = 0`, so `y* = μ̂ + r'R⁻¹(y−1μ̂)`, the same predictor. The interpretation is cute: the prediction is the value that, if I'd observed it, would be *most consistent* with the correlated data I already have.)

Right. I have a cheap surrogate that interpolates and reports its own ignorance. Now the actual question: **where do I sample next?** This is where it would be easy to do something dumb, so let me reason it out rather than guess a rule.

Dumb idea #1, pure exploitation: fit the surrogate, jump to the `x` that minimizes `ŷ(x)`, evaluate there, refit, repeat. Walk through what happens on a multimodal surface. The surrogate interpolates my data, so its minimum sits at — or very near — the lowest *data point* I happen to have, i.e. in whatever basin I've already sampled into. I jump there, evaluate, the new point pins the surrogate down locally, and the next minimum is again right beside it. I spiral into the nearest local minimum and never learn that a deeper basin sits across the box where I've never looked. The minimum of a fitted surface is a *local* minimum of the data. So pure exploitation can't be the rule — it's structurally a local search wearing a surrogate.

Dumb idea #2, pure exploration: sample wherever `s(x)` is largest — the point I know least about. That fills in the map, sure, but it spends my entire microscopic budget charting empty corners of the box and essentially never zeroes in on the optimum. Also useless.

So neither pole works, and the truth is in between: I need a single figure of merit that is large where `ŷ` is low (worth exploiting) *and* large where `s` is high (worth exploring), and that trades the two off **on its own**, without me hand-tuning some exploit/explore knob per problem. The trade-off has to be intrinsic to the criterion, or I'm back to babysitting.

What has the literature tried for the "where next" rule? Kushner (1964) modeled a 1-D objective as a Wiener process and sampled to maximize the *probability of improvement*: `P(Y(x) < f_min)`, the chance the new value beats the current best `f_min`. That's a real step — it uses the surrogate's distribution at `x`, not just its mean. But let me see what PI actually rewards. It counts *whether* I improve, not *by how much*. So a point sitting just barely below the incumbent, with tiny `s`, where improvement is nearly certain but minuscule, scores almost 1 — while a point with a long lower tail that *could* improve by a lot but only with moderate probability scores less. PI hugs the incumbent and is biased toward small, near-certain exploitation gains. To make it explore you have to inflate the target (`P(Y < f_min − ξ)` with a hand-set ξ) — back to a per-problem knob. And it was posed in 1-D on a Wiener model. The right instinct is there; the criterion is just measuring the wrong thing.

The flaw in PI points at its own fix. If "*whether* you improve" loses the magnitude, then score by the *amount* of improvement. Define the improvement random variable

  I(x) = max(f_min − Y(x), 0),

where `Y(x) ~ Normal(ŷ(x), s²(x))` is my surrogate's belief about the unknown value at `x`. `I` is the shortfall below the incumbent if `x` turns out better, and 0 if it doesn't — it's literally how much I'd gain. Now weight every possible improvement by its probability density and add them up: take the expectation. This is the *expected* improvement,

  E[I(x)] = E[ max(f_min − Y(x), 0) ],

which Mockus, Tiesis & Zilinskas (1978) posed global optimization in — the random-function, expected-gain framework. Where PI integrates the density over the improving region (probability mass only), this integrates `(improvement) × (density)`, so it credits both *how likely* an improvement is and *how big* it would be. What was missing before was a surrogate concrete enough to actually evaluate `E[I]` on, and a way to maximize the (nasty, multimodal) criterion reliably — and I now have the DACE surrogate for the first half.

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

Note it's `s` out front of the second term, not `s²` — easy to slip there, but the substitution makes it `s·φ`. That kind of factor is exactly where a derivation goes quietly wrong, so before I build anything on it let me check the closed form against the integral it came from at a couple of concrete points. Take `μ = 2.3`, `s = 0.8`, `f_min = 2.0` (predicted mean a hair *above* the incumbent, modest uncertainty). Then `z = (2.0 − 2.3)/0.8 = −0.375`; the formula gives `(−0.3)·Φ(−0.375) + 0.8·φ(−0.375) = (−0.3)(0.3538) + 0.8(0.3719) = −0.1061 + 0.2975 = 0.1913`. Numerically integrating `∫_{−∞}^{2.0}(2.0−y)·(1/0.8)φ((y−2.3)/0.8) dy` gives `0.19134`. They agree to four places. A second case with the mean well below the incumbent — `μ = 1.0`, `s = 0.5`, `f_min = 2.0`, so `z = 2.0`: formula gives `1.0·Φ(2)+0.5·φ(2) = 0.97725 + 0.5(0.05399) = 1.00425`, and the direct integral gives `1.00425` too. Both match, so the `s·φ` (not `s²·φ`) form is the right one and the algebra holds.

The first term, `(f_min − μ) Φ(z)`, is the exploitation part: it grows when the predicted mean `μ` sits below the incumbent `f_min`, and `Φ(z)` is the probability of actually landing below `f_min`. The second term, `s φ(z)`, is the exploration part: it grows with uncertainty, capturing the upside that even if `μ` is not promising, a wide error bar leaves room for a downward surprise. So the exploit/explore tension I was worried about hand-tuning is sitting inside one scalar with no knob: a point with low predicted mean scores through the first term; a point I know nothing about scores through the second; a point that's both gets both. That's the property pure-exploit (first term only) and pure-explore (second term only) each lacked.

Now I want to confirm this actually produces *global* search and not a dressed-up local one — and I can do that on a small worked example rather than asserting it. Take a 1-D problem on `[0,10]`. Sample five points only on the left half: `x = (0,1,2,3,4)` with `y = (3.0,1.0,2.0,2.5,3.0)`, so the incumbent is `f_min = 1.0` at `x=1`. Fit the surrogate with `θ=0.3`, `p=2`. The fitted constant mean comes out `μ̂ ≈ 3.36` with `σ ≈ 2.18`. First, EI at the sampled points: at each of `x=0,1,2,3,4` I get `s=0`, and the formula collapses to `0` (the `s` prefactor on both terms kills it) — confirmed numerically, EI is `0.0` at all five. So the criterion never wastes a call re-sampling a known point. Now the unsampled right half. The predicted mean out there reverts toward `μ̂ ≈ 3.4`, which is *well above* the incumbent `1.0`, so the exploitation term is actually *negative* — pure exploitation would never look here. Yet EI is large and positive:

  x=5:  ŷ=3.99, s=1.10, EI=0.0010  (exploit −0.009, explore +0.010)
  x=7:  ŷ=3.55, s=2.54, EI=0.210   (exploit −0.402, explore +0.612)
  x=9:  ŷ=3.37, s=2.61, EI=0.260   (exploit −0.432, explore +0.692)
  x=10: ŷ=3.36, s=2.61, EI=0.260   (exploit −0.432, explore +0.692)

The exploration term `s φ(z) ≈ 0.69` swamps the negative exploitation term, and the global EI maximum over the whole box lands at the far edge `x=10` with `EI ≈ 0.26`, whereas the best EI anywhere in the *sampled* left half is only about `0.009` — a factor of thirty smaller. So with a big unexplored gap, EI sends the next sample straight into the gap, not back toward the incumbent. That is the global behavior, and it's the exploration term doing it, not an assumption I baked in. (And once I evaluate out there and `s` collapses locally, that peak drops out and the argmax moves on — whether it returns to the incumbent's basin or to the next-emptiest region depends on what the new value turns out to be, which is exactly the data-driven balance I wanted. I won't claim it always ping-pongs across the box; on a configuration where the incumbent basin is still the most promising even after sampling, EI can keep refining there instead, and that's correct too.)

So maximizing EI, evaluating, refitting, and repeating gives me a search that exploits and explores by itself. The remaining worry is *reliably finding* the EI maximum. EI is itself multimodal — there's a bump near every under-explored region as well as near promising low-`μ` regions — and late in the run most of the box is a near-zero plateau. A naive multistart-from-random-seeds maximizer could miss the true max of EI, and missing it means I might sample a suboptimal point or, worse, falsely think EI is small everywhere and stop early. I'd like to maximize EI to optimality with a guarantee. Can I? EI is in closed form, so maybe I can exploit its structure with branch-and-bound: recursively split the box into sub-boxes, compute an *upper bound* on EI over each sub-box, and prune any sub-box whose upper bound is below the best EI found so far. For that I need a cheap, valid upper bound on EI over a rectangle.

The clean way to bound EI is to understand its monotonicity in `μ` and `s`, so let me differentiate `E[I]` w.r.t. each and watch the terms cancel. Write `g(μ,s) = (f_min−μ)Φ(z) + s φ(z)` with `z = (f_min−μ)/s`.

For `∂g/∂μ`: `∂z/∂μ = −1/s`. Then
  `∂g/∂μ = −Φ(z) + (f_min−μ)φ(z)·(−1/s) + s φ'(z)·(−1/s)`.
Use `φ'(z) = −z φ(z)` and `(f_min−μ)/s = z`:
  `= −Φ(z) − z φ(z) + s·(−z φ(z))·(−1/s) = −Φ(z) − z φ(z) + z φ(z) = −Φ(z) < 0.`
So EI is monotonically *decreasing* in `μ` — lower predicted mean, more expected improvement. Makes sense.

For `∂g/∂s`: `∂z/∂s = −(f_min−μ)/s² = −z/s`. Then
  `∂g/∂s = (f_min−μ)φ(z)·(−z/s) + φ(z) + s φ'(z)·(−z/s)`.
Again `(f_min−μ) = s z` and `φ'(z) = −z φ(z)`:
  `= s z · φ(z)·(−z/s) + φ(z) + s·(−z φ(z))·(−z/s) = −z² φ(z) + φ(z) + z² φ(z) = φ(z) > 0.`
So EI is monotonically *increasing* in `s` — more uncertainty, more expected improvement. Both signs match the intuition, and both derivatives collapsed to a single clean term. (The numbers in the worked example above are consistent with this: as `s` rose from 1.10 at x=5 to 2.61 at x=10, EI rose monotonically; as ŷ drifted down from 3.99 toward 3.36, that also pushed EI up.) If I can get a lower bound `y_L` on `ŷ(x)` and an upper bound `s_U` on `s(x)` over any rectangular sub-box, then plugging `μ = y_L` and `s = s_U` into the EI formula gives a valid upper bound on EI over that sub-box. That is enough for branch-and-bound.

I should not hand-wave those two bounds, because this is where a missed sign would wreck the guarantee. Over a box `l_h <= x_h <= u_h`, the correlations are variables constrained by

  ln(r_i) = -Σ_h θ_h |x_h - x_{ih}|^{p_h}.

The upper bound on `s²` starts by maximizing the MSE formula itself over `x` and `r` subject to those equations. I rewrite each equality as the two inequalities `ln(r_i) + Σ_h θ_h |x_h - x_{ih}|^{p_h} <= 0` and `-ln(r_i) - Σ_h θ_h |x_h - x_{ih}|^{p_h} <= 0`, add the interval bounds on `x` and the induced interval bounds `r_i^L <= r_i <= r_i^U`, and then negate the objective so the problem becomes a minimization. The Hessian of this negated objective, with respect to the `r` variables, is

  2σ²[ R⁻¹ - (R⁻¹1)(R⁻¹1)' / (1'R⁻¹1) ].

If its smallest eigenvalue `λ_min` is negative, I can force convexity with the αBB move: add `α Σ_i (r_i - r_i^L)(r_i - r_i^U)` and choose `α = max(0, -λ_min/2)`. The factor `1/2` is important because that added quadratic contributes `2α` to each diagonal Hessian entry. On the interval `[r_i^L, r_i^U]`, each product `(r_i - r_i^L)(r_i - r_i^U)` is nonpositive, so the modified minimization objective lies below the original objective while becoming convex. Then I replace the nonlinear pieces in the constraints by linear underestimators. That makes the constraints easier to satisfy, so I am still relaxing the original problem, not tightening it. Solving this convex relaxed minimization and reversing the sign gives an upper bound on the original maximum of `s²`, hence an upper bound on `s`.

For the lower bound on `ŷ`, the predictor is linear in `r` once the model is fit:

  ŷ(x) = μ̂ + c' r,  c = R⁻¹(y - 1μ̂).

I could minimize that linear function over the same convex relaxation, but it is loose. If I set every `p_h = 2`, which is the smooth case and the case that makes the fast bound simple, then `r_i(x) = exp(-z_i(x))` with `z_i(x) = Σ_h θ_h (x_h - x_{ih})²`, and

  ŷ(x) = μ̂ + Σ_i c_i exp(-z_i(x)).

Interval arithmetic over the sub-box gives `z_i^L <= z_i(x) <= z_i^U`. Now the sign of `c_i` decides the underestimator. If `c_i >= 0`, then `c_i exp(-z)` is convex, so its tangent at the interval midpoint lies below it. If `c_i < 0`, then multiplying by the negative coefficient makes it concave, so the chord over `[z_i^L, z_i^U]` lies below it. In both cases I get a line `a_i + b_i z_i` satisfying `a_i + b_i z_i <= c_i exp(-z_i)` throughout the interval. Substituting all those lines gives

  ŷ(x) >= μ̂ + Σ_i a_i + Σ_h θ_h Σ_i b_i (x_h - x_{ih})².

The right-hand side separates by coordinate. For each coordinate `h`, I only have to minimize a one-variable quadratic over `[l_h, u_h]`: if the quadratic coefficient is positive, use the vertex clipped to the interval; if it is negative, use the better endpoint; if it is zero, the remaining linear term picks an endpoint, and if that is zero too, every point in the interval is tied. Adding the coordinate minima gives a valid lower bound `y_L`. This is why fixing `p_h = 2` is attractive in the branch-and-bound version: the bound becomes fast and separable; the `p_h < 2` extension is possible but more complicated.

At this point the loop is forced. Before I can fit the correlation parameters θ, p at all, I need a spread of points across the box, and I want them to cover low-dimensional projections well without clustering, because clustered points waste evaluations and make `R` nearly singular. A Latin hypercube design (McKay, Conover & Beckman 1979) gives one stratified sample per coordinate level and shuffles the levels by dimension, so the one- and two-dimensional projections are nearly uniform. I evaluate `y` on those initial points, fit the surrogate by maximizing the concentrated likelihood over θ and p with `μ̂` and `σ̂²` closed inside, and check cross-validated standardized residuals by leaving out one point, predicting it, and dividing the error by the predicted RMSE. If the residuals are not within about 3 in magnitude, I should try a response transformation such as `log y` or `-1/y` so the smooth-Gaussian-field assumption has a better chance of being true. Then the sequential part is simple: maximize EI over the box, stop if the maximum EI is below about 1% of the current best `|f_min|` because EI is the model's own estimate of remaining one-step gain, otherwise evaluate the real objective at the EI maximizer, append the point, refit, and repeat. On a log-transformed response, an EI threshold of about `0.01` on the log scale corresponds to roughly a 1% relative change.

One more practical snag I should preempt: `R` can go nearly singular two ways. If the function is very smooth, neighboring columns of `R` are almost identical (everything correlates with everything) and `R` is ill-conditioned; and late in the run, as EI keeps pulling points into the same promising basin, near-duplicate points create near-duplicate columns. The fix is to solve the linear systems through an SVD of `R`, zeroing out tiny singular values rather than naively inverting (a small jitter / "nugget" on the diagonal does the same job in modern code). With that, the loop is stable.

Let me now write it against a real GP toolkit. Modern Gaussian-process libraries give me the DACE surrogate directly: scikit-learn's `GaussianProcessRegressor` with a Matérn kernel and per-dimension (ARD) length scales is the same animal — the Matérn smoothness parameter ν plays the role of the exponent `p` (smoothness), and the ARD length scales are the reciprocal analog of the activity weights `θ_h`. For a deterministic objective I want the noise level essentially zero, but I still need a tiny numerical nugget so the linear algebra does not fall apart when `R` is nearly singular. skopt's `gp_minimize` has the same skeleton: build the GP, generate initial points, optimize an acquisition such as EI by sampling plus L-BFGS-B, evaluate, tell the optimizer, and refit. Its `gaussian_ei` also exposes an optional `xi` margin; setting `xi = 0` gives the exact EI I just derived, while a small positive `xi` demands improvement beyond the incumbent by that margin.

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


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
    """GP/kriging surrogate with ARD length scales and a tiny numerical nugget."""

    def __init__(self, alpha=1e-10):
        self.alpha = alpha
        self.gp = None

    def fit(self, X, y):
        # Fit correlation parameters by maximum marginal likelihood.  The tiny
        # alpha is numerical jitter; the analytic DACE model is noise-free.
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        kernel = (ConstantKernel(1.0) *
                  Matern(length_scale=np.ones(X.shape[1]), nu=2.5))
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=self.alpha, normalize_y=True,
            n_restarts_optimizer=10)
        self.gp.fit(X, y)
        return self

    def predict(self, X, return_std=True):
        return self.gp.predict(np.atleast_2d(X), return_std=return_std)


def expected_improvement(X, surrogate, f_min, xi=0.0):
    # xi=0 is the original EI threshold; skopt exposes the same margin and
    # often sets xi=0.01 to demand a small improvement beyond the incumbent.
    mu, std = surrogate.predict(X, return_std=True)
    mu = np.atleast_1d(mu); std = np.atleast_1d(std)
    ei = np.zeros_like(mu)
    mask = std > 1e-12                               # EI = 0 where s = 0
    improve = f_min - xi - mu[mask]                  # f_min - mu when xi=0
    z = improve / std[mask]                          # z = improve / s
    ei[mask] = improve * norm.cdf(z) + std[mask] * norm.pdf(z)
    #          \_ exploitation: mu below f_min  \_ exploration: large s
    return ei


def maximize_acquisition(acq_fn, bounds, rng, n_restarts=20, n_raw=10000):
    # The acquisition is multimodal with flat near-zero plateaus, so seed
    # densely and polish the best starts, matching the skopt optimizer pattern.
    bounds = np.asarray(bounds, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    raw = lo + rng.uniform(size=(n_raw, len(bounds))) * (hi - lo)
    vals = acq_fn(raw)
    seeds = raw[np.argsort(vals)[-n_restarts:]]
    best_x, best_val = raw[vals.argmax()], float(vals.max())
    for x0 in seeds:
        res = minimize(lambda x: -float(acq_fn(x)[0]),
                       x0, bounds=list(map(tuple, bounds)), method="L-BFGS-B")
        if -res.fun > best_val:
            best_x, best_val = res.x, -res.fun
    return best_x, best_val


def efficient_global_optimization(objective, bounds, n_init=10, max_evals=40,
                                  ei_tol_frac=0.01, xi=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = latin_hypercube(n_init, bounds, rng)
    y = np.array([objective(x) for x in X])          # the only expensive calls
    surrogate = CorrelatedSurrogate()
    for _ in range(max_evals - n_init):
        surrogate.fit(X, y)
        f_min = y.min()
        acq_fn = lambda Xcand: expected_improvement(Xcand, surrogate, f_min, xi)
        x_next, ei = maximize_acquisition(acq_fn, bounds, rng)
        # stopping rule from EI itself: the model's own estimate of the
        # remaining gain has fallen below 1% of the current best.
        if ei < ei_tol_frac * max(abs(f_min), 1e-12):
            break
        y_next = objective(x_next)                   # one expensive call
        X = np.vstack([X, x_next]); y = np.append(y, y_next)
    i = y.argmin()
    return X[i], y[i]
```

So the whole chain, in one breath: the objective is an expensive, gradientless, possibly-multimodal black box with a budget of tens of calls, which kills grid/random/local/true-function methods because they re-query instead of *learning*. Plain regression can't model a deterministic code's correlated residuals and gives no honest local uncertainty — but kriging/DACE does both, handing me an interpolating surrogate (verified: predictor reproduces the data, `s²=0` at sampled points) with a mean ŷ to exploit and an error bar s to explore. Chasing min-ŷ alone falls into a local minimum; chasing max-s alone wastes the budget; so I want one criterion that fuses them with no tuning knob. Probability-of-improvement only counts *whether* I improve, biasing toward the incumbent — so I score by the *expected amount* of improvement instead, `E[max(f_min−Y,0)]`, which integrates to the closed form `(f_min−ŷ)Φ(z) + s φ(z)` with `z=(f_min−ŷ)/s` (checked against the direct integral to four places), whose two terms *are* exploitation and exploration. On a 1-D example with a deliberate unexplored gap, EI was zero at every sampled point and its global maximum landed out in the gap — pushed there by the exploration term against a negative exploitation term — so iterating it searches globally rather than locally. It is monotone, decreasing in ŷ and increasing in s, so I can bound and maximize it to guaranteed optimality; and its own magnitude is a built-in stopping rule. Maximize EI, evaluate the real function once, refit the kriging surrogate, repeat, and the code lands on the same GP-with-Matérn / expected-improvement pattern used by `gp_minimize`.
