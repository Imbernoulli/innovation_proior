Let me start from what actually goes wrong when I try to *choose* a portfolio to be safe. I have weights `w` to pick out of some feasible set `X` — budget `Σ wᵢ = 1`, maybe no shorting, maybe a return floor — and an uncertain market that turns `w` into a random loss `L(w, y) = -R(y)ᵀ w`, where `R` is the vector of asset returns. I want the big losses to be rare and not too big. The number everyone reaches for is Value-at-Risk: at confidence `α`, say 0.95, `VaR_α(w)` is the smallest threshold `ζ` such that `P(L(w,y) ≤ ζ) ≥ α` — the α-quantile of the loss. It is the regulatory standard, so the obvious move is "minimize `VaR_α(w)` over `w ∈ X`." Let me just try to do that and see where it hurts.

The first thing that bites is that a quantile is a brutal function of `w`. Picture the scenario world I actually live in: I don't have the true distribution of `y`, I have a panel of `N` historical or simulated draws `y_1, …, y_N`, each with weight `1/N`, and the loss in each is `L_k(w) = -R_kᵀ w`. Then `VaR_α(w)` is just the `⌈αN⌉`-th smallest of those `N` numbers — an order statistic. As I nudge `w`, the `N` losses slide around and *reorder*, so which scenario sits in the `⌈αN⌉`-th slot keeps changing. With linear scenario losses this order statistic is a continuous piecewise-linear surface: on one region it equals one scenario's loss line, on the next region it equals another, and the switch creates a kink. There is no reason for those glued-together pieces to be convex, and in high dimension they can make many local minima. If I hand that to a gradient method or a convex solver I get a local optimum that has nothing to do with the global one. So even setting aside *what* VaR measures, *minimizing* it is a combinatorial mess. Wall.

And there's a second thing wrong that's about *what* it measures, independent of optimization. VaR looks at exactly one point of the distribution — the quantile — and is completely blind to everything past it. Two portfolios can have the identical 95% VaR while one barely pokes over the threshold in its worst 5% and the other loses the whole book. VaR scores them the same. For a risk number whose entire job is to protect against disaster, being unable to distinguish "a little over the line" from "wiped out" is disqualifying.

Let me make the structural defect precise, because "blind to the tail" is vague and I want the sharp version. The clean way is to ask: does VaR reward diversification? Take two corporate bonds, A and B, each defaulting independently with probability `p = 0.04`, losing a fixed amount `L` on default and `0` otherwise. Work at `α = 0.95`, so the tail mass is `1 − α = 0.05`. One bond on its own: it loses `L` with probability `0.04` and `0` with probability `0.96`. Since `P(loss ≤ 0) = 0.96 ≥ 0.95`, the 95%-quantile is `0` — so `VaR(A) = VaR(B) = 0`. Now pool them. Independence gives `P(both survive) = 0.96² = 0.9216`, `P(exactly one defaults) = 2·0.04·0.96 = 0.0768`, `P(both default) = 0.0016`. So for the combined book, `P(loss ≤ 0) = 0.9216`, which is *below* 0.95, and I have to climb to `P(loss ≤ L) = 0.9216 + 0.0768 = 0.9984 ≥ 0.95` to clear the confidence level. Hence `VaR(A+B) = L`. Put the pieces together: `VaR(A+B) = L > 0 = VaR(A) + VaR(B)`. Combining two *independent* risks made VaR strictly *bigger* than the sum of the standalone risks. That's perverse — diversification is supposed to help — and it's fatal for capital budgeting, where I want to hand desks separate limits and trust that the parts add up to a bound on the whole. Subadditivity is exactly the property I just watched VaR violate.

So I can name what I want axiomatically, in the spirit of Artzner, Delbaen, Eber and Heath: a *coherent* risk measure, one that is subadditive (`ρ(X₁+X₂) ≤ ρ(X₁)+ρ(X₂)`, "a merger doesn't create risk"), positively homogeneous (`ρ(λX)=λρ(X)`), monotone (worse outcomes, higher risk), and translation-equivariant (adding sure cash shifts the risk by that amount). VaR fails the first. The same authors point at a coherent replacement — a worst-case / tail conditional expectation — but they characterize it as a supremum of expectations over a whole family of "generalized scenario" probability measures, and they're candid that it's impractical, computable only in narrow special cases. So I have a target — coherent, tail-aware — but no recipe for minimizing it over `w`. That's the gap.

What's the natural tail-aware number? If VaR is the threshold the loss breaches 5% of the time, the honest summary of "how bad is the bad 5%" is the *average* loss conditional on breaching it; in the continuous case I can write that as `E[ L | L ≥ VaR_α ]`. The mean of the tail. This is the mean-excess-loss / mean-shortfall idea from the insurance and extreme-value literature, and it is obviously what I want: it reads the whole tail, not one point of it, so it can tell a thin overshoot from a catastrophe, and intuitively averaging ought to behave better under pooling than a quantile does. Call this `φ_α(w)` — the conditional value-at-risk, the mean of the α-tail.

But look at how it's written: `φ_α(w) = E[ L | L ≥ VaR_α(w) ]`. It is *defined through* `VaR_α(w)`. To evaluate `φ` at a given `w` I must first find the quantile `VaR_α(w)` — the very object that's non-convex, nonsmooth, and multi-extremal in `w`. So if I try to optimize `φ` written this way, I drag VaR's pathology in through the back door: my objective contains an inner order statistic whose active scenario changes with `w`. Wall again. The concept is right; the formula is poisoned by its dependence on the quantile.

So the real question sharpens to: can I compute the tail-mean `φ_α(w)` *without* first computing `VaR_α(w)` — get it from something convex and direct? Let me think about what role the threshold actually plays. The trouble is that `VaR_α(w)` is a *function of `w`* sitting inside, with its active scenario changing as `w` changes. What if I stop treating the threshold as a function of `w` and instead make it a *free variable* `ζ` that I optimize over separately? Then there's no quantile-of-`w` buried in the objective — just an ordinary real number `ζ` I'm free to choose. The hope is that if I write the right function of `(w, ζ)` and minimize over `ζ`, the argmin becomes the quantile set and the optimal value *equals* the tail-mean. Let me try to build such a function.

I want an object that (a) for a *fixed* candidate threshold `ζ` is convex and easy, (b) when minimized over `ζ` reproduces the tail-mean, and (c) makes the right `ζ` come out as the quantile. The tail-mean is "threshold plus average overshoot beyond it." Average overshoot beyond `ζ` is `E[(L − ζ)⁺]`, with `(t)⁺ = max{t, 0}` — and crucially `(t)⁺` is convex in `t`, hence `(L − ζ)⁺` is convex in `ζ`, and the expectation of a convex function is convex. That's the seed of property (a). In the continuous case, if `ζ` were exactly the VaR, the probability of exceeding it is `1 − α`, so the *conditional* tail-mean is `ζ + (1/(1−α))·E[(L−ζ)⁺]` — divide the total overshoot by the tail probability `1−α` to turn a sum over the tail into an average over the tail. So I'll *define* this for an arbitrary `ζ`, not just at the VaR, and see what minimizing it does:

```
F_α(w, ζ) = ζ + (1/(1−α)) · E[ (L(w,y) − ζ)⁺ ].
```

Let me sanity-check the two extreme pulls on `ζ` to feel why a minimum exists somewhere sensible. If I push `ζ` way down (very negative), then `(L−ζ)⁺ ≈ L − ζ` almost surely, so `F ≈ ζ + (1/(1−α))(E[L] − ζ) = (1/(1−α))E[L] − (α/(1−α))ζ`, which *grows* as `ζ → −∞` (coefficient on `ζ` is `−α/(1−α) < 0`). If I push `ζ` way up (above the max loss), then `(L−ζ)⁺ = 0` and `F = ζ → +∞`. So `F` blows up in both directions and the minimum is in the interior. Good — there's a real minimizer to chase. The two terms are fighting: the bare `+ζ` rewards lowering `ζ`, the overshoot term penalizes lowering it (more of the distribution pokes above a lower threshold, and the overshoot is amplified by `1/(1−α)`). The balance point is where the marginal trade-off flips sign — and I bet that's the quantile. Let me prove it.

`F_α(w, ·)` is convex in `ζ` (sum of the linear `ζ` and a convex expectation), so it has one-sided derivatives everywhere and the minimizers are exactly the `ζ` where the left derivative is `≤ 0 ≤` the right derivative. So I need `∂F/∂ζ`. Let me compute the right derivative honestly from the difference quotient. Write `Ψ(ζ) = P(L ≤ ζ)` for the loss CDF (suppress `w`). Take `ζ' > ζ` and look at

```
[ F_α(w, ζ') − F_α(w, ζ) ] / (ζ' − ζ)
  = 1 + (1/(1−α)) · E{ [ (L−ζ')⁺ − (L−ζ)⁺ ] / (ζ' − ζ) }.
```

The +1 is from the bare `ζ` term. For the expectation, look at the integrand pointwise as a function of the realized loss `L`:
- If `L ≥ ζ'`: both `(L−ζ')⁺ = L−ζ'` and `(L−ζ)⁺ = L−ζ`, so the difference is `(L−ζ') − (L−ζ) = −(ζ'−ζ)`, and divided by `(ζ'−ζ)` it's `−1`.
- If `L ≤ ζ`: both positive parts are `0`, so the quotient is `0`.
- If `ζ < L < ζ'`: it's somewhere strictly between, in `(−1, 0)`.

So the integrand is `−1` on the event `{L ≥ ζ'}`, `0` on `{L ≤ ζ}`, and a bounded value in `(−1,0)` on the sliver `{ζ < L < ζ'}` which has probability `Ψ(ζ') − Ψ(ζ)`. Therefore

```
E{ [(L−ζ')⁺ − (L−ζ)⁺]/(ζ'−ζ) } = −[1 − Ψ(ζ')] − ρ·[Ψ(ζ') − Ψ(ζ)]
```

for some `ρ ∈ [0,1]` packaging the sliver's average. Now let `ζ' ↓ ζ`. The CDF is right-continuous, so `Ψ(ζ') → Ψ(ζ)`, the sliver's probability `Ψ(ζ')−Ψ(ζ) → 0`, and the whole expectation tends to `−[1 − Ψ(ζ)]`. Putting it back:

```
∂⁺F_α/∂ζ (w, ζ) = 1 + (1/(1−α))·( Ψ(ζ) − 1 ) = ( Ψ(ζ) − α ) / (1−α).
```

Beautiful — the right derivative is just `(Ψ(ζ) − α)/(1−α)`. Run the mirror argument from the left, `ζ' ↑ ζ`: now the boundary event is `{L ≥ ζ}` with probability `1 − Ψ(ζ⁻)` (left limit, because the point mass at `ζ` itself, if any, now counts on the "≥" side), and I get

```
∂⁻F_α/∂ζ (w, ζ) = ( Ψ(ζ⁻) − α ) / (1−α).
```

The minimizer condition `∂⁻ ≤ 0 ≤ ∂⁺` reads `Ψ(ζ⁻) ≤ α ≤ Ψ(ζ)`. But that is *exactly* the definition of `ζ` being an α-quantile of `L`: the CDF crosses level `α` at `ζ`. The smallest such `ζ` is precisely `VaR_α(w)`, and if the quantile is unique then the minimizing `ζ` is that VaR. If the distribution has a flat quantile interval, the whole interval minimizes `F`; the lower endpoint is the VaR. That still kills the chicken-and-egg problem: I never had to compute VaR first, and from the argmin interval I can recover it.

Now the payoff — what is the minimum *value*? Evaluate `F` at `ζ* = VaR_α(w)`. Take the clean continuous case first, where `Ψ` has no jump at `ζ*`, so `Ψ(ζ*) = α` exactly. Then

```
F_α(w, ζ*) = ζ* + (1/(1−α)) · E[(L − ζ*)⁺].
```

The expectation only sees `L ≥ ζ*`, which has probability `1 − Ψ(ζ*) = 1 − α`. So `E[(L−ζ*)⁺] = (1−α)·E[ L − ζ* | L ≥ ζ* ]`, and the `(1/(1−α))` cancels it:

```
F_α(w, ζ*) = ζ* + E[ L − ζ* | L ≥ ζ* ] = E[ L | L ≥ ζ* ] = φ_α(w).
```

The minimum value of `F` *is* the tail-mean — exactly the conditional value-at-risk I wanted. So

```
φ_α(w) = min_ζ F_α(w, ζ),   with VaR_α(w) as the lower endpoint of the minimizing ζ's.
```

I should pause on the discrete case, because scenarios always have atoms and that's where the naive `E[L | L ≥ VaR]` got incoherent. When `Ψ` jumps at `ζ*` — a probability atom of mass `Ψ(ζ*) − Ψ(ζ*⁻)` sits exactly at the VaR — there's a genuine ambiguity: the event `{L ≥ ζ*}` carries probability strictly more than `1 − α`, and `{L > ζ*}` carries strictly less. Neither conditional expectation has probability exactly `1−α`, so neither "`E[L | L ≥ VaR]`" nor "`E[L | L > VaR]`" is the honest "mean of the upper `1−α` of the distribution." But notice my formula doesn't ask that question. The derivative computation above went through for general `Ψ` (that's why I kept `Ψ(ζ⁻)` and `Ψ(ζ)` separate) — the minimizer condition `Ψ(ζ⁻) ≤ α ≤ Ψ(ζ)` and the minimum value `min_ζ F` are well-defined regardless of atoms. What `F` computes is the mean of the *α-tail distribution* with the atom at `ζ*` correctly *split*: a fraction of the threshold's mass is assigned weight `(Ψ(ζ*) − α)` and the rest goes to the strictly-greater part, so that the pieces sum to exactly `1 − α`. Concretely, the minimum equals a weighted average `λ·VaR_α + (1−λ)·E[L | L > VaR_α]` with `λ = (Ψ(ζ*) − α)/(1−α)` chosen precisely so the tail weight is `1−α`. The minimization formula handles the atom automatically; I don't have to special-case it. That's the thing the conditional-expectation definitions got wrong and `F` gets right.

Now the part I actually came for: minimizing over the portfolio. I want `min_{w ∈ X} φ_α(w)`. Substituting the formula, that's `min_{w∈X} min_ζ F_α(w, ζ)`, and minimizing over `ζ` for each `w` and then over `w` is the same as minimizing jointly:

```
min_{w ∈ X} φ_α(w) = min_{(w, ζ) ∈ X × ℝ} F_α(w, ζ).
```

Why is this a *good* swap and not just an algebraic shuffle? Because the joint object is convex. Look at `F_α(w, ζ) = ζ + (1/(1−α))E[(L(w,y) − ζ)⁺]`. The loss is `L(w,y) = −R(y)ᵀw`, *linear* in `w`, so `L(w,y) − ζ` is linear (hence convex) jointly in `(w, ζ)`; `(·)⁺` is convex and nondecreasing, so `(L(w,y) − ζ)⁺` is convex in `(w, ζ)`; the expectation of convex is convex; adding the linear `ζ` keeps it convex. So `F_α` is **jointly convex in `(w, ζ)`**, and over a convex feasible set `X` the joint minimization is a single convex program — no local minima, global optimum reachable. Contrast that with `φ_α(w)` written as a tail-mean over a quantile-of-`w`: same value at the optimum, but the convex *route* only appears once I lift to the `(w, ζ)` variables. The auxiliary variable `ζ` isn't a trick for its own sake; it's what convexifies the problem.

This also retroactively cures the coherence complaint. When `L(w,y)` is linear in `w`, `F_α` is jointly sublinear (convex + positively homogeneous), so `φ_α` is sublinear in the loss, i.e. subadditive *and* positively homogeneous — the two axioms VaR flunked. Monotonicity and translation-equivariance fall straight out of the `min_ζ` formula (shift the loss by a constant and the optimal `ζ` shifts with it). So `φ_α` is coherent, and unlike Artzner's abstract worst-case expectation it comes with a concrete convex program to minimize. And since the tail-mean always sits at or above the quantile (`φ_α(w) ≥ VaR_α(w)` — averaging the tail can't undershoot its lowest point), a portfolio with small CVaR automatically has small VaR. I get the coherent, tail-aware objective *and* a VaR bound for free.

Now make it solvable on actual scenario data. Replace the expectation by the empirical average over the `N` equally-weighted samples — each loss `L_k(w) = −R_kᵀw`:

```
F̃_α(w, ζ) = ζ + (1/(N(1−α))) · Σ_{k=1}^N ( L_k(w) − ζ )⁺.
```

This is convex and piecewise linear in `(w, ζ)`, but the `(·)⁺` is a kink, and solvers want it linear. The standard convexification of a `max` is the epigraph lift: a term `(L_k − ζ)⁺ = max{L_k − ζ, 0}` is the smallest `u_k` with `u_k ≥ L_k − ζ` *and* `u_k ≥ 0`. So introduce one auxiliary `u_k` per scenario, demand those two linear inequalities, and minimize with `u_k` *in place of* the positive part:

```
minimize_{w, ζ, u}   ζ + (1/(N(1−α))) Σ_k u_k
subject to           u_k ≥ L_k(w) − ζ,   u_k ≥ 0   for all k,
                     w ∈ X.
```

I should check the epigraph relaxation is tight, i.e. that at the optimum `u_k` really equals `(L_k − ζ)⁺` rather than floating above it. It does, because each `u_k` appears in the objective only through `+ (1/(N(1−α))) u_k` with a *positive* coefficient, so the minimizer drives each `u_k` down to the lowest value its two constraints allow, namely `max{L_k − ζ, 0}`. No `u_k` has any incentive to sit higher. So the relaxed LP and the original piecewise-linear problem have the same optimal value and the same `(w, ζ)`. And with `L_k(w) = −R_kᵀw` linear in `w`, every constraint here is linear in `(w, ζ, u)`: the objective is linear, `u_k ≥ −R_kᵀw − ζ` is linear, `u_k ≥ 0` is linear, and `X` (budget, box bounds, optional return floor) is linear. The whole thing is a **linear program** — exactly the large-scale-tractable form I was after, and the dimension is `n + 1 + N` variables with `2N` plus a handful of structural constraints, which scales fine.

Let me write it the way a convex-modelling layer actually wants it. I'll mirror the scenario-LP structure: a weight vector `w`, a scalar threshold variable I'll call `zeta`, a per-scenario vector `u`, the objective `zeta + (1/(N(1−α))) Σ u_k`, and the two constraint families. I'll keep the sign convention that the data are *returns* (gains) `R`, so the loss is `−Rᵀw` and the exceedance constraint `u_k ≥ L_k − ζ` becomes `u_k ≥ −R_kᵀw − ζ`, i.e. `R_kᵀw + ζ + u_k ≥ 0`.

```python
import cvxpy as cp
import numpy as np

class CVaREfficientPortfolio:
    """Minimize level-alpha CVaR of the loss L = -R w over weights w in X,
    via the auxiliary-variable LP:
        min_{w, zeta, u}  zeta + (1/(N(1-alpha))) * sum_k u_k
        s.t.  u_k >= L_k(w) - zeta,  u_k >= 0,  w in X.
    The optimal value is CVaR; zeta is an alpha-quantile threshold."""

    def __init__(self, expected_returns, returns, alpha=0.95, weight_bounds=(0, 1)):
        self.returns = np.asarray(returns)          # (N scenarios, n assets), gains R_k
        self.expected_returns = np.asarray(expected_returns)
        self.N, self.n = self.returns.shape
        self.alpha = alpha                          # confidence level
        self.lower, self.upper = weight_bounds
        self.w = cp.Variable(self.n)                # portfolio weights
        self.zeta = cp.Variable()                   # candidate alpha-quantile threshold
        self.u = cp.Variable(self.N)                # per-scenario loss-exceedance epigraph vars

    def _feasible_region(self, market_neutral=False):
        cons = [self.w >= self.lower, self.w <= self.upper]
        cons.append(cp.sum(self.w) == (0 if market_neutral else 1))
        return cons

    def _cvar_expr(self):
        # F_alpha = zeta + (1 / (N (1 - alpha))) * sum_k u_k     (the minimization formula)
        return self.zeta + 1.0 / (self.N * (1 - self.alpha)) * cp.sum(self.u)

    def _cvar_constraints(self):
        # epigraph of the positive part: u_k >= (L_k - zeta)  and  u_k >= 0,
        # with L_k = -R_k . w, so R_k . w + zeta + u_k >= 0.
        return [self.u >= 0,
                self.returns @ self.w + self.zeta + self.u >= 0]

    def min_cvar(self, market_neutral=False):
        objective = self._cvar_expr()
        constraints = self._feasible_region(market_neutral) + self._cvar_constraints()
        cp.Problem(cp.Minimize(objective), constraints).solve()
        return self.w.value                          # self.zeta.value is an alpha-quantile

    def efficient_return(self, target_return, market_neutral=False):
        # minimise CVaR subject to a floor on expected return: convex, still an LP.
        objective = self._cvar_expr()
        constraints = self._feasible_region(market_neutral) + self._cvar_constraints()
        constraints.append(self.expected_returns @ self.w >= target_return)
        cp.Problem(cp.Minimize(objective), constraints).solve()
        return self.w.value

    def efficient_risk(self, target_cvar, market_neutral=False):
        # maximise expected return subject to a CVaR ceiling -> trace the frontier.
        objective = self.expected_returns @ self.w
        constraints = self._feasible_region(market_neutral) + self._cvar_constraints()
        constraints.append(self._cvar_expr() <= target_cvar)
        cp.Problem(cp.Maximize(objective), constraints).solve()
        return self.w.value
```

So the whole chain, end to end: minimizing VaR over `w` is hopeless because a quantile is non-convex and tail-blind in `w`, and it even punishes diversification by failing subadditivity. The fix I want is the tail-mean — coherent and tail-aware — but written as `E[L | L ≥ VaR]` it's poisoned by an inner quantile-of-`w`. Freeing the threshold into an independent variable `ζ` and forming `F_α(w,ζ) = ζ + (1/(1−α))E[(L−ζ)⁺]` breaks that dependence: minimizing `F` over `ζ` makes the quantile interval appear as the argmin and lands the value on the tail-mean, and it does so *convexly* and even gets the discrete atom-splitting right. Because `F` is jointly convex in `(w, ζ)`, I minimize over the portfolio and the threshold *together* in one convex program — which, on scenario data with the positive part lifted to epigraph variables `u_k ≥ L_k − ζ, u_k ≥ 0`, is a plain linear program, with VaR recoverable as the lower endpoint of the optimal threshold interval.
