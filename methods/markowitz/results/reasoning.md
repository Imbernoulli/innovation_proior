Let me start from the thing that actually bugs me, which is that the standard advice for picking a portfolio and the thing every sensible investor actually does are flatly inconsistent, and nobody seems to treat that as a problem.

The standard advice, the one you get from the present-value tradition — Williams' investment-value book, the discounting view Hicks uses — is: form, for each security, the discounted value of its anticipated future returns, call it `R_i`, and then prefer the securities with the larger `R_i`. Risk, in this picture, is handled by bumping the discount rate a little or padding the "anticipated" return. Williams is actually explicit that risk isn't a real obstacle, because diversification washes it out: gains offset losses, and you end up earning the pure interest rate, so the net risk is nil. Fine. So what portfolio does this rule actually tell me to hold?

Let me just write it down. I have `N` securities, weights `X_i >= 0` (no short sales for now), fully invested so `sum_i X_i = 1`. Each security has its discounted return `R_i`, a fixed number that doesn't depend on how much of it I buy. The discounted return of the whole portfolio is `R = sum_i X_i R_i`. And now I stare at this. The `R_i` are constants; the `X_i` are non-negative and sum to one. So `R` is just a *weighted average* of the `R_i` with weights I get to choose, subject to them being a probability vector. How do I maximize a weighted average of fixed numbers when I control the weights? I dump everything onto the largest one. Set `X_a = 1` for the security with the biggest `R_a`, zero everywhere else. If a few are tied for the max, any split among *those* is equally good, but there's never any reason to touch a security that isn't tied for first.

So this rule never — not for any choice of discount rates, not for any way of forming the anticipated returns, not ever — prefers a diversified portfolio over the best undiversified one. It always sends me to a single corner of the simplex. (And if I let short sales back in, it's worse: I'd lever infinitely into the top security. The non-negativity is the only thing keeping the answer finite, and even then it's a corner.) That can't be right. Diversification isn't some irrational habit; it's observed everywhere and it's obviously sensible. A rule that *structurally cannot* recommend spreading your money has to be rejected as a guide to behavior. Not patched — rejected. The defect isn't in the numbers I plug in; it's in the shape of the objective. A linear objective over a simplex is always maximized at a vertex. If I want diversification to ever be optimal, the objective cannot be linear in `X`.

Now, people have a patch for this, and I want to see if it survives. The patch: don't put everything in the single top security; instead spread your money across *all* the securities that share the maximum expected return, and lean on the law of large numbers — with enough independent draws, the realized average yield will be almost exactly the expected yield, so you get the high mean and the variance melts away. This is the Bernoulli-law-of-large-numbers intuition that's been seducing people since forever, and it's what lets Williams claim net risk is nil.

Does it hold? The law of large numbers needs the things you're averaging to be, if not independent, at least not strongly co-moving — its whole force comes from independent errors cancelling. But security returns are *not* like that. They're heavily intercorrelated: firms in the same industry tend to have bad years together; the whole market moves together. When the things you average all rise and fall in lockstep, averaging them doesn't cancel anything. So the premise is empirically false for securities, and the conclusion — that diversification drives variance to zero — is false with it. Diversification reduces dispersion, but it cannot eliminate it, precisely because the cross-correlations don't vanish. And there's a second crack: the patch quietly assumes there's one portfolio that is simultaneously highest-mean and lowest-variance. Why would that be true? The security with the best mean need not be the one that combines to the smallest variance. So even the existence of the "have it all" portfolio it commends is an assumption, not a fact.

So both standard rules are out, and the reason both are out is the same: they pretend the only thing that matters about a portfolio's return is its mean, and they treat risk as something that either doesn't exist or auto-cancels. I need to take the variance of the portfolio seriously as its own object. Let me actually compute it and see what it depends on, because I suspect that's where the real structure is.

The portfolio return is `R = sum_i X_i R_i`, a weighted sum of random variables, weights fixed by me. The mean is easy and I already used it: expectation is linear, so `E = sum_i X_i mu_i`, with `mu_i = E[R_i]`. The variance is the thing I keep hearing is "not so simple," so let me grind it out from the definition. `V = E[(R - E[R])^2] = E[(sum_i X_i (R_i - mu_i))^2]`. Expand the square of the sum:

`V = E[ sum_i sum_j X_i X_j (R_i - mu_i)(R_j - mu_j) ] = sum_i sum_j X_i X_j E[(R_i - mu_i)(R_j - mu_j)]`.

That inner expectation is exactly the covariance of `R_i` and `R_j`; call it `sigma_ij` (and `sigma_ii` is just the variance of security `i`). So

`V = sum_i sum_j X_i X_j sigma_ij`.

Let me split the diagonal from the off-diagonal to actually see it:

`V = sum_i X_i^2 sigma_ii + sum_i sum_{j != i} X_i X_j sigma_ij`.

There it is. The first piece is what the law-of-large-numbers crowd implicitly thinks is the whole thing — own variances, weighted by `X_i^2`, and *that* piece does shrink as you spread out (each `X_i^2` gets tiny). But the second piece, the cross terms, is the off-diagonal mass of the covariance matrix, and writing `sigma_ij = rho_ij sigma_i sigma_j`, it's a double sum over every *pair*. As you add securities, the number of own-variance terms grows like `N` but the number of covariance terms grows like `N^2`. So in a well-spread portfolio the variance is dominated, asymptotically, by the *average covariance*, not by any individual security's own variance. If all the `rho_ij` were zero, the cross terms vanish and the law of large numbers would be right — variance would go to zero. They're not zero. That residual, the average covariance, is the floor diversification can never push through. This is the precise statement of why Williams is wrong: it's not that diversification doesn't help, it's that it can only kill the diagonal, never the off-diagonal.

And this reframes what "risk of a security" even means. Look at the marginal effect of nudging one weight: `dV/dX_i = 2 sum_j sigma_ij X_j`. The contribution of security `i` to the portfolio's variance is governed by its covariance with *everything I already hold*, not by its own `sigma_ii` in isolation. A wildly volatile security that moves opposite to the rest of my book can *reduce* my variance. So I can't evaluate a security on its own; whether I want it depends entirely on the company it keeps. That also tells me what *kind* of diversification works. It's not "hold a lot of names" — sixty railroads all share the railroad covariance and barely help. It's "hold names with low covariance among themselves," which in practice means spreading across industries with different economic drivers. The right kind of diversification, for the right reason, falls straight out of which term of `V` you're trying to shrink.

Good. So I have two summary numbers for any portfolio: `E = mu^T X` (linear) and `V = X^T Sigma X` (a convex quadratic, since `Sigma` is a covariance matrix, hence positive semidefinite). Now what's the rule? My first instinct is to go back to expected-utility — von Neumann–Morgenstern and Savage have just put choice-under-uncertainty on axiomatic footing, so the "right" answer is supposedly: write down `U(W)`, maximize `E[U(W)]`. But to *use* that I'd have to extract the investor's entire utility function over wealth, and I have no operational way to do that, and it doesn't hand me a portfolio from the inputs I actually possess, which are just `mu` and `Sigma`. I want something that works off the first two moments and nothing else. So let me set utility aside and ask only: the investor likes `E`, dislikes `V`. Can I get a rule out of just that ordering?

The naive move is "maximize `E` and minimize `V`," but those fight each other — you generally can't do both, because the maximum-`E` portfolio is some aggressive corner and the minimum-`V` portfolio is some balanced interior point, and they're different portfolios. There's a rate at which I can buy extra `E` by accepting extra `V`, or shed `V` by giving up `E`. So there is no single portfolio to "maximize"; there's a trade-off curve. That actually feels like the honest answer. Instead of collapsing the two numbers into one, look at the whole *set* of attainable `(E, V)` pairs as `X` ranges over the feasible portfolios, and notice that the investor would never want most of them. If portfolio `A` has the same `E` as `B` but lower `V`, nobody who dislikes variance picks `B`. If `A` has the same `V` as `B` but higher `E`, nobody picks `B`. So strip out every dominated point. What's left is the *efficient set*: the portfolios with minimum `V` for their level of `E` (equivalently, maximum `E` for their level of `V`). That's the rule — not "pick the optimum," but "compute the efficient frontier, then let the investor (or a chosen risk attitude) pick a point on it." This is the move that lets me dodge the utility function entirely: I don't need to know how the investor trades `E` against `V`, I just need to hand them the menu of all non-dominated trades.

(Roy, working at the same time, refuses the utility function for the same reason but commits to one point on this menu: pick a "disaster level" `d` you most fear falling below, and minimize the probability of landing below it. If you summarize the portfolio by mean and standard deviation, minimizing `P(R < d)` means pushing `d` as many standard deviations below the mean as possible, i.e. maximizing `(mu_P - d)/sigma_P`. He writes down the very same covariance expression for `V` and lands on a mean–variance efficient set too. Same surface; he just picks the disaster-ratio point on it instead of leaving the choice open.)

Now I have to actually *find* this efficient set. Let me get concrete with three securities, because then the budget constraint `X_1 + X_2 + X_3 = 1` lets me eliminate one variable and work in a plane. Set `X_3 = 1 - X_1 - X_2`. The feasible region is the three inequalities `X_1 >= 0`, `X_2 >= 0`, `X_3 = 1 - X_1 - X_2 >= 0`, which carve out a triangle in the `(X_1, X_2)` plane. Substitute into `E` and `V`. `E` is affine in `(X_1, X_2)` — for instance `E = mu_3 + (mu_1 - mu_3) X_1 + (mu_2 - mu_3) X_2`. So the level sets of `E`, the *isomean* curves, are parallel straight lines (the gradient of an affine function is constant). `V` is a quadratic in `(X_1, X_2)`, and because `Sigma` is positive (semi)definite it's a convex quadratic — its level sets, the *isovariance* curves, are concentric ellipses, all centered on the single point that minimizes `V` over the plane (ignoring the triangle). Call that unconstrained center `X-hat`; variance grows as you move away from it.

So picture the plane: parallel isomean lines marching across it in one direction, nested isovariance ellipses around `X-hat`. To trace the efficient set, fix a target return `E` — that's one isomean line — and ask: among all points on that line, which has the least variance? Geometrically, slide inward through the ellipses until the isomean line just kisses one — the point of tangency. That tangent point is the min-variance portfolio for that `E`. Now let `E` vary: the isomean line translates (same slope, different intercept), and the tangency point moves. Where does it go? At a tangency, the gradient of `V` is parallel to the gradient of `E`; the gradient of `V` is linear in `X` (it's `2 Sigma X` plus constants from the substitution), and the gradient of `E` is a fixed direction. Setting "`grad V` parallel to a fixed vector" is a linear condition on `X`. So the locus of tangency points, as `E` sweeps, is a *straight line*. That's the **critical line**. It passes through `X-hat` (where the smallest ellipse degenerates to the center, trivially the min-`V` point for `E = E-hat`), and moving along it in either direction increases `V`.

But that's the *unconstrained* min-variance-per-`E` locus. I haven't used the triangle yet. The efficient set has to live inside the feasible triangle. So: start at the global minimum-variance feasible portfolio. If `X-hat` is inside the triangle, that's it; start there. Walk along the critical line in the direction of increasing `E` (and increasing `V`). I keep following the critical line as long as I stay inside the triangle. The moment the critical line hits a boundary edge of the triangle — meaning one of the securities' weights has been driven to zero — I can't continue in the interior. From there the constrained min-`V`-per-`E` point is forced to ride *along that edge* (one security pinned at zero, the other two trading off), until I reach the vertex of maximum attainable `E`. So the efficient set, in weight space, is a sequence of connected line segments: a piece of the critical line, then a piece of a boundary edge, ending at the max-`E` corner. And if I plot `V` against `E` for these portfolios, each linear segment in `X` maps to a parabola segment in `(E, V)` (because `V` is quadratic along a line) — the frontier is a string of connected parabola arcs, convex, rising.

This is the whole picture, and it generalizes. With `N` securities I can't draw it, but the logic is identical and it's screaming "algorithm" at me. At any point on the efficient frontier some securities are strictly interior (weights properly between their bounds — call these the *free* set `F`) and the rest are pinned at a bound, almost always at zero under long-only (the *bounded* set `B`). On a stretch where the membership of `F` and `B` doesn't change, the constrained problem is just a min-variance-per-`E` over the free coordinates with the bounded ones held fixed — exactly the tangency condition — so the solution moves along a straight line (the critical line for that particular free set). The frontier is piecewise linear, and it kinks exactly when `F`/`B` membership changes: a free weight gets driven to a bound, or a bounded weight wants to come off its bound. If I can (1) write the linear solution on each segment in closed form and (2) detect the next kink, I can hop from kink to kink — "turning point" to turning point — and trace the entire frontier exactly, no matter how big `N` is. Let me derive both.

I'll parametrize the trade-off the clean way, with a Lagrange multiplier on return instead of a hard target, because then sweeping one scalar sweeps the whole frontier. Consider, for a scalar `lambda >= 0`,

minimize over `w` of `(1/2) w^T Sigma w - lambda * mu^T w`, subject to `1^T w = 1` and box bounds `l <= w <= u`.

At `lambda = 0` this is pure minimum variance (the bottom tip of the frontier); as `lambda -> infinity` the return term dominates and it's the maximum-return corner (the top). Every efficient portfolio is the solution for some `lambda` in between, and increasing `lambda` walks me up the frontier. (Equivalently one writes `max mu^T w - (delta/2) w^T Sigma w` with risk-aversion `delta = 1/lambda`, or hard-targets `mu^T w = m`; same family, same frontier.) Form the Lagrangian, with `gamma` the multiplier on the budget and `alpha_i, beta_i >= 0` on the lower/upper bounds:

`L = (1/2) w^T Sigma w - lambda mu^T w - gamma (1^T w - 1) - sum_i alpha_i (w_i - l_i) - sum_i beta_i (u_i - w_i)`.

Stationarity in `w`: `Sigma w - lambda mu - gamma 1 - alpha + beta = 0`, together with complementary slackness — `alpha_i = 0` unless `w_i = l_i`, `beta_i = 0` unless `w_i = u_i`. Split into the free set `F` (where `l_i < w_i < u_i`, so `alpha_i = beta_i = 0`) and the bounded set `B` (where `w_i` is pinned at `l_i` or `u_i`, a known value). For the free coordinates the multipliers drop and the stationarity rows read

`Sigma_FF w_F + Sigma_FB w_B = lambda mu_F + gamma 1_F`,

so, with `Sigma_FF` invertible (it's a principal block of a PD matrix),

`w_F = Sigma_FF^{-1} ( lambda mu_F + gamma 1_F - Sigma_FB w_B )`.

This is the closed form on a segment: `w_F` is **affine in `lambda`** (and in the auxiliary `gamma`), which is the critical line, now in `N` dimensions. I still have to pin `gamma` from the budget. The full budget is `1_F^T w_F + 1_B^T w_B = 1`, i.e. `1_F^T w_F = 1 - 1_B^T w_B`. Substitute the expression for `w_F`:

`1_F^T Sigma_FF^{-1} ( lambda mu_F + gamma 1_F - Sigma_FB w_B ) = 1 - 1_B^T w_B`.

Solve for `gamma`:

`gamma = [ (1 - 1_B^T w_B) + 1_F^T Sigma_FF^{-1} Sigma_FB w_B - lambda * (1_F^T Sigma_FF^{-1} mu_F) ] / ( 1_F^T Sigma_FF^{-1} 1_F )`.

So `gamma` is itself affine in `lambda`, and substituting it back, `w_F` is purely affine in `lambda`: `w_F(lambda) = a_F + lambda b_F` for vectors `a_F, b_F` built from `Sigma_FF^{-1}`, `mu_F`, `1_F`, and the frozen `w_B`. That is the critical line, parametrically, with no geometry needed. (Special check: if there are no bounded assets — everything free, no `w_B` — then `w = a + lambda b` with `a = (1/(1^T Sigma^{-1} 1)) Sigma^{-1} 1` and the variance-only end `lambda = 0` gives `w* = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)`, the textbook global minimum-variance portfolio, with variance `1 / (1^T Sigma^{-1} 1)`, so `sigma_min = sqrt(1 / (1^T Sigma^{-1} 1))`. Reassuring — the unconstrained two-fund hyperbola falls out as the special case.)

Now the turning points. As I increase `lambda` along a segment, `w_F(lambda) = a_F + lambda b_F` slides linearly. Two kinds of event can end the segment. **Event (a): a free weight hits a bound.** For each free `i`, find the `lambda` at which `w_i(lambda)` equals its lower bound `l_i` or upper bound `u_i` — that's one linear equation per asset, `a_i + lambda b_i = boundary`, solved by `lambda = (boundary - a_i)/b_i`. **Event (b): a bounded weight wants to leave its bound.** A pinned asset stays pinned only while its multiplier (`alpha_i` or `beta_i`) keeps the right sign; the `lambda` at which that multiplier crosses zero is the `lambda` at which asset `i` should re-enter `F`. Compute the candidate `lambda` for *every* asset under both event types; the next turning point is the binding one — the first event encountered as `lambda` moves. At that `lambda` I update the partition (move the offending asset between `F` and `B`), re-form `Sigma_FF^{-1}` for the new free set, and continue. Between consecutive turning points the portfolio is the linear interpolation of the two endpoint portfolios (since `w` is affine in `lambda` and `lambda` is monotone along the frontier), so once I have all the turning points I have the entire frontier by interpolation. That's the critical-line algorithm: a finite sequence of turning points, each a small linear-algebra update, exact, and it handles arbitrary `l_i, u_i`.

Where do I start the walk? At the extreme-return end (`lambda` large): pour wealth into the highest-`mu` assets, filling each to its upper bound in order of decreasing `mu` until the budget `sum w = 1` is exhausted — the last asset partially filled is the lone free asset, everything above it pinned at its upper bound, everything below at its lower bound. That's the unambiguous max-return corner, the natural top of the frontier; then drive `lambda` down toward zero, picking off turning points, until I reach the global minimum-variance portfolio at the bottom.

Two checks before I trust the algorithm. First, numerical hygiene: a near-singular `Sigma` can throw up spurious turning points that slightly violate `sum w = 1` or the box — I should purge any stored point whose weights breach the budget or a bound beyond a tiny tolerance. Second, the algorithm could emit points that aren't actually on the *upper* (efficient) boundary if it wanders; I should purge any point that's dominated — if a later turning point has higher mean than an earlier one I kept, the earlier one wasn't on the efficient edge and should go. With those two purges the surviving sequence is exactly the efficient frontier.

One more rule I'll want is the single best risk-adjusted point, the tangency / maximum-Sharpe portfolio — the place on the frontier where the line from a reference rate `r` is steepest, i.e. `(mu^T w - r)/sqrt(w^T Sigma w)` is maximized. Two ways. Inside the critical-line picture it's easy: the maximum-Sharpe portfolio lies on the frontier, so it's somewhere on one of the turning-point segments; on each segment `w(a) = a*w0 + (1-a)*w1` for `a` in `[0,1]`, the Sharpe ratio is a smooth function of the scalar `a`, so I just line-search `a` on each segment (golden-section is fine) and take the best across segments. The other way, if I'm solving with a general convex solver instead, the raw maximization of `(mu - r)^T w / sqrt(w^T Sigma w)` isn't convex — ratios of an affine over a square root aren't. The standard fix is a change of variable: let `y = w/k` with a scalar `k > 0`, normalize so that `(mu - r)^T y = 1`, and then *minimize* `y^T Sigma y` subject to `1^T y = k` (and the bounds scaled by `k`, `k >= 0`). That's a convex quadratic program in `(y, k)`; recover `w = y/k`. Maximizing the ratio became minimizing a quadratic on a slice — convex, solvable.

So the rule is settled, and there are two faithful ways to compute it: a general convex-QP formulation (clean, flexible about constraints, one solve per frontier point), and the native critical-line algorithm (exact, gives the *entire* frontier from a handful of linear-algebra steps). Let me write both side by side.

```python
import numpy as np
import cvxpy as cp


def _as_column(x):
    return np.asarray(x, dtype=float).reshape(-1, 1)


def _scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def _weight_bounds(n, weight_bounds):
    if len(weight_bounds) == n and not np.isscalar(weight_bounds[0]):
        bounds = np.asarray(weight_bounds, dtype=float)
        return bounds[:, 0], bounds[:, 1]
    lo, hi = weight_bounds
    lo = np.full(n, -1.0 if lo is None else lo) if np.isscalar(lo) or lo is None else np.asarray(lo, dtype=float)
    hi = np.full(n, 1.0 if hi is None else hi) if np.isscalar(hi) or hi is None else np.asarray(hi, dtype=float)
    return lo, hi


class EfficientFrontier:
    def __init__(self, mu, Sigma, weight_bounds=(0, 1), solver=None):
        self.mu = np.asarray(mu).ravel()                  # expected returns
        self.Sigma = np.asarray(Sigma)                    # covariance (PSD)
        self.n = len(self.mu)
        self.w = cp.Variable(self.n)
        self.lo, self.hi = _weight_bounds(self.n, weight_bounds)
        self.solver = solver
        self._base_constraints = [self.w >= self.lo, self.w <= self.hi]

    def _solve(self, objective, constraints=()):
        constraints = list(self._base_constraints) + list(constraints)
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=self.solver)
        if self.w.value is None:
            raise ValueError(f"optimization failed: {prob.status}")
        return np.asarray(self.w.value).round(16) + 0.0

    def min_volatility(self):
        # bottom tip of the frontier: minimise V = w^T Sigma w
        return self._solve(
            cp.quad_form(self.w, self.Sigma, assume_PSD=True),
            [cp.sum(self.w) == 1],
        )

    def efficient_return(self, target_return):
        # min V s.t. mean >= target  -> one efficient point per target return
        return self._solve(
            cp.quad_form(self.w, self.Sigma, assume_PSD=True),
            [cp.sum(self.w) == 1, self.w @ self.mu >= target_return],
        )

    def efficient_risk(self, target_volatility):
        # max mean s.t. volatility <= target  (minimise -mean)
        if target_volatility < 0:
            raise ValueError("target_volatility must be non-negative")
        variance = cp.quad_form(self.w, self.Sigma, assume_PSD=True)
        return self._solve(
            -(self.w @ self.mu),
            [cp.sum(self.w) == 1, variance <= target_volatility**2],
        )

    def max_quadratic_utility(self, risk_aversion=1.0):
        # the Lagrangian form: max mu^T w - (delta/2) w^T Sigma w
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        util = (self.w @ self.mu
                - 0.5 * risk_aversion * cp.quad_form(self.w, self.Sigma, assume_PSD=True))
        return self._solve(-util, [cp.sum(self.w) == 1])

    def max_sharpe(self, risk_free_rate=0.0):
        # substitute y = w/k, fix (mu-r)^T y = 1, minimise y^T Sigma y
        if np.max(self.mu) <= risk_free_rate:
            raise ValueError("at least one expected return must exceed the reference rate")
        y, k = cp.Variable(self.n), cp.Variable()
        constraints = [
            (self.mu - risk_free_rate) @ y == 1,
            cp.sum(y) == k,
            k >= 0,
            y >= self.lo * k,
            y <= self.hi * k,
        ]
        prob = cp.Problem(cp.Minimize(cp.quad_form(y, self.Sigma, assume_PSD=True)), constraints)
        prob.solve(solver=self.solver)
        if y.value is None or k.value is None:
            raise ValueError(f"optimization failed: {prob.status}")
        return (y.value / k.value).round(16) + 0.0


class CLA:
    def __init__(self, mu, Sigma, weight_bounds=(0, 1)):
        self.mean = np.asarray(mu, float).reshape(-1, 1)  # column vector
        self.cov = np.asarray(Sigma, float)
        lo, hi = _weight_bounds(len(self.mean), weight_bounds)
        self.lB, self.uB = _as_column(lo), _as_column(hi)
        self.w, self.ls, self.g, self.f = [], [], [], []  # weights, lambdas, gammas, free-sets

    @staticmethod
    def _infnone(x):
        return float("-inf") if x is None else x

    def _init_algo(self):
        # max-return corner: fill highest-mu assets to their upper bound until budget spent
        idx = np.argsort(self.mean.ravel())            # ascending mu
        if float(np.sum(self.lB)) > 1 or float(np.sum(self.uB)) < 1:
            raise ValueError("bounds cannot satisfy a fully invested portfolio")
        w = np.copy(self.lB)
        i = len(idx)
        while float(np.sum(w)) < 1:
            i -= 1
            w[idx[i]] = self.uB[idx[i]]
        w[idx[i]] += 1 - float(np.sum(w))               # last asset partially filled = first free
        return [idx[i]], w

    def _compute_w(self, covF_inv, covFB, meanF, wB):
        onesF = np.ones(meanF.shape)
        # gamma from the budget; affine in the current lambda (= self.ls[-1])
        g1 = _scalar(onesF.T @ covF_inv @ meanF)
        g2 = _scalar(onesF.T @ covF_inv @ onesF)
        if wB is None:                                  # every asset free
            g = (-self.ls[-1] * g1 + 1) / g2
            w1 = 0
        else:
            onesB = np.ones(wB.shape)
            w1 = covF_inv @ covFB @ wB                  # Sigma_FF^{-1} Sigma_FB w_B
            g = (-self.ls[-1] * g1 + (1 - _scalar(onesB.T @ wB) + _scalar(onesF.T @ w1))) / g2
        # w_F = -Sigma_FF^{-1} Sigma_FB w_B + gamma Sigma_FF^{-1} 1 + lambda Sigma_FF^{-1} mu
        wF = -w1 + g * (covF_inv @ onesF) + self.ls[-1] * (covF_inv @ meanF)
        return wF, g

    def _compute_lambda(self, covF_inv, covFB, meanF, wB, i, bi):
        # the lambda at which free asset i would reach boundary bi (a turning point)
        onesF = np.ones(meanF.shape)
        c1 = _scalar(onesF.T @ covF_inv @ onesF)
        c2 = covF_inv @ meanF
        c3 = _scalar(onesF.T @ covF_inv @ meanF)
        c4 = covF_inv @ onesF
        c = -c1 * _scalar(c2[i]) + c3 * _scalar(c4[i])
        if abs(c) < 1e-14:
            return None, None
        if isinstance(bi, list):                        # pick lower/upper by sign of c
            bi = _scalar(bi[1]) if c > 0 else _scalar(bi[0])
        else:
            bi = _scalar(bi)
        if wB is None:
            res = (_scalar(c4[i]) - c1 * bi) / c
        else:
            onesB = np.ones(wB.shape)
            l1 = _scalar(onesB.T @ wB)
            l3 = covF_inv @ covFB @ wB
            l2 = _scalar(onesF.T @ l3)
            res = ((1 - l1 + l2) * _scalar(c4[i]) - c1 * (bi + _scalar(l3[i]))) / c
        return res, bi

    def _get_b(self, f):
        return [i for i in range(self.mean.shape[0]) if i not in f]

    @staticmethod
    def _reduce(matrix, rows, cols):
        if len(rows) == 0 or len(cols) == 0:
            return None
        return matrix[np.ix_(rows, cols)]

    def _get_matrices(self, f):
        b = self._get_b(f)
        return (
            self._reduce(self.cov, f, f),
            self._reduce(self.cov, f, b),
            self._reduce(self.mean, f, [0]),
            self._reduce(self.w[-1], b, [0]),
        )

    def _purge_num_err(self, tol=1e-9):
        i = 0
        while i < len(self.w):
            bad_budget = abs(float(np.sum(self.w[i])) - 1) > tol
            bad_bounds = np.any(self.w[i] < self.lB - tol) or np.any(self.w[i] > self.uB + tol)
            if bad_budget or bad_bounds:
                del self.w[i], self.ls[i], self.g[i], self.f[i]
            else:
                i += 1

    def _purge_excess(self):
        i, repeat = 0, False
        while True:
            if not repeat:
                i += 1
            if i == len(self.w) - 1:
                break
            mu_i = _scalar(self.w[i].T @ self.mean)
            repeat = False
            for j in range(i + 1, len(self.w)):
                if mu_i < _scalar(self.w[j].T @ self.mean):
                    del self.w[i], self.ls[i], self.g[i], self.f[i]
                    repeat = True
                    break

    def _solve(self):
        f, w = self._init_algo()
        self.w, self.ls, self.g, self.f = [np.copy(w)], [None], [None], [f[:]]
        while True:
            l_in = i_in = bi_in = None
            if len(f) > 1:
                covF, covFB, meanF, wB = self._get_matrices(f)
                covF_inv = np.linalg.inv(covF)
                for j, asset in enumerate(f):
                    lam, bi = self._compute_lambda(
                        covF_inv, covFB, meanF, wB, j, [self.lB[asset], self.uB[asset]]
                    )
                    if lam is not None and lam > CLA._infnone(l_in):
                        l_in, i_in, bi_in = lam, asset, bi

            l_out = i_out = None
            if len(f) < self.mean.shape[0]:
                for asset in self._get_b(f):
                    covF, covFB, meanF, wB = self._get_matrices(f + [asset])
                    covF_inv = np.linalg.inv(covF)
                    lam, _ = self._compute_lambda(
                        covF_inv, covFB, meanF, wB, meanF.shape[0] - 1, self.w[-1][asset]
                    )
                    if lam is not None and (self.ls[-1] is None or lam < self.ls[-1]):
                        if lam > CLA._infnone(l_out):
                            l_out, i_out = lam, asset

            if (l_in is None or l_in < 0) and (l_out is None or l_out < 0):
                self.ls.append(0)
                covF, covFB, meanF, wB = self._get_matrices(f)
                covF_inv = np.linalg.inv(covF)
                meanF = np.zeros(meanF.shape)
            else:
                if CLA._infnone(l_in) > CLA._infnone(l_out):
                    self.ls.append(l_in)
                    f.remove(i_in)
                    w[i_in] = bi_in
                else:
                    self.ls.append(l_out)
                    f.append(i_out)
                covF, covFB, meanF, wB = self._get_matrices(f)
                covF_inv = np.linalg.inv(covF)

            wF, gamma = self._compute_w(covF_inv, covFB, meanF, wB)
            for j, asset in enumerate(f):
                w[asset] = wF[j]
            self.w.append(np.copy(w))
            self.g.append(gamma)
            self.f.append(f[:])
            if self.ls[-1] == 0:
                break

        self._purge_num_err()
        self._purge_excess()

    def min_volatility(self):
        if not self.w:
            self._solve()
        variances = [_scalar(w.T @ self.cov @ w) for w in self.w]
        return self.w[int(np.argmin(variances))].ravel()

    def _golden_section(self, obj, a, b, args=(), minimum=True, tol=1e-9):
        sign = 1 if minimum else -1
        r, c = 0.618033989, 1 - 0.618033989
        x1, x2 = r * a + c * b, c * a + r * b
        f1, f2 = sign * obj(x1, *args), sign * obj(x2, *args)
        for _ in range(int(np.ceil(-2.078087 * np.log(tol / abs(b - a))))):
            if f1 > f2:
                a, x1, f1 = x1, x2, f2
                x2 = c * a + r * b
                f2 = sign * obj(x2, *args)
            else:
                b, x2, f2 = x2, x1, f1
                x1 = r * a + c * b
                f1 = sign * obj(x1, *args)
        return (x1, sign * f1) if f1 < f2 else (x2, sign * f2)

    def _eval_sr(self, a, w0, w1, risk_free_rate):
        w = a * w0 + (1 - a) * w1
        ret = _scalar(w.T @ self.mean) - risk_free_rate
        vol = _scalar(w.T @ self.cov @ w) ** 0.5
        return ret / vol

    def max_sharpe(self, risk_free_rate=0.0):
        if not self.w:
            self._solve()
        candidates, ratios = [], []
        for i in range(len(self.w) - 1):
            w0, w1 = np.copy(self.w[i]), np.copy(self.w[i + 1])
            a, ratio = self._golden_section(
                self._eval_sr, 0, 1, args=(w0, w1, risk_free_rate), minimum=False
            )
            candidates.append(a * w0 + (1 - a) * w1)
            ratios.append(ratio)
        return candidates[int(np.argmax(ratios))].ravel()

    def efficient_frontier(self, points=100):
        if not self.w:
            self._solve()
        mus, sigmas, weights = [], [], []
        per_segment = max(points // max(len(self.w), 1), 2)
        grid = np.linspace(0, 1, per_segment)[:-1]
        for i in range(len(self.w) - 1):
            if i == len(self.w) - 2:
                grid = np.linspace(0, 1, per_segment)
            for a in grid:
                w = self.w[i + 1] * a + (1 - a) * self.w[i]
                weights.append(np.copy(w).ravel())
                mus.append(_scalar(w.T @ self.mean))
                sigmas.append(_scalar(w.T @ self.cov @ w) ** 0.5)
        return mus, sigmas, weights
```

The causal chain, start to finish: the present-value rule maximizes a linear function of the weights, so it lands on a single corner and can never recommend diversification, which is observed and sensible — so a usable rule cannot be linear in the weights. The law-of-large-numbers patch assumes the off-diagonal of the covariance matrix vanishes, which is empirically false for securities, so it can't be the fix either. Taking variance seriously, `V = w^T Sigma w`, splits into a diagonal that diversification kills and an off-diagonal of covariances it cannot — so a security's risk is its covariance with the rest of the book, and the right diversification is across low-covariance names. Return and variance trade off rather than co-optimize, so the object isn't a single optimum but the efficient frontier — minimum variance per level of return — which sidesteps needing a utility function entirely. Stationarity of the Lagrangian `(1/2)w^T Sigma w - lambda mu^T w` under the budget and box constraints makes the free weights affine in `lambda` (the critical line), the frontier piecewise-linear, kinking only when an asset enters or leaves the free set; hopping those turning points traces the whole frontier exactly. And dropping the bounds recovers `Sigma^{-1} 1 / (1^T Sigma^{-1} 1)`, the classic two-fund minimum-variance portfolio, as the bottom of that frontier. The code is just those two readings of the same object: a convex QP per point, or the critical-line walk for all points at once.
