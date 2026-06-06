# Optimality of $(s,S)$ policies via $K$-convexity

## Problem

A single product, periodic review, finite horizon of $T$ periods. At the start of period $t$ the inventory level is $x$; order $u\ge0$ (delivered immediately), reaching post-order level $y=x+u$; then random demand $w_t$ (known distribution) is realized and the inventory becomes $y-w_t$ (backlogging allowed, so it can go negative). Costs per period: a fixed setup charge $K$ whenever $u>0$, a linear purchase cost $c$ per unit, and an end-of-period convex holding/shortage cost $L(x)=h\,x^+ + p\,x^-$. A discount factor $\gamma\in(0,1]$ applies, and a convex nonnegative terminal cost $v_{T+1}$ on the final inventory. Minimize total expected discounted cost. Standing assumptions: $c+h>0$ and $p>c$ (otherwise never ordering would be optimal).

## Bellman recursion

With inventory $x$ the state and post-order level $y\ge x$ the decision,
$$
J_{T+1}^*(x)=v_{T+1}(x),\qquad
J_t^*(x)=\min_{y\ge x}\Big\{K\,\delta(y-x)+c(y-x)+\bar L_t(y)+\gamma_t\,E_w[J_{t+1}^*(y-w)]\Big\},\quad t=T,\ldots,1,
$$
where $\delta(z)=1$ if $z>0$ else $0$, and $\bar L_t(y)=E_w[L_t(y-w)]$ is the newsvendor expected cost. In the continuous case the single-period minimizer satisfies $F(S)=p/(p+h)$; with atoms it is any quantile satisfying $F(S^-)\le p/(p+h)\le F(S)$. Pulling out the linear term, define
$$
G_t(y)=cy+\bar L_t(y)+\gamma_t\,E_w[J_{t+1}^*(y-w)],\qquad
J_t^*(x)=\min_{y\ge x}\{K\,\delta(y-x)+G_t(y)\}-cx .
$$

## Key idea: $K$-convexity

Without the fixed cost ($K=0$), $G_t$ is convex and coercive, so it has a single minimizer $S_t$ and the optimal rule is **base-stock**: order up to $S_t$ iff $x<S_t$. The convexity of $J_t^*$ is preserved by the recursion, closing the induction.

With $K>0$ the value function develops a downward dent of size at most $K$ and is no longer convex, so that argument collapses. The fix is the weaker invariant that *does* survive the operator.

**Definition.** $f:\mathbb R\to\mathbb R$ is **$K$-convex** ($K\ge0$) if for all $y\le y'$, $\theta\in[0,1]$,
$$
f(\theta y+(1-\theta)y')\le \theta f(y)+(1-\theta)\big(K+f(y')\big),
$$
i.e. on $[y,y']$ the graph lies below the segment from $(y,f(y))$ to $(y',f(y')+K)$ (the right endpoint lifted by $K$). Equivalent forms:
$$
K+f(y+a)\ge f(y)+\tfrac{a}{b}\big[f(y)-f(y-b)\big]\quad(\forall\,y,\ a\ge0,\ b>0);
\qquad
K+f(y)\ge f(x)+f'(x)(y-x)\quad(x\le y,\ f\text{ diff.}).
$$
$K=0$ is ordinary convexity. A $K$-convex function may be non-convex, may have several local minima, and may have downward jumps bounded by $K$.

**Closure (preservation).** (i) If $f_1$ is $K_1$-convex and $f_2$ is $K_2$-convex, then $\alpha f_1+\beta f_2$ ($\alpha,\beta>0$) is $(\alpha K_1+\beta K_2)$-convex. (ii) If $f$ is $K$-convex then $y\mapsto E_w[f(y-w)]$ is $K$-convex (translation preserves $K$-convexity; expectation is linear). Hence if $J_{t+1}^*$ is $K$-convex, $G_t=cy+\bar L_t+\gamma_t E_w[J_{t+1}^*(\cdot-w)]$ is $\gamma_t K$-convex, therefore $K$-convex (since $\gamma_t\le1$), and it is coercive and continuous.

**$(s,S)$ from $K$-convexity.** If $f$ is continuous, coercive, and $K$-convex, let $S$ be a global minimizer and let $s$ be the smallest $z\le S$ with $f(z)=f(S)+K$. Then: (1) $S$ minimizes $f$; (2) $f(s)=f(S)+K$ and $f(y)>f(s)$ for $y<s$; (3) $f$ is decreasing on $(-\infty,s)$; (4) $f(y)\le f(y')+K$ for $s\le y\le y'$. Applied to $G_t$: if $x<s_t$ then $G_t(x)>K+G_t(S_t)$, so ordering up to $S_t$ strictly wins; if $x\ge s_t$ then $G_t(x)\le K+G_t(y)$ for every $y>x$, so not ordering wins, with equality at the boundary allowed. The optimal policy is **$(s_t,S_t)$**: order up to $S_t$ below the reorder boundary, and do not order above it.

**Closing the induction.** Plugging the $(s_t,S_t)$ policy back, $\tilde G_t(x):=J_t^*(x)+cx$ equals the constant $K+G_t(S_t)=G_t(s_t)$ for $x\le s_t$ and equals $G_t(x)$ for $x\ge s_t$. A three-case check against the definition (both points $\ge s_t$: it is $G_t$; both $\le s_t$: constant; $y<s_t<y'$: the flat part sits under the increasing $K$-lifted chord, and the right part is under it by $K$-convexity of $G_t$) shows $\tilde G_t$, hence $J_t^*$, is $K$-convex, nonnegative, continuous. With base case $J_{T+1}^*=v_{T+1}$ ($0$-convex), induction gives an optimal $(s_t,S_t)$ policy at every stage. When $K=0$, $s_t=S_t$ and the policy collapses to base-stock.

**Reading off the parameters:** $S_t=\arg\min_y G_t(y)$; $s_t$ is the smallest $z\le S_t$ with $G_t(z)=G_t(S_t)+K$. On an integer grid, the implementation reports the largest grid state whose selected post-order level is still $S_t$.

## Code

Backward DP on a discretized inventory grid. The expensive part $H_t(y)=\bar L_t(y)+\gamma_t E_w[\theta_{t+1}(y-w)]$ is precomputed once per $(t,y)$ and reused across all $x$ in the inner minimization. $S_t$ is the order-up-to level at the lowest $x$; the reported grid reorder point is the largest $x$ whose selected post-order level remains $S_t$.

```python
import numpy as np
import warnings
from scipy.stats import norm

def _period_list(v, T, name):
    if isinstance(v, (list, tuple, np.ndarray)):
        values = list(v)
        if len(values) == T:
            return [None] + values
        if len(values) == T + 1:
            return values
        raise ValueError(f"{name} must have length {T} or {T + 1}")
    return [None] + [v] * T

def _normal_loss(y, mu, sigma):
    if sigma <= 0:
        return max(mu - y, 0.0), max(y - mu, 0.0)
    z = (y - mu) / sigma
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    shortage = sigma * phi + (mu - y) * (1 - Phi)
    holding = sigma * phi + (y - mu) * Phi
    return shortage, holding

def _eoq_with_backorders(K, h, p, mean):
    if K <= 0 or h <= 0 or p <= 0 or mean <= 0:
        return 0.0
    return np.sqrt(2.0 * K * mean * (h + p) / (h * p))

def finite_horizon_dp(num_periods, holding_cost, stockout_cost,
                      terminal_holding_cost, terminal_stockout_cost,
                      purchase_cost, fixed_cost,
                      demand_mean=None, demand_sd=None, demand_source=None,
                      discount_factor=1.0, initial_inventory_level=0.0,
                      trunc_tol=0.02, d_spread=4, s_spread=5,
                      oul_matrix=None, x_range=None):
    """Finite-horizon DP with the same state/action layout as the stockpyl routine."""
    T = num_periods
    if T <= 0 or int(T) != T:
        raise ValueError("num_periods must be a positive integer")
    if terminal_holding_cost < 0 or terminal_stockout_cost < 0:
        raise ValueError("terminal costs must be non-negative")

    h = _period_list(holding_cost, T, "holding_cost")
    p = _period_list(stockout_cost, T, "stockout_cost")
    c = _period_list(purchase_cost, T, "purchase_cost")
    K = _period_list(fixed_cost, T, "fixed_cost")
    gamma = _period_list(discount_factor, T, "discount_factor")
    mu = _period_list(demand_mean, T, "demand_mean")
    sigma = _period_list(demand_sd, T, "demand_sd")
    source = _period_list(demand_source, T, "demand_source")

    for t in range(1, T + 1):
        if source[t] is not None:
            dist = source[t].demand_distribution
            mu[t] = mu[t] if mu[t] is not None else dist.mean()
            sigma[t] = sigma[t] if sigma[t] is not None else dist.std()
        elif mu[t] is None or sigma[t] is None:
            raise ValueError("provide demand_mean and demand_sd, or demand_source")

    for values, name in [(h, "holding_cost"), (p, "stockout_cost"),
                         (c, "purchase_cost"), (K, "fixed_cost")]:
        if np.any(np.asarray(values[1:], dtype=float) < 0):
            raise ValueError(f"{name} must be non-negative")
    if np.any(np.asarray(gamma[1:], dtype=float) <= 0) or np.any(np.asarray(gamma[1:], dtype=float) > 1):
        raise ValueError("discount_factor must be > 0 and <= 1")

    mu_vals = np.asarray(mu[1:], dtype=float)
    sd_vals = np.asarray(sigma[1:], dtype=float)
    d_min = int(max(0, round(np.min(mu_vals) - d_spread * np.max(sd_vals))))
    d_max = int(round(np.max(mu_vals) + d_spread * np.max(sd_vals)))
    d_range = np.arange(d_min, d_max + 1)

    def cdf(t, z):
        if source[t] is None:
            return norm.cdf(z, float(mu[t]), float(sigma[t]))
        return source[t].demand_distribution.cdf(z)

    outside = np.array([cdf(t, d_min) + (1 - cdf(t, d_max)) for t in range(1, T + 1)])
    if np.any(outside > trunc_tol):
        warnings.warn("demand truncation probability exceeds trunc_tol")

    if oul_matrix is not None:
        user_provided_oul_matrix = True
        if x_range is None:
            raise ValueError("x_range is required when oul_matrix is provided")
        x_range = np.asarray(x_range, dtype=int)
        x_min, x_max = int(np.min(x_range)), int(np.max(x_range))
    elif x_range is not None:
        user_provided_oul_matrix = False
        x_range = np.asarray(x_range, dtype=int)
        x_min, x_max = int(np.min(x_range)), int(np.max(x_range))
    else:
        user_provided_oul_matrix = False
        nv = mu_vals
        Q = np.array([_eoq_with_backorders(float(K[t]), float(h[t]), float(p[t]), float(mu[t]))
                      for t in range(1, T + 1)])
        x_min = int(round(np.min(nv) - np.max(mu_vals) - np.max(sd_vals) * (s_spread + d_spread)))
        x_max = int(round(np.max(nv) + np.max(Q) + np.max(sd_vals) * s_spread))
        x_range = np.arange(x_min, x_max + 1)

    def demand_probabilities(t):
        if source[t] is not None and getattr(source[t], "is_discrete", False):
            dist = source[t].demand_distribution
            return np.array([dist.pmf(d) for d in d_range])
        return np.array([cdf(t, d + 0.5) - cdf(t, d - 0.5) for d in d_range])

    def period_loss(t, y, prob):
        if source[t] is None:
            shortage, holding = _normal_loss(y, float(mu[t]), float(sigma[t]))
            return float(h[t]) * holding + float(p[t]) * shortage
        holding = np.dot(prob, np.maximum(y - d_range, 0))
        shortage = np.dot(prob, np.maximum(d_range - y, 0))
        return float(h[t]) * holding + float(p[t]) * shortage

    done = False
    while not done:
        nx = len(x_range)
        reorder_points = [0] * (T + 1)
        order_up_to_levels = [0] * (T + 1)
        cost_matrix = np.zeros((T + 2, nx))
        if not user_provided_oul_matrix:
            oul_matrix = np.zeros((T + 1, nx))
        H = np.zeros((T + 1, nx))
        cost_matrix[T + 1, :] = terminal_holding_cost * np.maximum(x_range, 0) \
                              + terminal_stockout_cost * np.maximum(-x_range, 0)
        abort = False

        for t in range(T, 0, -1):
            prob = demand_probabilities(t)

            for y in range(x_min, x_max + 1):
                d_eff = np.maximum(np.minimum(d_range, y - x_min), y - x_max)
                future = float(gamma[t]) * np.dot(prob, cost_matrix[t + 1, y - d_eff - x_min])
                H[t, y - x_min] = period_loss(t, y, prob) + future

            for x in range(x_min, x_max + 1):
                best_cost = float("inf")
                if user_provided_oul_matrix:
                    y_values = [int(oul_matrix[t, x - x_min])]
                else:
                    y_values = range(x, x_max + 1)

                for y in y_values:
                    order = float(c[t]) * (y - x) + float(K[t]) if y > x else 0.0
                    cost = order + H[t, y - x_min]
                    if cost < best_cost:
                        best_cost, best_y = cost, y
                        if y == x_max and x < x_max and not user_provided_oul_matrix:
                            warnings.warn("cost is still decreasing at the upper end of the y range; expanding")
                            abort = True
                            x_max *= 2
                            x_range = np.arange(x_min, x_max + 1)
                            break
                if abort:
                    break
                cost_matrix[t, x - x_min] = best_cost
                oul_matrix[t, x - x_min] = best_y

            if abort:
                break

            order_up_to_levels[t] = oul_matrix[t, 0]
            reorder_points[t] = x_range[0]
            while (reorder_points[t] < x_max and
                   oul_matrix[t, reorder_points[t] + 1 - x_min] == order_up_to_levels[t]):
                reorder_points[t] += 1

            prob_below = 1 - cdf(t, reorder_points[t] - x_min)
            if prob_below > trunc_tol:
                warnings.warn("probability of falling below the x-grid exceeds trunc_tol")

        if not abort:
            done = True

    total_cost = cost_matrix[1, int(initial_inventory_level) - x_min]
    return reorder_points, order_up_to_levels, total_cost, cost_matrix, oul_matrix, x_range
```
