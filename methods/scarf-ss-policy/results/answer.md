# Optimality of $(s,S)$ policies via $K$-convexity

## Problem

A single product, periodic review, finite horizon of $T$ periods. At the start of period $t$ the inventory level is $x$; order $u\ge0$ (delivered immediately), reaching post-order level $y=x+u$; then random demand $w_t$ (known distribution) is realized and the inventory becomes $y-w_t$ (backlogging allowed, so it can go negative). Costs per period: a fixed setup charge $K$ whenever $u>0$, a linear purchase cost $c$ per unit, and an end-of-period convex holding/shortage cost $L(x)=h\,x^+ + p\,x^-$. A discount factor $\gamma\in(0,1]$ applies, and a convex nonnegative terminal cost $v_T$ on the final inventory. Minimize total expected discounted cost. Standing assumptions: $c+h>0$ and $p>c$ (otherwise never ordering would be optimal).

## Bellman recursion

With inventory $x$ the state and post-order level $y\ge x$ the decision,
$$
J_T^*(x)=v_T(x),\qquad
J_t^*(x)=\min_{y\ge x}\Big\{K\,\delta(y-x)+c(y-x)+\bar L(y)+\gamma\,E_w[J_{t+1}^*(y-w)]\Big\},
$$
where $\delta(z)=1$ if $z>0$ else $0$, and $\bar L(y)=E_w[L(y-w)]$ is the newsvendor expected cost ($\bar L$ minimized at the critical fractile $F(y)=p/(p+h)$). Pulling out the linear term, define
$$
G_t(y)=cy+\bar L(y)+\gamma\,E_w[J_{t+1}^*(y-w)],\qquad
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

**Closure (preservation).** (i) If $f_1$ is $K_1$-convex and $f_2$ is $K_2$-convex, then $\alpha f_1+\beta f_2$ ($\alpha,\beta>0$) is $(\alpha K_1+\beta K_2)$-convex. (ii) If $f$ is $K$-convex then $y\mapsto E_w[f(y-w)]$ is $K$-convex (translation preserves $K$-convexity; expectation is linear). Hence if $J_{t+1}^*$ is $K$-convex, $G_t=cy+\bar L+\gamma E_w[J_{t+1}^*(\cdot-w)]$ is $\gamma K$-convex, therefore $K$-convex (since $\gamma\le1$), and it is coercive and continuous.

**$(s,S)$ from $K$-convexity.** If $f$ is continuous, coercive, and $K$-convex, let $S=\arg\min f$ and let $s$ be the smallest $z\le S$ with $f(z)=f(S)+K$. Then: (1) $S$ minimizes $f$; (2) $f(s)=f(S)+K$ and $f(y)>f(s)$ for $y<s$; (3) $f$ is decreasing on $(-\infty,s)$; (4) $f(y)\le f(y')+K$ for $s\le y\le y'$. Applied to $G_t$: if $x<s_t$ then $G_t(x)>K+G_t(S_t)$, so ordering up to $S_t$ strictly wins; if $x\ge s_t$ then $G_t(x)\le K+G_t(y)$ for every $y>x$, so not ordering wins. The optimal policy is **$(s_t,S_t)$**: order up to $S_t$ iff $x<s_t$.

**Closing the induction.** Plugging the $(s_t,S_t)$ policy back, $\tilde G_t(x):=J_t^*(x)+cx$ equals the constant $K+G_t(S_t)=G_t(s_t)$ for $x\le s_t$ and equals $G_t(x)$ for $x\ge s_t$. A three-case check against the definition (both points $\ge s_t$: it is $G_t$; both $\le s_t$: constant; $y<s_t<y'$: the flat part sits under the increasing $K$-lifted chord, and the right part is under it by $K$-convexity of $G_t$) shows $\tilde G_t$, hence $J_t^*$, is $K$-convex, nonnegative, continuous. With base case $J_T^*=v_T$ ($0$-convex), induction gives an optimal $(s_t,S_t)$ policy at every stage. When $K=0$, $s_t=S_t$ and the policy collapses to base-stock.

**Reading off the parameters:** $S_t=\arg\min_y G_t(y)$; $s_t=$ smallest $z\le S_t$ with $G_t(z)=G_t(S_t)+K$.

## Code

Backward DP on a discretized inventory grid. The expensive part $H_t(y)=\bar L(y)+\gamma E_w[\theta_{t+1}(y-w)]$ is precomputed once per $(t,y)$ and reused across all $x$ in the inner minimization. $S_t$ is the order-up-to level at the lowest $x$; $s_t$ is the largest $x$ that still orders up to $S_t$.

```python
import numpy as np
from scipy.stats import norm

def finite_horizon_dp(num_periods, holding_cost, stockout_cost,
                      terminal_holding_cost, terminal_stockout_cost,
                      purchase_cost, fixed_cost,
                      demand_mean, demand_sd,
                      discount_factor=1.0, initial_inventory_level=0.0,
                      d_spread=4, s_spread=5):
    """theta_t(x) = min_{y>=x} { K*1[y>x] + c*(y-x) + Lbar(y) + gamma*E[theta_{t+1}(y-D)] }.
    K-convexity of theta_t makes the minimizer an (s,S) rule; read s_t, S_t off it.
    Returns (reorder_points[t]=s_t, order_up_to_levels[t]=S_t, total_cost)."""
    T = num_periods
    def lst(v):
        return [None] + [v]*T if np.isscalar(v) else v
    h, p, c, K = lst(holding_cost), lst(stockout_cost), lst(purchase_cost), lst(fixed_cost)
    mu, sd, g = lst(demand_mean), lst(demand_sd), lst(discount_factor)

    # Demand support: mu +/- d_spread*sigma, clipped at 0.
    d_min = int(max(0, round(min(mu[1:]) - d_spread*max(sd[1:]))))
    d_max = int(round(max(mu[1:]) + d_spread*max(sd[1:])))
    d_range = np.arange(d_min, d_max+1)

    # Inventory grid: s ~ newsvendor, S ~ s + EOQ-with-backorders batch size.
    nv = [None] + [mu[t] for t in range(1, T+1)]
    Q  = [None] + [np.sqrt(2*K[t]*mu[t]/max(h[t], 1e-9)) for t in range(1, T+1)]
    x_min = int(round(min(nv[1:]) - max(mu[1:]) - max(sd[1:])*(s_spread+d_spread)))
    x_max = int(round(max(nv[1:]) + max(Q[1:]) + max(sd[1:])*s_spread))
    x_range = np.arange(x_min, x_max+1)
    nx = len(x_range)

    reorder_points = [0]*(T+1)        # s_t
    order_up_to_levels = [0]*(T+1)    # S_t
    theta = np.zeros((T+2, nx))       # cost-to-go

    # Terminal cost theta_{T+1}(x) = h_T x^+ + p_T x^-.
    theta[T+1, :] = terminal_holding_cost*np.maximum(x_range, 0) \
                  + terminal_stockout_cost*np.maximum(-x_range, 0)

    for t in range(T, 0, -1):
        prob = norm.cdf(d_range+0.5, mu[t], sd[t]) - norm.cdf(d_range-0.5, mu[t], sd[t])

        # Precompute H_t(y) = Lbar(y) + gamma * E[theta_{t+1}(y - D)].
        H = np.zeros(nx)
        for j, y in enumerate(x_range):
            hold  = np.dot(prob, np.maximum(y - d_range, 0))
            short = np.dot(prob, np.maximum(d_range - y, 0))
            Lbar = h[t]*hold + p[t]*short
            d_eff = np.clip(d_range, y - x_max, y - x_min)   # keep y-d on the grid
            future = g[t]*np.dot(prob, theta[t+1, (y - d_eff) - x_min])
            H[j] = Lbar + future

        # Minimize over y >= x with the fixed-cost toll.
        oul = np.zeros(nx)
        for i, x in enumerate(x_range):
            best_cost, best_y = np.inf, x
            for j in range(i, nx):
                y = x_range[j]
                order = (K[t] + c[t]*(y - x)) if y > x else 0.0
                cost = order + H[j]
                if cost < best_cost:
                    best_cost, best_y = cost, y
            theta[t, i] = best_cost
            oul[i] = best_y

        # S_t = OUL at lowest x; s_t = largest x still ordering up to S_t.
        S_t = oul[0]
        order_up_to_levels[t] = S_t
        s_idx = 0
        while s_idx+1 < nx and oul[s_idx+1] == S_t:
            s_idx += 1
        reorder_points[t] = x_range[s_idx]   # == S_t when K=0

    total_cost = theta[1, int(round(initial_inventory_level)) - x_min]
    return reorder_points, order_up_to_levels, total_cost
```
