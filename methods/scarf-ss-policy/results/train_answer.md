I run a single product with periodic review over a finite horizon of $T$ periods. At the start of period $t$ I see inventory $x$, order $u\ge0$ (it arrives immediately), reaching post-order level $y=x+u$; then a random demand $w_t$ with known distribution hits and inventory becomes $y-w_t$, where unmet demand is backlogged so the level can go negative. Each period I pay a linear purchase cost $c$ per unit, an end-of-period convex holding/shortage cost $L(x)=h\,x^+ + p\,x^-$, and — the difficulty — a fixed setup charge $K$ that is incurred whenever I order at all, regardless of order size. A truck rolls, paperwork is filed, a line is changed over, and ordering one unit or five hundred eats the same $K$. I want the policy that minimizes total expected discounted cost, with discount $\gamma\in(0,1]$ and a convex nonnegative terminal cost $v_{T+1}$ on the final inventory; throughout I assume $c+h>0$ and $p>c$ (otherwise never ordering is optimal). And I do not want an opaque lookup table — I want to know whether the optimal rule has a compact, parametric shape.

The single-period atom is the newsvendor: order up to $S$, pay $\bar L(S)=E[h(S-w)^+ + p(w-S)^+]$, an expectation of a convex function, minimized at the critical fractile $F(S^*)=p/(p+h)$ in the continuous case (with atoms, any $S^*$ with $F(S^{*-})\le p/(p+h)\le F(S^*)$). Chaining periods with Bellman's recursion, with inventory $x$ the state and post-order level $y\ge x$ the decision, the optimal cost-to-go is $J_t^*(x)=\min_{y\ge x}\{K\,\delta(y-x)+c(y-x)+\bar L_t(y)+\gamma_t E_w[J_{t+1}^*(y-w)]\}$, where $\delta(z)=1$ if $z>0$ and $0$ otherwise. When $K=0$ this closes beautifully: defining $G_t(y)=cy+\bar L_t(y)+\gamma_t E_w[J_{t+1}^*(y-w)]$, if $J_{t+1}^*$ is convex then $G_t$ is convex and coercive, hence has a unique minimizer $S_t$, the rule is base-stock (order up to $S_t$ iff $x<S_t$), and the resulting $J_t^*$ is again convex — the whole induction is powered by the single fact that the operator preserves convexity. The prior structural work suspected the right shape with a fixed cost is a two-number reorder-point / order-up-to band, but the route to it was the problem: Arrow–Harris–Marschak named the form, and Dvoretzky–Kiefer–Wolfowitz proved $(s,S)$ optimality only under restrictive assumptions and intricate case analysis that does not transfer to general convex costs and arbitrary demand. The deeper trouble is that the moment $K>0$, the cost-to-go grows a downward dent — it can go flat at the level "$K$ above the order-up-to minimum" and then slope downward into that minimum, or even pick up a downward jump after the dent feeds through later recursions. A convex function can do neither, so the convexity-preserved-by-the-operator engine that made $K=0$ trivial dies at the very first backward step, and one cannot even start the induction.

I propose to characterize the optimal policy through a relaxed invariant I call **$K$-convexity**, which is exactly the dented version of convexity that survives the Bellman operator and still pins the policy to two numbers, yielding the optimal **$(s,S)$ policy**. Pulling out the linear term so $J_t^*(x)=\min_{y\ge x}\{K\,\delta(y-x)+G_t(y)\}-cx$, everything reduces to the shape of $G_t$: minimize it over $y\ge x$ with a $K$ toll the instant you move off $y=x$. The defining relaxation is to lift the right end of the convexity chord by $K$. A function $f:\mathbb R\to\mathbb R$ is $K$-convex ($K\ge0$) if for all $y\le y'$ and $\theta\in[0,1]$,
$$f(\theta y+(1-\theta)y')\le \theta f(y)+(1-\theta)\big(K+f(y')\big),$$
so on $[y,y']$ the graph must lie below the segment from $(y,f(y))$ to $(y',f(y')+K)$ — the right endpoint raised by $K$, the left endpoint not. With $K=0$ this is ordinary convexity; with $K>0$ a $K$-convex function may be non-convex, may have several local minima, and may have downward jumps bounded by $K$. This is precisely the slack the dented cost-to-go needs: the flat top sits exactly $K$ above the minimum, and the dip below the flat is exactly $K$, so it still sits under any $K$-lifted chord. Two equivalent forms are easier to compute with: writing collinear points $y-b$, $y$, $y+a$ ($b>0$, $a\ge0$) and substituting the convex weights $\theta=a/(a+b)$ gives
$$K+f(y+a)\ \ge\ f(y)+\tfrac{a}{b}\big[f(y)-f(y-b)\big],\qquad \forall\,y,\ a\ge0,\ b>0,$$
which reads "$f(y+a)$ cannot fall below the forward extrapolation of the backward slope at $y$ by more than $K$," and sending $b\to0$ for differentiable $f$ gives the tangent form $K+f(y)\ge f(x)+f'(x)(y-x)$ for $x\le y$, "the graph lies above its tangents, minus $K$."

What makes $K$-convexity the right invariant — and why it beats simply hoping the policy stays base-stock with a harder proof — is that it satisfies the two demands convexity used to satisfy: it is preserved by the operator, and it forces the $(s,S)$ shape. Preservation rests on two closure facts. First, positive combinations: adding the defining inequalities shows that if $f_1$ is $K_1$-convex and $f_2$ is $K_2$-convex then $\alpha f_1+\beta f_2$ ($\alpha,\beta>0$) is $(\alpha K_1+\beta K_2)$-convex. Second, the demand-shift expectation: for fixed $w$, $y\mapsto f(y-w)$ is a horizontal translation and translation leaves $K$-convexity untouched with the same $K$, and since the inequality is linear in $f$, taking $E_w$ of both sides preserves it — so $y\mapsto E_w[f(y-w)]$ is $K$-convex. Hence if $J_{t+1}^*$ is $K$-convex, then $\gamma_t E_w[J_{t+1}^*(\cdot-w)]$ is $\gamma_t K$-convex, and adding the $0$-convex $cy$ and the $0$-convex $\bar L_t$ keeps it $\gamma_t K$-convex; since $\gamma_t\le1$ and lifting the chord by more only makes the inequality easier, $G_t$ is $K$-convex, and it is coercive ($c+h>0$, $p>c$) and continuous.

Now extract the band from "$f$ continuous, coercive, $K$-convex." Let $S$ be a global minimizer and $s$ the smallest $z\le S$ with $f(z)=f(S)+K$ — such $z$ exists because $f(S)<f(S)+K$ while coercivity sends $f\to\infty$ leftward, so continuity forces a crossing. The load-bearing geometric consequences are exact. If $u<v$ and $f(u)=K+f(v)$, then for any $z=\theta u+(1-\theta)v\in[u,v]$, $K$-convexity gives $f(z)\le\theta f(u)+(1-\theta)(K+f(v))=K+f(v)$, so the whole interval stays below the lifted level. With $u=s$, $v=S$ this gives $f(y)>f(s)$ for $y<s$ (an earlier crossing would contradict the leftmost choice of $s$), and on $(-\infty,s)$ the function strictly decreases: for $y<y'<s$, $K$-convexity on $[y,S]$ with $\theta=(S-y')/(S-y)$ yields $f(y')\le\theta f(y)+(1-\theta)(K+f(S))$, and since $K+f(S)=f(s)<f(y')$, substituting the stricter bound and cancelling the common term gives $f(y')<f(y)$. The no-order region needs the other direction: for $s\le y\le y'$, writing $y=\theta s+(1-\theta)y'$ gives $f(y)\le\theta f(s)+(1-\theta)(K+f(y'))=K+\theta f(S)+(1-\theta)f(y')\le K+f(y')$ because $S$ is a global minimizer. Translated to the decision: if $x<s$ then $G_t(x)>K+G_t(S)$, so ordering up to $S$ strictly wins; if $x\ge s$ then $G_t(x)\le K+G_t(y)$ for every $y>x$, so not ordering wins (with a tie at the boundary broken by not ordering). That is the $(s_t,S_t)$ policy, read straight off $G_t$: $S_t=\arg\min G_t$, and $s_t$ is the leftmost $z\le S_t$ with $G_t(z)=G_t(S_t)+K$. The gap $S_t-s_t$ is the batch the fixed charge forces — I let inventory drift down without ordering, then place one lump order — and when $K=0$ the crossing collapses onto the minimizer, $s_t=S_t$, and base-stock falls out as the special case.

Finally I close the induction by showing the output is again $K$-convex. Plugging the optimal action back and stripping the linear part, $\tilde G_t(x):=J_t^*(x)+cx$ equals the constant $K+G_t(S_t)=G_t(s_t)$ for $x\le s_t$ and equals $G_t(x)$ for $x\ge s_t$, continuous at $s_t$ by the definition of $s_t$ and nonnegative as a cost-to-go of nonnegative costs. Checking the definition in three cases settles it: both points $\ge s_t$ it is just $G_t$ (already $K$-convex); both $\le s_t$ it is constant (hence $K$-convex); and for $y<s_t<y'$, the flat left part sits under the nondecreasing $K$-lifted chord (whose right height $K+G_t(y')\ge K+G_t(S_t)$ equals its left height), while the right part lies under that chord because the $K$-lifted chord of $G_t$ on $[s_t,y']$ shares the right endpoint and starts no higher at $s_t$. So $\tilde G_t$, hence $J_t^*$, is $K$-convex, nonnegative, and continuous — exactly the induction hypothesis for the previous stage. With base case $J_{T+1}^*=v_{T+1}$ ($0$-convex), the induction runs from $T$ down to $1$, an $(s_t,S_t)$ policy is optimal at every stage, and the fixed cost that broke ordinary convexity is absorbed exactly by the $K$ of give.

Landing this on a discretized backward DP, the one efficiency move is that in $\theta_t(x)=\min_{y\ge x}\{K\delta(y-x)+c(y-x)+\bar L_t(y)+\gamma_t E[\theta_{t+1}(y-w)]\}$ the part $H_t(y)=\bar L_t(y)+\gamma_t E[\theta_{t+1}(y-w)]$ does not depend on $x$ — that expectation over demand is where the cost lives — so it is precomputed once per $(t,y)$ and reused across all $x$. Then for each $x$ I scan $y\ge x$; $S_t$ is the order-up-to level at the lowest $x$ in range, and the reported grid reorder point $s_t$ is the largest $x$ whose selected post-order level remains $S_t$.

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
