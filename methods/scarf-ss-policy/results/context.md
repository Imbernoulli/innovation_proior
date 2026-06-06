# Context: optimal ordering for a single product over time under a fixed ordering charge

## Research question

A firm stocks one product, reviews its inventory at the start of each period, and must decide how much to order. Demand in each period is a random variable with a known distribution; whatever is ordered arrives immediately and can serve that period's demand; unmet demand is backlogged (carried forward as negative inventory). Three costs accrue: a holding cost on every unit left at the end of a period, a shortage (penalty) cost on every unit backlogged, and an ordering cost. The ordering cost is the crux: besides a linear per-unit purchase cost $c$, every order — no matter how small — incurs a **fixed setup charge $K>0$** (paperwork, a delivery fee, a line changeover). The firm wants the ordering rule, over a finite horizon of $T$ periods, that minimizes total expected (optionally discounted) cost.

The question is *structural*, not merely numerical. A brute-force dynamic program can in principle compute the optimal order for every inventory level and every period, but that produces an opaque lookup table. What a planner — and the theory — wants is to know whether the optimal rule has a **simple, parametric form** that can be stated with a few numbers per period and computed and audited cheaply. Without the fixed cost it is known that a single "order-up-to" number per period suffices (a base-stock policy). The open question is what happens once $K>0$ breaks that, because the fixed cost makes frequent small orders wasteful and so should induce *batching* — wait until inventory is low enough, then place one larger order. The goal is to prove (or disprove) that the optimum has the two-number "reorder-point / order-up-to" shape, and to characterize the two numbers, for general convex holding/shortage costs and general demand distributions.

## Background

**The single-period root: the newsvendor.** The atom of the whole field is the one-period problem. Order up to level $S$, face random demand $w$; pay holding $h$ on leftovers $(S-w)^+$ and penalty $p$ on shortages $(w-S)^+$. Expected cost $\bar L(S)=E[h(S-w)^+ + p(w-S)^+]$ is convex in $S$, and its minimizer is the *critical fractile* $S^*$ with $F(S^*)=p/(p+h)$, where $F$ is the demand CDF. This model traces to Edgeworth (1888) in a banking-reserve setting and was formalized in the general inventory context by Arrow, Harris and Marschak (1951). It has no fixed cost and no future: it sets the natural target stock for a single shot.

**The multi-period machinery: dynamic programming.** Bellman (1957) supplied the tool for chaining periods together. With inventory level $x$ as the state, the principle of optimality gives a backward functional equation: the optimal cost-to-go $J_k(x)$ from period $k$ equals the best immediate decision plus the expected optimal cost-to-go from the next state. The post-order inventory $y=x+u$ is the natural decision variable, and the per-period transition is $x_{k+1}=y_k-w_k$. This turns a $T$-period optimization into the repeated application of a single one-step operator, and — crucially — into a vehicle for *proving structural properties of the value function by backward induction*: if one can show the value function carries some property $P$ at period $k+1$, that the one-step operator preserves $P$, and that $P$ forces the optimal decision into a particular shape, then induction delivers the policy structure for the whole horizon.

**Why no-fixed-cost is easy.** When $K=0$ the chain closes cleanly. Define $G_k(y)=cy+\bar L(y)+\gamma\, E_w[J_{k+1}(y-w)]$, the total expected cost of bringing stock to $y$ now and acting optimally afterward, where $\bar L(y)=E_w[L(y-w)]$ and $L(x)=p\,x^- + h\,x^+$. If $J_{k+1}$ is convex, then $\bar L$ is convex (an expectation of a convex function), $E_w[J_{k+1}(y-w)]$ is convex, $cy$ is linear, so $G_k$ is convex; and with $c+h>0$ and $p>c$ it is coercive ($G_k\to\infty$ as $|y|\to\infty$). A convex coercive function has a single unconstrained minimizer $S_k$, and minimizing $G_k(y)$ over the constraint $y\ge x$ is trivial: go to $S_k$ if $x<S_k$, else stay. Plugging back shows $J_k(x)=G_k(\max(x,S_k))-cx$ is again convex, so the induction sustains itself. The optimal rule is a **base-stock policy**: one target level $S_k$ per period, order up to it whenever below it.

**The diagnostic difficulty the fixed cost creates.** Once $K>0$, the cost-to-go develops a *downward jump*. At the inventory level where "don't order" stops beating "order up to $S_k$ and pay $K$", the cost-to-go drops discontinuously — the moment ordering becomes worthwhile, the realized gain over not-ordering already exceeds the setup charge. A function with a downward jump is not convex, so the convexity-preserved-by-the-operator argument collapses at the first backward step. This is the load-bearing obstacle: the standard tool that proves base-stock optimality literally stops applying, and there is no a-priori reason the optimum should remain two-parameter. The natural worry is that for some awkward demand distribution the optimal order could depend on $x$ in a complicated, non-monotone way.

**What was already known about the answer's shape.** Arrow–Harris–Marschak had already identified the two-number "best maximum stock and best reorder point" form as the natural object of interest. Dvoretzky, Kiefer and Wolfowitz, in "The inventory problem" (Econometrica 1952) and "On the optimal character of the $(s,S)$ policy in inventory theory" (Econometrica 1953), established the optimality of the $(s,S)$ form, but through restrictive assumptions and intricate case analyses rather than a single clean structural property of the value function that the dynamic-programming recursion could be shown to preserve. So the *shape* of the answer was strongly suspected; what was missing was a property — weaker than convexity, but still strong enough to force the $(s,S)$ shape — that the min-plus-expectation operator provably preserves under general convex costs and general demand.

## Baselines

**Newsvendor / critical-fractile (Edgeworth 1888; Arrow–Harris–Marschak 1951).** One period, convex holding+shortage cost, no setup charge; optimal order-up-to $S^*$ at the critical fractile $F(S^*)=p/(p+h)$. The core idea and exact target the multi-period theory must reduce to. Gap: single period, no fixed cost, no dynamics — says nothing about when to order across time.

**Base-stock dynamic program (the $K=0$ convexity argument, on Bellman 1957).** Backward DP with $G_k$ convex coercive $\Rightarrow$ unique minimizer $S_k$ $\Rightarrow$ order-up-to-$S_k$ optimal, value function stays convex, induction closes. Clean and complete — but only for $K=0$. Gap: with a positive setup charge the cost-to-go picks up a downward jump and is no longer convex, so this entire proof breaks at the first induction step; it cannot even be started, let alone yield $(s,S)$.

**Arrow–Harris–Marschak (1951), dynamic uncertainty model.** Posed the dynamic stochastic inventory problem and named the "best maximum stock $S$ and best reorder point $s$" as the quantities to determine as functions of the demand distribution and the order/penalty costs. Gap: identifies the form and the comparative statics but does not give a general optimality proof for the two-parameter rule under a fixed cost.

**Dvoretzky–Kiefer–Wolfowitz (1952, 1953).** Proved optimality of the $(s,S)$ policy in inventory theory — pioneering and correct, establishing that the two-number form is genuinely optimal in their setting. Gap: relies on restrictive conditions and case-by-case arguments specific to their assumptions; it does not isolate a single structural invariant of the cost-to-go that the DP operator preserves, so it does not extend transparently to general convex costs and arbitrary demand distributions, nor generalize to the array of variant models a reusable structural property would unlock.

## Evaluation settings

The natural test instance is a finite-horizon, periodic-review, single-product system specified by: number of periods $T$; per-period holding cost $h$ and shortage/penalty cost $p$ (possibly time-varying); per-unit purchase cost $c$; fixed ordering cost $K\ge0$; a demand distribution per period (e.g. normal with mean $\mu_t$ and standard deviation $\sigma_t$, or an arbitrary discrete/continuous law); a discount factor $\gamma\in(0,1]$; an initial inventory level $x_1$; and a terminal cost function on the leftover inventory at the horizon's end (typically $h_T x^+ + p_T x^-$). The yardstick for any proposed structural result is the exact optimum produced by a full backward dynamic program over a discretized inventory grid: the structural claim is that this exact optimum coincides, in every period, with a two-number reorder-point / order-up-to rule, collapsing to a single number when $K=0$. The relevant outputs are the per-period parameters and the total expected discounted cost from $x_1$.

## Code framework

The pre-existing primitives are: a per-period convex holding/shortage cost (a newsvendor-style expected-cost evaluation $\bar L(y)=E_w[L(y-w)]$ over a demand distribution), a way to take expectations over a (discretized) demand distribution, and the generic backward dynamic-programming loop over a discretized state grid. What does not yet exist is the structural characterization of the optimizer of the per-period problem and the extraction of the policy parameters from it.

```python
import numpy as np

def expected_period_cost(y, demand_pmf, d_values, h, p):
    """Newsvendor expected holding+shortage cost L_bar(y) = E[h (y-w)^+ + p (w-y)^+]."""
    hold = np.sum(demand_pmf * np.maximum(y - d_values, 0.0))
    short = np.sum(demand_pmf * np.maximum(d_values - y, 0.0))
    return h * hold + p * short

def order_cost(u, K, c):
    """Linear cost plus a fixed charge incurred only when something is ordered."""
    return (K + c * u) if u > 0 else 0.0

def H(y, t, theta_next, demand_pmf, d_values, x_min, num_x, h, p, gamma):
    """Everything about period t that depends on the post-order level y but not on x:
    H_t(y) = L_bar(y) + gamma * E_w[ theta_{t+1}(y - w) ]. Precomputed once per (t,y)."""
    Lbar = expected_period_cost(y, demand_pmf, d_values, h, p)
    idx = np.clip((y - d_values - x_min).astype(int), 0, num_x - 1)
    future = gamma * np.sum(demand_pmf * theta_next[idx])
    return Lbar + future

def solve_period(t, theta_next, x_grid, x_min, demand_pmf, d_values, h, p, c, K, gamma):
    """For each starting inventory x, choose y >= x minimizing
       K*delta(y>x) + c*(y-x) + H_t(y).
    Return theta_t over the grid and the optimal post-order level for each x."""
    theta_t = np.zeros(len(x_grid))
    best_y = np.zeros(len(x_grid))
    # TODO: the structural result we are about to derive determines the SHAPE of the
    #       minimizer here (how the optimal y depends on x) and how to read off the
    #       per-period policy parameters from it.
    raise NotImplementedError

def extract_policy_parameters(best_y, x_grid):
    """Read the per-period policy parameters off the optimal-action profile.
    The form of these parameters is exactly what the structural result establishes."""
    # TODO
    raise NotImplementedError

def finite_horizon_dp(T, h, p, c, K, demand, gamma=1.0, terminal_cost=None):
    """Backward DP over periods T, T-1, ..., 1 on a discretized inventory grid.
    Returns the per-period policy parameters and the total expected discounted cost."""
    # build x_grid, demand pmf/d_values, terminal theta_{T+1}
    # for t = T down to 1: theta_t, best_y = solve_period(...); record parameters
    # TODO
    raise NotImplementedError
```
