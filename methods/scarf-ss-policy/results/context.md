# Context: optimal ordering for a single product over time under a fixed ordering charge

## Research question

A firm stocks one product, reviews its inventory at the start of each period, and must decide how much to order. Demand in each period is a random variable with a known distribution; whatever is ordered arrives immediately and can serve that period's demand; unmet demand is backlogged (carried forward as negative inventory). Three costs accrue: a holding cost on every unit left at the end of a period, a shortage (penalty) cost on every unit backlogged, and an ordering cost. The ordering cost includes a linear per-unit purchase cost $c$ and a **fixed setup charge $K>0$** (paperwork, a delivery fee, a line changeover) incurred on every order regardless of size. The firm wants the ordering rule, over a finite horizon of $T$ periods, that minimizes total expected (optionally discounted) cost.

The question is *structural*: a brute-force dynamic program can in principle compute the optimal order for every inventory level and every period, but the goal is to determine whether the optimal rule has a **simple, parametric form** that can be stated with a few numbers per period and computed and audited cheaply.

## Background

**The single-period root: the newsvendor.** The atom of the whole field is the one-period problem. Order up to level $S$, face random demand $w$; pay holding $h$ on leftovers $(S-w)^+$ and penalty $p$ on shortages $(w-S)^+$. Expected cost $\bar L(S)=E[h(S-w)^+ + p(w-S)^+]$ is convex in $S$, and a minimizer is a critical-fractile quantile: in the continuous case $F(S^*)=p/(p+h)$, while with atoms any $S^*$ satisfying $F(S^{*-})\le p/(p+h)\le F(S^*)$ is optimal. This model traces to Edgeworth (1888) in a banking-reserve setting and was formalized in the general inventory context by Arrow, Harris and Marschak (1951). It has no fixed cost and no future.

**The multi-period machinery: dynamic programming.** Bellman (1957) supplied the tool for chaining periods together. With inventory level $x$ as the state, the principle of optimality gives a backward functional equation: the optimal cost-to-go $J_k(x)$ from period $k$ equals the best immediate decision plus the expected optimal cost-to-go from the next state. The post-order inventory $y=x+u$ is the natural decision variable, and the per-period transition is $x_{k+1}=y_k-w_k$. This turns a $T$-period optimization into the repeated application of a single one-step operator, and into a vehicle for *proving structural properties of the value function by backward induction*: the value function carries a property $P$ at period $k+1$, the one-step operator preserves $P$, and $P$ forces the optimal decision into a particular shape; induction then delivers the policy structure for the whole horizon.

**The no-fixed-cost case.** When $K=0$, define $G_k(y)=cy+\bar L(y)+\gamma\, E_w[J_{k+1}(y-w)]$, the total expected cost of bringing stock to $y$ now and acting optimally afterward, where $\bar L(y)=E_w[L(y-w)]$ and $L(x)=p\,x^- + h\,x^+$. If $J_{k+1}$ is convex, then $\bar L$ is convex (an expectation of a convex function), $E_w[J_{k+1}(y-w)]$ is convex, $cy$ is linear, so $G_k$ is convex; and with $c+h>0$ and $p>c$ it is coercive ($G_k\to\infty$ as $|y|\to\infty$). A convex coercive function has a single unconstrained minimizer $S_k$, and minimizing $G_k(y)$ over the constraint $y\ge x$ yields: go to $S_k$ if $x<S_k$, else stay. Plugging back shows $J_k(x)=G_k(\max(x,S_k))-cx$ is again convex, so the induction sustains itself. The optimal rule is a **base-stock policy**: one target level $S_k$ per period, order up to it whenever below it.

**The fixed-cost setting.** Once $K>0$, the cost-to-go can develop a downward dent and may develop downward jumps. Around the inventory level where "don't order" and "order up to $S_k$ and pay $K$" are equally costly, the value function can go flat before heading downward toward the order-up-to minimum. The convexity argument that proves base-stock optimality no longer applies under a fixed charge: the one-step operator does not preserve convexity of the value function when $K>0$.

**Prior structural results.** Arrow–Harris–Marschak (1951) posed the dynamic stochastic inventory problem and identified the "best maximum stock $S$ and best reorder point $s$" as the natural quantities to determine as functions of the demand distribution and costs. Dvoretzky, Kiefer and Wolfowitz, in "The inventory problem" (Econometrica 1952) and "On the optimal character of the $(s,S)$ policy in inventory theory" (Econometrica 1953), proved optimality of the two-number form under their assumptions.

## Baselines

**Newsvendor / critical-fractile (Edgeworth 1888; Arrow–Harris–Marschak 1951).** One period, convex holding+shortage cost, no setup charge; optimal order-up-to $S^*$ at the critical fractile, with $F(S^*)=p/(p+h)$ in the continuous case and $F(S^{*-})\le p/(p+h)\le F(S^*)$ when demand has atoms. Sets the natural target stock for a single shot.

**Base-stock dynamic program (the $K=0$ convexity argument, on Bellman 1957).** Backward DP with $G_k$ convex coercive $\Rightarrow$ unique minimizer $S_k$ $\Rightarrow$ order-up-to-$S_k$ optimal, value function stays convex, induction closes. Optimal for $K=0$.

**Arrow–Harris–Marschak (1951), dynamic uncertainty model.** Posed the dynamic stochastic inventory problem and named the "best maximum stock $S$ and best reorder point $s$" as the quantities to determine as functions of the demand distribution and the order/penalty costs.

**Dvoretzky–Kiefer–Wolfowitz (1952, 1953).** Proved optimality of the two-number ordering policy in inventory theory, establishing the parametric form as genuinely optimal under their assumptions.

## Evaluation settings

The natural test instance is a finite-horizon, periodic-review, single-product system specified by: number of periods $T$; per-period holding cost $h$ and shortage/penalty cost $p$ (possibly time-varying); per-unit purchase cost $c$; fixed ordering cost $K\ge0$; a demand distribution per period (e.g. normal with mean $\mu_t$ and standard deviation $\sigma_t$, or an arbitrary discrete/continuous law); a discount factor $\gamma\in(0,1]$; an initial inventory level $x_1$; and a terminal cost function on the leftover inventory at the horizon's end (typically $h_T x^+ + p_T x^-$). The yardstick for any proposed structural result is the exact optimum produced by a full backward dynamic program over a discretized inventory grid: the structural claim is that this exact optimum coincides, in every period, with a two-number reorder-point / order-up-to rule, collapsing to a single number when $K=0$. The relevant outputs are the per-period parameters and the total expected discounted cost from $x_1$.

## Code framework

The pre-existing primitives are: a per-period convex holding/shortage cost (a newsvendor-style expected-cost evaluation $\bar L(y)=E_w[L(y-w)]$ over a demand distribution), a way to take expectations over a discretized demand distribution, a state-grid truncation rule, and the generic backward dynamic-programming loop. What does not yet exist is the structural characterization of the optimizer of the per-period problem and the compact summary of the optimal-action profile.

```python
import numpy as np

def finite_horizon_dp(num_periods, holding_cost, stockout_cost,
                      terminal_holding_cost, terminal_stockout_cost,
                      purchase_cost, fixed_cost,
                      demand_mean=None, demand_sd=None, demand_source=None,
                      discount_factor=1.0, initial_inventory_level=0.0,
                      trunc_tol=0.02, d_spread=4, s_spread=5,
                      action_matrix=None, x_range=None):
    """Backward dynamic program on an integer inventory grid."""
    # validate and broadcast scalar inputs to period-indexed arrays
    # build a demand distribution for each period
    # choose a demand grid and an inventory grid
    # initialize theta_{T+1}(x) = terminal_holding_cost*x^+ + terminal_stockout_cost*x^-

    for t in range(num_periods, 0, -1):
        # compute the demand probabilities used in the expectation
        # precompute H_t(y) = L_bar_t(y) + discount_t * E[theta_{t+1}(y-D)]
        # for each starting inventory x, scan feasible post-order y >= x:
        #     cost = fixed/order cost, if any, plus H_t(y)
        #     store the minimum cost and the minimizing post-order y
        # TODO: characterize the optimal-action profile and reduce it to a compact
        #       set of per-period policy numbers.
        pass

    # return the compact policy summary, the starting cost, the full cost matrix,
    # the action matrix, and the grid used for the discretization
    raise NotImplementedError
```
