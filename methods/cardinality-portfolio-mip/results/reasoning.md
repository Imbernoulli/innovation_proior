Let me start from the thing that's actually bugging me, which is that I have a perfectly good continuous mean-variance optimizer and yet the answer it hands me is not something I can put on a trading desk. The continuous rule is settled: minimize `x' Sigma x - q mu' x` subject to `sum_i x_i = 1` and `l_i <= x_i <= u_i`, sweep the risk tolerance `q`, and I trace the efficient frontier. Convex QP, global optimum, done. So what's wrong? I solve it, I look at the weight vector, and it's a smear — thirty, forty, sixty tiny positions, fractions of a percent each, spread across the whole universe. Mathematically optimal. Operationally useless.

Why useless, concretely. A real fund holds a *bounded* number of names — at most `K`, where `K` might be ten or twenty out of an `N` of five hundred. Sometimes that's a prospectus rule for a focused fund; sometimes it's that monitoring and oversight cost real money per name; sometimes it's that I don't want to look like a closet indexer holding the whole market in miniature; sometimes it's a small separate account where the per-name ticket charge swamps any benefit from a 0.1% position. Whatever the reason, "hold at most `K` names" is a hard constraint, and the continuous optimizer has no idea it exists. Worse, even the names it *does* want, it wants in absurd sizes — a 0.02% weight that costs a fixed ticket to establish and contributes nothing. So if I'm going to hold a name at all, I want it to be at least some floor `l_i`. And finally, when I actually trade, I buy whole shares — or whole lots of a hundred — not 0.0734 of a position. The deployable object is an integer count of shares, not a real number.

So three things stand between the frontier and the desk: a cap on *how many* names, a floor on *how big* each held name is, and *integrality* of the actual shares. Let me see whether the convex machinery can swallow any of them.

Take the floor first, because it's the cleanest to state. I want: either I don't hold name `i` at all, so `x_i = 0`, or I hold it properly, so `x_i >= l_i`. Write that out as a feasible set for a single coordinate: `x_i in {0} union [l_i, u_i]`. Stare at it. That's a *gap* in the middle — everything between `0` and `l_i` is forbidden. The set is not an interval; it's two disconnected pieces. And a convex set can't have a hole like that: if `0` is feasible and `l_i` is feasible, convexity would force every point on the segment between them, i.e. all of `[0, l_i]`, to be feasible too — which is exactly what I'm trying to forbid. So the floor constraint is genuinely non-convex. No reformulation in `x` alone, keeping the feasible set convex, can express it.

The cardinality cap is the same disease, one level up. "At most `K` of the `x_i` are nonzero" — let me think about what feasible set that carves out. For each subset `S` of names with `|S| <= K`, the portfolios supported on `S` form a face of the simplex (the other coordinates pinned to zero). The full feasible set is the *union* of all those faces, one per allowed subset. A union of convex pieces, and the pieces overlap only on their shared boundaries — definitely not convex. Pick a portfolio holding names `{1,2,3}` and one holding names `{4,5,6}`, both feasible at `K=3`; their midpoint holds all six, violating `K=3`. So averaging two feasible points leaves the feasible set. Non-convex again, and for the same structural reason: it's a disjunction. "This subset OR that subset." Disjunctions and convexity don't mix.

So my convex QP solver, which assumes one convex region and walks to its optimum, is the wrong tool for the feasible set, even though it's exactly the right tool for the *objective* — the objective `x' Sigma x - q mu' x` is still a lovely convex quadratic, and I don't want to give that up. I want to keep the quadratic and somehow teach the feasible set to be a union of faces.

Let me try the lazy thing first and see how it breaks, because the failure will tell me what I actually need. The lazy thing: solve the continuous QP, then keep the `K` largest weights, zero the rest, renormalise so they sum to one. Truncate-and-renormalise. Is that the optimal `K`-name portfolio? No — and I can see exactly why not. Suppose the continuous optimum holds a big slug of name `A` and a small slug of name `B`, where `B` is small only because it's a near-substitute for `A` and the optimizer split the position. If `K` forces me to drop `B`, the *right* response isn't to keep `A` at its old size and renormalise; it's to re-solve, and the survivors' optimal weights shift because the covariances that `B` was hedging now have to be carried by `A` and the others. Dropping a name changes the whole quadratic interaction among the rest. So the best `K`-name portfolio is generally *not* the top-`K` names of the unconstrained one, nor is it the unconstrained weights renormalised. Truncation answers a different question. And it can leave a survivor below its floor `l_i`, or push one past its cap `u_i`, so even feasibility isn't guaranteed. The lazy patch is out.

Another tempting patch: don't impose a hard count, just *encourage* sparsity with a penalty — add `lambda sum_i |x_i|` to the objective, the way a lasso encourages sparse regression coefficients. Tune `lambda` up until few names survive. Let me think about whether that can ever do what I need. Two problems. First, on the simplex `sum_i x_i = 1` with `x_i >= 0`, the L1 norm `sum_i |x_i|` is just `sum_i x_i = 1` — a constant. It does *nothing*. (I'd have to drop the long-only or the budget for it to even bite, and then it shrinks rather than zeros.) Second, and more fundamentally, even where an L1 penalty does induce sparsity, it does so by *shrinking* — it makes weights small, and only incidentally zero. It gives me a knob that trades sparsity against fit continuously; it can't deliver "exactly at most `K`, and every survivor at least `l_i`." Shrinkage is the opposite of a floor: the floor wants survivors to be *at least* `l_i`, while shrinkage pulls everything *toward* zero. A penalty cannot enforce a hard count or a hard minimum. I need to actually *decide*, per name, in or out — a genuine yes/no — not a soft nudge. The disjunction won't be smoothed away.

OK. So I need to represent a discrete choice — "is name `i` in or out?" — explicitly. That's a binary variable. Introduce `y_i in {0, 1}`, one per name, with the reading `y_i = 1` if I hold name `i`, `0` if I don't. The whole game is now: can I *link* this binary to the continuous weight `x_i` so that the linkage reproduces the disjunction "`x_i = 0` or `x_i in [l_i, u_i]`", using only linear constraints? Because if the link is linear, the only non-convex thing left in the whole problem is the integrality of `y`, and integrality is exactly what branch-and-bound is built to handle.

What linear relationship between `x_i` and `y_i` does the job? I want `y_i = 0` to force `x_i = 0`, and `y_i = 1` to release `x_i` into `[l_i, u_i]`. Try sandwiching `x_i` between two multiples of `y_i`:

`l_i * y_i  <=  x_i  <=  u_i * y_i`.

Check the two cases. If `y_i = 0`: the constraint reads `0 <= x_i <= 0`, so `x_i = 0`. The name is forced out — and crucially, the whole interval `(0, l_i)` is forbidden *because* the only way to make `x_i` positive is to flip `y_i` to `1`, which then drags the *lower* bound up to `l_i`. If `y_i = 1`: it reads `l_i <= x_i <= u_i`, exactly the held interval with its floor and cap. So this single pair of linear inequalities, with one binary, *is* the semi-continuous disjunction. The upper bound `u_i` is doing double duty: it's the position cap, and it's the "big `M`" that, when `y_i = 0`, clamps `x_i` to zero. (If I had no natural cap I'd still need some finite `M_i >= x_i` to write `x_i <= M_i y_i`; here the weight cap `u_i` is the tightest honest choice, and tightness matters — a loose `M` makes the continuous relaxation slacker and the branch-and-bound slower. So I *want* `u_i` to be the real cap, not an arbitrary big number.) The floor `l_i` got encoded for free, as the lower side of the same sandwich. Beautiful — both the cap and the buy-in floor fall out of one linking pair.

Now the cardinality cap. With the indicators in hand it's almost trivial: `y_i = 1` exactly when name `i` is held, so the number held is `sum_i y_i`, and the cap is just

`sum_i y_i  <=  K`.

Linear in the binaries. And `<= K`, not `= K`, on purpose: I want *at most* `K` names; if the optimizer can do better with fewer, let it — forcing exactly `K` could be infeasible or could waste a slot on a name that earns its floor `l_i` but nothing more. So the full discrete portfolio problem is:

minimize `x' Sigma x - q mu' x` (or, flipping signs, maximize `q mu' x - (1/2) x' Sigma x`),
subject to `sum_i x_i = 1`, `l_i y_i <= x_i <= u_i y_i` for all `i`, `sum_i y_i <= K`, `y_i in {0,1}`.

Convex quadratic objective, linear constraints, some binary variables. This is a mixed-integer quadratic program. The objective and all constraints are convex *except* for the integrality of `y`. That single non-convexity is the entire difficulty, and it's a familiar one.

How do I actually solve it? The integrality is the wall, so relax it: let `y_i in [0,1]` instead of `{0,1}`. Now everything is convex — it's a continuous QP again, just with the extra variables `y` and the linking constraints — and I can solve it for a *lower bound* on the true minimum (it's a relaxation, so its optimum can only be better, i.e. smaller, than the constrained one). Solve the relaxation. If every `y_i` comes out `0` or `1` already, I'm done — that relaxed solution is feasible for the original and optimal. Usually it doesn't: some `y_s` comes back fractional, say `y_s = 0.6`, which means "the relaxation wants to hold 60% of a decision," which is meaningless for a real hold/don't-hold. So I *branch* on it: split into two subproblems, one with `y_s = 0` (name `s` forced out — set `x_s = 0` and drop it) and one with `y_s = 1` (name `s` forced in — `l_s <= x_s <= u_s`). Each subproblem is again an MIQP with one fewer free binary; solve each one's relaxation, branch again on any fractional indicator, recurse. This is branch-and-bound: a tree of convex-QP relaxations. The "bound" part is what makes it finite and fast — at any node, if the relaxation's optimal value is already worse than the best integer-feasible solution I've found anywhere in the tree (the *incumbent*), I prune that whole subtree without exploring it, because the relaxation is a lower bound and the subtree can only do worse. The quadratic objective doesn't change this picture at all; it just means each node solves a convex QP rather than an LP. The combinatorial work is the same branching on the discrete decisions.

There's an equivalent way to impose the count that some prefer — instead of binaries, keep only `x` and write the surrogate `sum_i (x_i / u_i) <= K`, then branch on the continuous variables directly: add `x_j <= 0` to drive a name out, or `x_j >= l_j` to drive it in, descending the tree by tightening `x` itself rather than fixing a `y`. The surrogate is exact at integer-support points (when each held `x_i` sits anywhere in `(0, u_i]`, `x_i/u_i <= 1`, so the sum counts held names with weights `<= 1` each, bounded by the count), and branching on `x` avoids ever adding the binaries. Same combinatorial structure — a tree of convex QPs, pruned by the relaxation bound — just stated without the indicator variables and with the bound constraints folded into the node's QP rather than added as rows. I'll keep the binaries because they state the cardinality and the floor most transparently and let a general MIQP solver do the branch-and-bound for me, but it's the same animal either way, and one can tighten the relaxation further with valid inequalities (cuts) — branch-and-cut — when the bound is loose.

Good — cardinality and the floor are handled. Now the round lots, because so far `x_i` is still a real-valued weight and I claimed I trade whole shares. Two ways to read this. The cleaner conceptual one: replace the weight by an integer share (or lot) count directly. If name `i` trades in lots of `s_i` shares at price `p_i`, and my budget is `V`, then holding `n_i` lots is a position worth `n_i s_i p_i`, i.e. a weight `x_i = n_i s_i p_i / V`, and `n_i` is a non-negative integer. Now the *variable itself* is integer, not merely switched by an indicator. The cardinality and floor linking still ride on top: `n_i <= M_i y_i` to switch it off, `n_i >= (l_i V / (s_i p_i)) y_i` for the floor, `sum_i y_i <= K`. So the fully discrete problem has both integer position counts and binary indicators — a richer MIQP, but the same machinery solves it.

In practice it's often cleaner to *separate the concerns*: first solve the cardinality-and-floor MIQP in continuous weights to get the target portfolio `x*` (the strategic decision — which names, in what proportions), then *project* `x*` onto whole shares under the budget (the deployment decision — how many shares of each). The projection is its own little optimization. I have a budget `V`, prices `p`, and target weights `x*`. I want integer share counts `n_i >= 0` whose dollar values `n_i p_i` reproduce the target dollar allocations `x*_i V` as closely as possible, without overspending. "As closely as possible" in what metric? The deviation of name `i` is `x*_i V - n_i p_i` — target dollars minus deployed dollars — and I want all of these small. Minimise the total absolute deviation `sum_i |x*_i V - n_i p_i|`, plus the leftover cash `r = V - sum_i n_i p_i` (so I'm not rewarded for just holding cash). Absolute values aren't linear, but the standard trick is to introduce an upper-bound variable `u_i >= |deviation_i|` via the two inequalities `u_i >= dev_i` and `u_i >= -dev_i`, then minimise `sum_i u_i`; at the optimum `u_i` is squeezed down to exactly `|dev_i|`. With `n_i` integer, `n_i >= 0`, and `r >= 0` (don't overspend), this is a mixed-integer *linear* program — the objective and all constraints are linear; only `n_i` is integer. So the round-lot projection is an MILP, solvable by the same branch-and-bound on a linear relaxation. Naively rounding `x*_i V / p_i` to the nearest integer would, in general, overspend or underspend the budget and could even resurrect a name I'd zeroed or push a kept name below its lot — the MILP respects the budget (`r >= 0`) and minimises the tracking error honestly.

Now the last piece, rebalancing, because I almost never build from cash — I start from a current book `x0` and trade to the new portfolio, and trading costs money. So the objective has to charge for *moving*, not just reward the destination. The trade in name `i` is `x_i - x0_i`. Costs come in pieces. The proportional piece — commissions, half-spread, linear price impact — is a rate times the dollar (or weight) amount traded, and it's charged on buys and sells alike. The amount traded is `|x_i - x0_i|`, so the proportional cost is `tau sum_i |x_i - x0_i|` for a symmetric rate `tau` (or asymmetric, `tau^+` on buys and `tau^-` on sells). That absolute value is the *turnover*, and I'll want to both penalise it in the objective and possibly cap it (a desk may forbid turning over more than, say, 80% of the book in one rebalance).

How do I handle `|x_i - x0_i|` cleanly, especially with asymmetric buy/sell rates? Split the trade into a non-negative buy part and a non-negative sell part:

`x_i - x0_i = b_i - s_i`,  `b_i >= 0`,  `s_i >= 0`.

Then the amount traded is `b_i + s_i`, and the turnover is `sum_i (b_i + s_i)`. Is the split *exact* — does `b_i + s_i` really equal `|x_i - x0_i|` at the optimum, rather than something inflated like `b_i = s_i = 100` for a zero net trade? Yes, and I don't even need an extra constraint to force it. The objective *minimises* cost, hence minimises `sum_i (b_i + s_i)` (turnover enters the cost with a positive coefficient); for a fixed net `x_i - x0_i = b_i - s_i`, driving the smaller of `b_i, s_i` to zero strictly lowers `b_i + s_i` while keeping the net fixed. So the optimizer automatically sets `b_i s_i = 0` — one of them is always zero — and then `b_i + s_i = |x_i - x0_i|` exactly. The split linearises the absolute value for free, and it gives me asymmetric costs (`tau^+ b_i + tau^- s_i`) and a turnover cap (`sum_i (b_i + s_i) <= T_max`) as bonuses. I fold the proportional cost into the objective: maximize `q mu' x - (1/2) x' Sigma x - (tau^+ b + tau^- s summed)`, i.e. the trade cost is *subtracted* from the net return I'm optimizing.

There's a non-convex cost too, the fixed ticket: a flat charge `beta_i` the instant I trade name `i` at all, regardless of size — a step function of `|x_i - x0_i|`, zero at zero and `beta_i` for any nonzero trade. Step functions are non-convex, so, just like the holding decision, I model the *trading* decision with a binary: `t_i in {0,1}`, `t_i = 1` if name `i` is traded, linked by `|x_i - x0_i| <= M t_i` (equivalently `b_i + s_i <= M t_i`), and add `sum_i beta_i t_i` to the cost. Same indicator-plus-big-M pattern as the holding floor — when `t_i = 0` no trade is allowed, when `t_i = 1` a trade up to `M` is, and it pays `beta_i`. (Ticket costs are usually second-order next to proportional impact, so I'll keep them optional, but the modelling is identical.)

So everything reduces to the same two moves: a convex quadratic mean-variance objective, and binary indicators with big-M linking wherever the real world inserts a disjunction — "zero or at least the floor" for holdings, "at most `K` names" for cardinality, "trade or don't" for the fixed cost — plus integer share counts for the round lots, all solved by branch-and-bound on the convex relaxation. The continuous frontier was the easy convex inside; the deployable portfolio is that inside wrapped in a combinatorial shell, and binary variables are the language of the shell.

Let me write it. I'll express the cardinality-and-floor MIQP with `cvxpy`, using a boolean indicator vector `y` and the linking inequalities, the cardinality sum, the mean-variance objective, and an optional rebalancing block with the buy/sell split and a turnover cap. Then the round-lot projection as the integer MILP. A solver that handles a quadratic objective with integer variables (branch-and-bound over convex-QP relaxations) does the work.

```python
import numpy as np
import cvxpy as cp


def cardinality_portfolio(
    mu, Sigma, K,
    w_lower=0.05, w_upper=0.40,     # per-name buy-in floor and cap
    risk_tolerance=1.0,             # q: weight on return vs. (1/2) variance
    w_prev=None,                    # current book, for rebalancing
    tc_buy=0.0, tc_sell=0.0,        # proportional buy/sell cost rates
    turnover_limit=None,            # cap on sum_i |x_i - w_prev_i|
    solver=cp.SCIP,                 # any MIQP-capable solver
):
    n = len(mu)
    x = cp.Variable(n)                  # post-trade weights
    y = cp.Variable(n, boolean=True)    # y_i = 1 iff name i is held

    # mean-variance core:  maximize  q * mu'x  -  (1/2) x'Sigma x
    variance = cp.quad_form(x, Sigma, assume_PSD=True)
    objective = risk_tolerance * (mu @ x) - 0.5 * variance

    constraints = [
        cp.sum(x) == 1,                 # fully invested
        x >= w_lower * y,               # buy-in floor when held (y=1 -> x >= l)
        x <= w_upper * y,               # cap, and y=0 -> x=0 (big-M = w_upper)
        cp.sum(y) <= K,                 # hold at most K names
    ]

    # rebalancing: split the trade into buys/sells, charge and/or cap turnover
    if w_prev is not None and (tc_buy > 0 or tc_sell > 0 or turnover_limit is not None):
        b = cp.Variable(n, nonneg=True)         # buy amounts
        s = cp.Variable(n, nonneg=True)         # sell amounts
        constraints += [x - w_prev == b - s]    # net trade; b_i s_i = 0 at optimum
        turnover = cp.sum(b + s)                # = sum_i |x_i - w_prev_i|
        objective = objective - (tc_buy * cp.sum(b) + tc_sell * cp.sum(s))
        if turnover_limit is not None:
            constraints += [turnover <= turnover_limit]

    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=solver)           # branch-and-bound over convex-QP relaxations
    return np.asarray(x.value).round(6), np.asarray(y.value).round(0).astype(int), prob.value


def round_lot_allocation(weights, prices, total_value, lot_size=1, solver=cp.HIGHS):
    """Project target weights onto whole-lot integer share counts under the budget."""
    p = np.asarray(prices, float) * lot_size        # cost of one lot of each name
    w = np.asarray(weights, float)
    n = len(p)
    lots = cp.Variable(n, integer=True)             # integer lot counts
    cash = total_value - p @ lots                   # leftover cash
    dev = w * total_value - cp.multiply(lots, p)    # target dollars - deployed dollars
    u = cp.Variable(n)
    constraints = [u >= dev, u >= -dev,             # u_i >= |dev_i|
                   lots >= 0, cash >= 0]            # no shorting, don't overspend
    prob = cp.Problem(cp.Minimize(cp.sum(u) + cash), constraints)
    prob.solve(solver=solver)                       # MILP: linear relaxation + branch-and-bound
    return np.rint(lots.value).astype(int), float(cash.value)


if __name__ == "__main__":
    np.random.seed(0)
    mu = np.array([0.12, 0.10, 0.07, 0.03, 0.15, 0.09, 0.05, 0.11])
    A = np.random.randn(8, 8) * 0.1
    Sigma = A @ A.T + np.diag(np.full(8, 0.02))     # PSD covariance

    # at most 3 names, each at least 5%
    x, y, val = cardinality_portfolio(mu, Sigma, K=3, w_lower=0.05, w_upper=0.6,
                                      risk_tolerance=2.0)
    held = x[x > 1e-6]
    print("held:", y, " count:", int(y.sum()), " min held weight:", round(held.min(), 4))

    # rebalance an existing book, charging and capping turnover
    w_prev = np.zeros(8); w_prev[[3, 6]] = [0.5, 0.5]
    x2, _, _ = cardinality_portfolio(mu, Sigma, K=3, w_lower=0.05, w_upper=0.6,
                                     risk_tolerance=2.0, w_prev=w_prev,
                                     tc_buy=0.1, tc_sell=0.1, turnover_limit=1.5)
    print("turnover:", round(np.abs(x2 - w_prev).sum(), 4))

    # deploy the target weights as whole shares on a $100k budget
    prices = np.array([50, 120, 33, 200, 17, 90, 45, 80.0])
    lots, cash = round_lot_allocation(x, prices, total_value=100000.0)
    print("shares:", lots, " cash left:", round(cash, 2))
```

The causal chain, end to end: the continuous mean-variance optimum is a dense, fractional, costless portfolio, but a desk needs at most `K` names, a floor under every held position, and whole shares — and each of those is a disjunction ("zero OR at least `l_i`", "this subset OR that subset", "trade OR don't"), so the feasible set is a non-convex union of faces that no convex reformulation in the weights alone can capture, and no shrinkage penalty can enforce a hard count or a hard floor. The fix is to name each discrete decision with a binary `y_i` and link it to the continuous weight by `l_i y_i <= x_i <= u_i y_i`, which reproduces the semi-continuous floor-or-zero exactly with `u_i` as the big-M, makes the cardinality cap the linear `sum_i y_i <= K`, and turns the whole thing into a mixed-integer quadratic program whose only non-convexity is the integrality — handled by branch-and-bound over convex-QP relaxations, pruned by the relaxation's lower bound. Round lots become integer share counts projected onto the target by an MILP that respects the budget; rebalancing costs become a buy/sell split `x - x0 = b - s` whose `b_i + s_i` is the exact turnover, charged in the objective and optionally capped, with fixed ticket costs modelled by the same indicator-plus-big-M pattern. The convex frontier is the easy inside; binary indicators with big-M linking are the language for the combinatorial shell the real desk lives in.
