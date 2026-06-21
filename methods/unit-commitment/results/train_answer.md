Every day I have to schedule a fleet of $G$ thermal generators over the next $T$ hours — typically a 24-to-48-hour horizon — deciding for each unit $g$ and each hour $t$ both whether it is switched on, $u_g(t)\in\{0,1\}$, and how much it produces, $p_g(t)$, so that total generation meets the forecast demand $D(t)$ in every hour (with a spinning-reserve margin $R(t)$) at least total cost. The structure of the problem is worth pinning down before reaching for a solver, because it dictates everything. If someone handed me the on/off pattern, what is left is, in each hour, splitting that hour's demand among the running units to minimize fuel cost, $\min \sum_g \mathrm{cost}_g(p_g)$ subject to $\sum_g p_g = D$, where each unit has a convex (roughly quadratic, $a_g + b_g p + c_g p^2$) cost over its band $\underline P_g \le p_g \le \overline P_g$. That is classic economic dispatch — a convex program whose optimum is the equal-incremental-cost condition, all marginal units sitting at one system price $\lambda$ — and it solves exactly in milliseconds. The entire difficulty lives outside that easy core, in the binary on/off layer, and specifically in the thermal rules that couple hours together: once started, a unit must run at least $UT_g$ consecutive hours; once shut, it must stay off at least $DT_g$ hours; the cost to restart rises with how long the unit has been cold (a hot start is cheap, a cold start expensive); and output cannot jump arbitrarily between hours, bounded by ramp rates $RU_g$, $RD_g$, with special start-up and shut-down allowances $SU_g$, $SD_g$. Each of these ties hour $t$ to earlier hours, so the commitment cannot be decided period by period — it is one coupled combinatorial program over the whole horizon, with on the order of $2^{GT}$ on/off patterns.

The existing options all fall short in a definite way. A priority list — rank units by average full-load cost, commit down the list each hour until capacity covers demand plus reserve — is instant and auditable but myopic: deciding each hour in isolation, it cannot honor minimum up/down times or weigh a start-up cost against a few hours of cheaper running, and it gives no bound on its suboptimality. Exact dynamic programming over the per-hour commitment vector respects the time-coupling and is optimal, but the state is a length-$G$ binary vector, $2^G$ states per hour, so it is hopeless beyond a handful of units. Lagrangian relaxation ran the grid for two decades by exploiting that only the demand-balance and reserve constraints couple different units: dualize those with multipliers $\lambda(t)$ and the relaxed problem separates into $G$ independent single-unit dynamic programs, cheap and parallel, with subgradient updates driving the system back toward balance. But the commitment variables are integer, so the problem is nonconvex; the dual bound generally sits strictly below the true optimum with a duality gap I cannot drive to zero, and the schedule read off the relaxed subproblems is almost never feasible — the units do not add up to demand each hour — so it must be patched by a heuristic, leaving a feasible-ish schedule of uncertified quality.

What I propose is to write the *entire* commitment problem as a single mixed-integer linear program and hand it to a general-purpose branch-and-cut solver, which branches on the binaries and at every node of its search tree bounds with the LP relaxation (the same model with $u$ allowed fractional in $[0,1]$) to prune — yielding exactly the provable optimality-gap certificate that decompose-and-patch could never produce. The catch is the whole game: two MILPs can have the identical integer-feasible set and yet wildly different LP relaxations, one hugging the integer convex hull and one bulging far from it. Branch-and-cut is fast only when the relaxation is *tight*, because a tight relaxation gives strong bounds, makes fractional optima rare, and keeps the tree shallow; a loose one forces enumeration of an astronomical tree. So the contribution is not "use MILP" — it is *how* the MILP is written, and the design is a deliberate trade of tightness against compactness (a tighter description usually needs more variables and rows, which slows each node), judged against a real solver. I call the method a tight, compact MILP formulation of thermal unit commitment, and three moves are what make it work.

The first move is the encoding of minimum up and down time. The generation band is conditional and linear, $\underline P_g\, u_g(t) \le p_g(t) \le \overline P_g\, u_g(t)$, pinning an off unit to zero and an on unit into its band. For minimum up time the English is "if the unit just turned on at $t$, the next $UT$ statuses must all be one." The turn-on event is arithmetic: $u(t)-u(t-1)$ equals $+1$ exactly at a start, $-1$ at a shut, $0$ otherwise, so the compact single-$u$ encoding is
$$\sum_{n=t}^{t+UT-1} u(n) \;\ge\; UT\,\bigl(u(t)-u(t-1)\bigr),$$
which forces all $UT$ statuses to one at a start and is slack otherwise. This is compact but its LP relaxation is loose: with $u(t)-u(t-1)$ taking fractional values, fractional points far from any integer schedule satisfy it, and it is known that describing the min-up/down feasible set *exactly* using only $u$ requires exponentially many inequalities. Stuck between a weak relaxation and an explosion, the way out is a *redundant* variable that carries information the single $u$ cannot express compactly. I introduce an explicit turn-on indicator $v(t)=1$ when the unit starts at $t$ and a turn-off indicator $w(t)=1$ when it shuts, linked to $u$ by the same arithmetic,
$$u_g(t) - u_g(t-1) = v_g(t) - w_g(t), \qquad v_g(t) + w_g(t) \le 1$$
(a unit cannot start and shut in the same hour). With these, min up/down become statements about at most one start (or shut) in a trailing window:
$$\sum_{k=t-UT+1}^{t} v_g(k) \;\le\; u_g(t), \qquad \sum_{k=t-DT+1}^{t} w_g(k) \;\le\; 1-u_g(t).$$
If the unit started anywhere in the trailing $UT$-window the left side is at least one, forcing $u_g(t)=1$ — still on — and symmetrically for shuts. The payoff that justifies the extra variables is that these turn-on/turn-off inequalities are *facets* of the convex hull of the min-up/down polytope: together with the linking equation and the bounds they give the ideal, convex-hull-tight description in only $O(T)$ rows rather than exponentially many. That is the trade I want — a couple of extra binaries per unit-hour buy an LP relaxation that, for this hardest sub-problem, equals the integer convex hull. Initial conditions are pinned directly: a unit entering with accrued on-hours that leave $UT$ unfinished is forced on for the residual hours, one entering with accrued off-hours is forced off for its residual down-hours.

The second move charges the time-dependent start-up cost without a selection binary. The cost to start rises with how long the unit has been off; I discretize into categories $s$, hottest to coldest, where category $s$ applies after the unit has been off at least $T_s$ hours, with costs $K_s^{SU}$ increasing in $s$. The naive temptation is a binary that *selects* which category a start falls in — exactly the extra combinatorial decision and model bloat I am avoiding. Instead I carry a single continuous auxiliary $c^{SU}(t)\ge 0$ and lower-bound it once per category,
$$c^{SU}(t) \;\ge\; K_s^{SU}\Bigl(\,u(t) - \sum_{i=1}^{T_s} u(t-i)\Bigr).$$
The bracket is one exactly when the unit genuinely starts ($u(t)=1$) and was off throughout the $T_s$-hour look-back (the sum is zero), and is $\le 0$ otherwise, leaving that category slack. Because $c^{SU}(t)$ enters the minimized objective with a positive coefficient, the LP drives it down to the largest active lower bound; the categories nest (off $\ge T_s$ implies off past every hotter threshold), so the colder the unit the more bounds are active and the binding one is the correct, more-expensive cold-start cost — the minimization selects the category for free, with one continuous variable and $S$ linear rows per hour. In the assembled model this is realized with per-category indicators $d_g(s,t)$ summing to the start, $\sum_s d_g(s,t)=v_g(t)$, each eligible only if the unit shut within that category's offline window, the coldest category an unconstrained catch-all, and the minimized startup cost picking the cheapest eligible one.

The third move is the production cost. The true curve is convex quadratic, but each node must solve an LP, so I sample $L$ breakpoints and replace the curve by its interpolating piecewise-linear function — convex, with marginal costs increasing across segments. Convexity hands a gift: because the slopes rise and the objective is minimized, the solver fills the cheap low segments before the expensive high ones automatically, with *no* ordering binaries or special-ordered-set machinery. I write this with one fill weight $\lambda_l(t)\in[0,1]$ per breakpoint, $\sum_l \lambda_l(t)=u(t)$, reading off both output above minimum and cost above minimum as the same convex combination. Ramps couple consecutive outputs through $v,w$; tracking output above minimum and folding the start-up/shut-down allowance into the generation upper limit collapses the inter-temporal rows to plain ramp-rate bounds. Finally the two system-coupling constraints that Lagrangian relaxation used to dualize are simply kept as hard linear rows — demand balance $\sum_g p_g(t)=D(t)$ each hour and the reserve requirement $\sum_g r_g(t)\ge R(t)$ via each unit's reported headroom. Stacking the conditional band, the facet min-up/down inequalities, the per-category start-cost bounds, the convex piecewise cost, the ramp couplings, and the system rows yields one MILP; because the min-up/down piece is in facet form the node LP is strong, so the tree stays shallow and branch-and-cut proves a small gap in operational time — the certificate the decompose-and-repair approaches could never give.

```python
import pyomo.environ as pyo


def build_uc(data):
    """data = {
        'gens': { name: {
            'Pmin','Pmax','UT','DT','RU','RD','SU','SD',
            'u0','p0','init_status_hours',          # +on / -off run length entering hour 1
            'startup':  [(lag_hours, cost), ...],   # increasing lag, increasing cost
            'piecewise':[(mw, cost), ...] } },       # mw[0]=Pmin, increasing; convex costs
        'T': int, 'demand': [..T..], 'reserves': [..T..] }
    """
    gens, T = data['gens'], data['T']
    demand, reserve = data['demand'], data['reserves']

    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=list(gens))
    m.T = pyo.RangeSet(1, T)

    nS = {g: len(gens[g]['startup'])   for g in gens}     # start-up categories
    nL = {g: len(gens[g]['piecewise']) for g in gens}     # cost segments

    # --- variables ---
    m.p  = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)   # output above Pmin
    m.r  = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)   # reserve headroom
    m.u  = pyo.Var(m.G, m.T, within=pyo.Binary)             # on/off
    m.v  = pyo.Var(m.G, m.T, within=pyo.Binary)             # turn-on
    m.w  = pyo.Var(m.G, m.T, within=pyo.Binary)             # turn-off
    m.d  = pyo.Var(((g, s, t) for g in m.G for s in range(nS[g]) for t in m.T),
                   within=pyo.Binary)                        # start-up category
    m.lam = pyo.Var(((g, l, t) for g in m.G for l in range(nL[g]) for t in m.T),
                    within=pyo.UnitInterval)                  # piecewise fill
    m.cprod = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)  # cost above Pmin-cost

    def U(g, t):   # status of unit g entering hour t (t may be <= 0)
        return gens[g]['u0'] if t < 1 else m.u[g, t]

    def W(g, t):   # turn-off indicator at hour t; a pre-horizon shut shows up at t<1
        gen = gens[g]
        if t >= 1:
            return m.w[g, t]
        # entering off with `off0` accrued down-hours == a shut at hour `1-off0`
        off0 = -gen['init_status_hours'] if gen['u0'] == 0 else 0
        return 1 if t == 1 - off0 else 0

    # --- objective: production + min-output running cost + start-up cost ---
    def obj(m):
        return sum(
            m.cprod[g, t]
            + gens[g]['piecewise'][0][1] * m.u[g, t]                       # cost at Pmin when on
            + sum(gens[g]['startup'][s][1] * m.d[g, s, t] for s in range(nS[g]))
            for g in m.G for t in m.T)
    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    m.con = pyo.ConstraintList()
    for g in m.G:
        gen = gens[g]
        Pmin, Pmax = gen['Pmin'], gen['Pmax']
        UT, DT = min(gen['UT'], T), min(gen['DT'], T)
        pc = gen['piecewise']

        # initial must-run / must-stay-off implied by entering history
        hist = gen['init_status_hours']            # +on hours / -off hours already accrued
        if hist > 0:
            must_on = max(0, gen['UT'] - hist)
            for t in range(1, min(must_on, T) + 1):
                m.con.add(m.u[g, t] == 1)
        elif hist < 0:
            must_off = max(0, gen['DT'] + hist)
            for t in range(1, min(must_off, T) + 1):
                m.con.add(m.u[g, t] == 0)

        for t in m.T:
            # state transition:  u(t) - u(t-1) = v(t) - w(t),   v + w <= 1
            m.con.add(m.u[g, t] - U(g, t - 1) == m.v[g, t] - m.w[g, t])
            m.con.add(m.v[g, t] + m.w[g, t] <= 1)

            # tight (facet) minimum up time
            if t >= UT:
                m.con.add(sum(m.v[g, k] for k in range(t - UT + 1, t + 1)) <= m.u[g, t])
            # tight (facet) minimum down time
            if t >= DT:
                m.con.add(sum(m.w[g, k] for k in range(t - DT + 1, t + 1)) <= 1 - m.u[g, t])

            # a start at t falls in exactly one cost category
            m.con.add(m.v[g, t] == sum(m.d[g, s, t] for s in range(nS[g])))
            # category s (a hotter, cheaper one) is allowed only if the unit shut down
            # within s's offline window [lag_s, lag_{s+1}) hours ago; the coldest
            # category nS-1 is the catch-all (unconstrained), and since startup cost is
            # minimized the LP picks the cheapest *eligible* category for free
            for s in range(nS[g] - 1):
                lag, lag_next = gen['startup'][s][0], gen['startup'][s + 1][0]
                m.con.add(m.d[g, s, t] <=
                          sum(W(g, t - i) for i in range(lag, lag_next)))

            # generation: lower band via Pmin*u below; upper with start-up tightening,
            # and a shut-down tightening at t (a unit shutting at t+1 can't be at the top)
            m.con.add(m.p[g, t] + m.r[g, t] <=
                      (Pmax - Pmin) * m.u[g, t] - max(Pmax - gen['SU'], 0.0) * m.v[g, t])
            if t < T:
                m.con.add(m.p[g, t] + m.r[g, t] <=
                          (Pmax - Pmin) * m.u[g, t]
                          - max(Pmax - gen['SD'], 0.0) * m.w[g, t + 1])

            # ramps couple consecutive hours' output-above-minimum
            p_prev = gen['u0'] * (gen['p0'] - Pmin) if t == 1 else m.p[g, t - 1]
            m.con.add(m.p[g, t] + m.r[g, t] - p_prev <= gen['RU'])
            m.con.add(p_prev - m.p[g, t] <= gen['RD'])

            # piecewise-linear convex production cost (segments fill cheap-first automatically)
            m.con.add(m.p[g, t]   == sum((pc[l][0] - pc[0][0]) * m.lam[g, l, t] for l in range(nL[g])))
            m.con.add(m.cprod[g, t] == sum((pc[l][1] - pc[0][1]) * m.lam[g, l, t] for l in range(nL[g])))
            m.con.add(m.u[g, t]   == sum(m.lam[g, l, t] for l in range(nL[g])))

    # --- system coupling: demand balance and spinning reserve each hour ---
    def balance(m, t):
        return sum(m.p[g, t] + gens[g]['Pmin'] * m.u[g, t] for g in m.G) == demand[t - 1]
    m.balance = pyo.Constraint(m.T, rule=balance)

    def reserve_rule(m, t):
        return sum(m.r[g, t] for g in m.G) >= reserve[t - 1]
    m.reserve = pyo.Constraint(m.T, rule=reserve_rule)
    return m


def solve(m, solver='cbc', mipgap=0.01, tee=True):
    opt = pyo.SolverFactory(solver)            # branch-and-cut backend
    return opt.solve(m, options={'ratioGap': mipgap}, tee=tee)
```
