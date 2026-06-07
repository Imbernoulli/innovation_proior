# Thermal Unit Commitment: a tight, compact MILP

## Problem

Schedule a fleet of `G` thermal generators over a horizon of `T` hours (typically 24–48). For each unit `g` and hour `t` decide the **on/off status** `u[g,t] in {0,1}` and the **output** `p[g,t]`, so that total generation meets the forecast demand `D[t]` in every hour (with a spinning-reserve margin `R[t]`) at minimum total cost. Fixing the on/off pattern leaves the classic convex **economic dispatch** — easy. The hardness lives in the binary layer and in the *thermal rules that couple hours together*: a started unit must run a minimum number of hours `UT`; a shut unit must stay off `DT` hours; the cost to start rises with how long the unit has been cold (hot vs cold start); and output can change by at most a ramp rate `RU`/`RD` between hours, with special start-up/shut-down allowances `SU`/`SD`. These make it one coupled combinatorial program over the whole horizon, with `~2^{GT}` commitment patterns.

## Key idea

Write the *entire* commitment problem as one **mixed-integer linear program** and hand it to a branch-and-cut solver, which branches on the binaries and prunes with the LP relaxation at each node. The decisive engineering choice is *how* the MILP is written: two formulations with identical integer-feasible sets can have very different LP relaxations, and branch-and-cut is fast only when the relaxation is **tight** (close to the integer convex hull). Three moves make it tight *and* compact:

- **Three binary tracks** linked by `u[g,t] - u[g,t-1] = v[g,t] - w[g,t]` (`v` = turn-on, `w` = turn-off), with `v + w <= 1`. The redundant `v,w` let min up/down be written as **facet** inequalities — the *ideal* (convex-hull-tight) description in only `O(T)` rows, instead of the loose single-`u` form or its exponentially-large exact hull:

      sum_{k=t-UT+1..t} v[g,k] <= u[g,t]          (a start in the trailing UT-window forces still-on)
      sum_{k=t-DT+1..t} w[g,k] <= 1 - u[g,t]      (a shut in the trailing DT-window forces still-off)

- **Time-dependent start-up cost without a selection binary.** Charge a continuous `csu[g,t] >= 0`, lower-bounded once per cold/warm/hot category `s` (which applies after being off `Tlag[s]` hours, cost `Ksu[s]` increasing in `s`):

      csu[g,t] >= Ksu[s] * ( u[g,t] - sum_{i=1..Tlag[s]} u[g,t-i] ).

  The bracket is 1 exactly when the unit has been off long enough to be in category `s`, else `<= 0` (slack). Since `csu` is minimized, the LP drives it to the largest active bound — the correct, more-expensive cold-start cost — picking the category *for free*.

- **Piecewise-linear convex production cost.** Replace the quadratic `a+b·p+c·p²` by `L` segments with increasing marginal costs. Because the objective is minimized and slopes increase, cheap segments fill before expensive ones automatically — no ordering binaries or SOS sets needed.

Generation band `Pmin·u <= p <= Pmax·u` (off ⇒ 0, on ⇒ in band), ramp couplings, and the system-coupling demand balance + reserve close the model. Because the min-up/down piece is in facet form, the node LP is strong: shallow tree, provable optimality gap in operational time — the certificate Lagrangian relaxation could never give.

## Algorithm

1. Variables: binary `u,v,w`; continuous output-above-minimum `p >= 0`; reserve headroom `r >= 0`; per-category start indicator `d[g,s,t]` (binary) with `sum_s d = v`; piecewise fill `lam[g,l,t] in [0,1]`; production cost `cprod[g,t]`.
2. State logic `u-u_prev = v-w` (using each unit's initial status for `t=1`); pin initial must-run/must-stay-off hours.
3. Facet min-up and min-down inequalities over trailing windows.
4. Generation upper limit `p + r <= (Pmax-Pmin)·u - max(Pmax-SU,0)·v`; lower band via `Pmin·u`.
5. Ramp-up / ramp-down couplings on consecutive `p`.
6. Piecewise cost: `p = sum_l (mw_l-mw_0)·lam`, `cprod = sum_l (cost_l-cost_0)·lam`, `u = sum_l lam`.
7. Per-hour demand balance `sum_g (p + Pmin·u) = D[t]` and reserve `sum_g r >= R[t]`.
8. Objective = production + min-output running cost + start-up cost. Solve with branch-and-cut to a target MIP gap.

## Code

Pyomo over a branch-and-cut backend (CBC/Gurobi). Self-contained; the three binary tracks give the tight constraints, start-up cost is charged through a per-category indicator the objective minimizes over, and the convex piecewise cost fills cheapest-first.

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
