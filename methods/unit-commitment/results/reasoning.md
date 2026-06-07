Let me start from the actual operational pain. I run a fleet of thermal generators and every day I have to decide, for each of the next 24 hours, which units are switched on and how much each one produces, so that the total exactly meets the forecast demand in every hour — and I want to do it at minimum cost. That's the whole job. The instinct is to reach for an optimizer, but before I do that I want to understand *what kind* of problem this is, because that determines everything.

So let me strip it down. Suppose someone handed me the on/off pattern — told me exactly which units are running in each hour. What's left? In each hour I have a set of running units, each with a cost-versus-output curve (roughly quadratic, a + b·p + c·p², convex), each constrained to its band P ≤ p ≤ P̄, and I have to split that hour's demand D among them to minimize total fuel cost: minimize Σ cost_g(p_g) subject to Σ p_g = D. That's a convex program. Its optimum is the classic equal-incremental-cost condition — push output onto units until they all sit at the same marginal cost λ, the system price. It solves in milliseconds, exactly. This is *economic dispatch*, and it's easy. The hours don't even couple if the only constraint is meeting demand each hour: I could solve 24 independent dispatch problems.

So the difficulty is not the dispatch. The difficulty is that I *don't* get handed the on/off pattern — choosing it is the problem. Let me introduce the obvious variable: u_g(t) ∈ {0,1}, one if unit g is on in hour t, zero if off. The moment those are variables, everything changes. First, the generation band has to become conditional on u: if the unit is off it must produce exactly nothing, if on it must be inside its band. I can write that as a single pair of linear inequalities,

  P_g·u_g(t) ≤ p_g(t) ≤ P̄_g·u_g(t).

When u=0 this pins p to 0; when u=1 it's the band. Linear in the variables, exact. So far so good — if that were all, I'd have a problem with binaries but no coupling across time, and I could almost solve each hour separately.

But it isn't all, and this is where the real structure is. Thermal units are physical objects with thermal inertia, and that imposes rules that tie hour t to earlier hours:

— Once I start a unit, I can't just shut it next hour; it has to run at least some minimum number of hours, UT. Minimum up time.
— Once I shut it, it has to stay off at least DT hours before I can restart. Minimum down time.
— Starting a unit costs money, and *how much* depends on how long it's been off: a unit that just cooled is cheap to reheat (hot start), one that's been cold for hours is expensive (cold start). A start-up cost that's a step function of how long it's been off.
— And its output can't jump arbitrarily between consecutive hours — there's a ramp limit RU going up, RD coming down.

Every one of those couples hour t to hours before it. So I can't decompose by hour anymore. It's one big coupled problem over the whole horizon, and the coupling is through *integer* decisions. With G units over T hours there are on the order of 2^{GT} on/off patterns. That's the combinatorial explosion. The easy convex dispatch is sitting inside a hard combinatorial shell.

Let me think about how people have attacked this, because I don't want to walk into known dead ends.

The crudest thing is a priority list: rank units by their average full-load cost, cheapest first, and each hour commit down the list until you've covered demand plus a reserve margin. It's instant and a human can audit it. But it's myopic — it decides each hour in isolation, so it has no way to honor "you must keep this thing on for UT hours" or to weigh a start-up cost against a few hours of cheaper running. It can be badly suboptimal and gives no bound on how bad. Not good enough when start-up costs and time-coupling are exactly what's expensive.

The exact thing is dynamic programming over commitment states: let the state be the vector of which units are on, and find the least-cost path across hours. This *can* respect the time-coupling, and it's optimal. But the state is a length-G binary vector, so there are 2^G states per hour. For a handful of units, fine; for a real fleet of hundreds, completely hopeless. The curse of dimensionality kills it.

The approach that actually ran the grid for two decades is Lagrangian relaxation, and it's worth understanding precisely *why* it was attractive and precisely *where* it falls short, because that's what motivates moving past it. Look at the constraint structure. Almost every constraint — the band, min up/down, ramps, start cost — involves a *single* unit. Only two kinds of constraint tie *different* units together: the hourly demand balance Σ_g p_g(t) = D(t), and the reserve requirement. So if I take those coupling constraints and pull them into the objective with multipliers λ(t) — a Lagrangian — the relaxed problem *separates*. It decomposes into G independent single-unit problems, each one a tiny dynamic program over that one unit's own on/off history (only that unit's states, so no 2^G blowup). Each subproblem is cheap. Then I do a subgradient iteration on λ(t) to push the system back toward demand balance. That's elegant and it scales, because the heavy lifting is per-unit and parallel.

But here's the wall. The commitment variables are integers, so this problem is *nonconvex*. For a nonconvex problem the Lagrangian dual doesn't close the gap: the best bound I get from the dual sits strictly *below* the true optimum — there's a genuine duality gap. Worse, the schedule I read off the relaxed subproblems is almost never feasible — the individually-optimized units don't add up to exactly meet demand each hour. So I have to bolt on a heuristic to repair the schedule into something feasible, and now I've got a feasible schedule whose distance from optimal I can't certify, produced by a procedure with finicky step-size and repair tuning. I get a bound and a workable schedule, but not a *provably near-optimal* one. That residual gap and the lack of a certificate is the thing I'd want to remove.

So let me reconsider the whole frame. Instead of decomposing and patching, what if I write the *entire* commitment problem — all the binaries, all the coupling — as one single mixed-integer linear program, and hand it to a general-purpose branch-and-cut solver? The solver branches on the binaries, and at every node of its search tree it solves the *LP relaxation* (the same model but with u allowed to be fractional in [0,1]) to get a bound it uses to prune. Modern MILP solvers are extremely good at this — they'll prove a small optimality gap, which is exactly the certificate Lagrangian relaxation couldn't give me.

But there's an immediate catch, and it's the whole game. The LP relaxation bound is only useful if it's *strong*. Two MILPs can have the identical set of integer-feasible points and yet completely different LP relaxations: one whose fractional feasible region hugs tightly around the integer convex hull, and one that bulges far out from it. If the relaxation is *tight* — close to the convex hull — then the LP bound is strong, fractional optima are rare, branch-and-cut barely has to branch, and it's fast. If it's *loose*, the bound is weak and the solver enumerates an astronomical tree. So the deciding factor isn't "use MILP" — it's *how I write the MILP*. The contribution I'm after is a tight encoding of these combinatorial constraints. And there's a counter-pressure: a tighter description usually needs *more* variables and constraints, which makes each LP node bigger and slower. So I'm trading tightness against compactness, and the sweet spot is judged empirically against a real solver. Good — now I know what I'm optimizing: a formulation that's tight enough to give strong bounds, compact enough to be cheap per node, and uses as few binaries as I can manage.

Let me now actually build the constraints, one combinatorial rule at a time, each time worrying about linearity first and tightness second.

The generation band I already have, linear and exact: P·u(t) ≤ p(t) ≤ P̄·u(t).

Now minimum up time. The English is: "if the unit was off in hour t−1 and on in hour t — i.e. it just started — then it must stay on for the next UT hours." The trouble is that "just started" is a logical event and "must stay on" is an implication, and implications aren't linear. Let me find the start event arithmetically. The quantity u(t) − u(t−1) is +1 exactly when the unit turns on at t, −1 when it turns off, 0 when nothing changes. So "turned on at t" is captured by u(t) − u(t−1) = 1. Now I want: when that's 1, the next UT statuses are all forced to 1. Consider the sum of the status over the window of UT hours starting at t:

  Σ_{n=t}^{t+UT−1} u(n).

If the unit just turned on at t, I need every term in this window to be 1, i.e. the sum must be at least UT. If the unit did *not* just turn on, I don't want to force anything. So write

  Σ_{n=t}^{t+UT−1} u(n) ≥ UT·(u(t) − u(t−1)).

Check it. Turn-on at t: RHS = UT·(1) = UT, and since each u≤1 the only way the LHS can reach UT is all UT of them equal to 1 — exactly the minimum-up requirement. No turn-on, or a turn-off: RHS ≤ 0, the inequality is slack, no constraint imposed. That's a correct linear encoding of minimum up time using only the single binary u. The window runs over the "middle" of the horizon (t from after the initial must-run hours up to T−UT+1); near the end of the horizon, where a full UT-window would run past T, I instead require Σ_{n=t}^{T}(u(n) − (u(t) − u(t−1))) ≥ 0 so a late start still forces all remaining hours on; and at the very start I force the unit on for whatever residual must-run hours its initial history demands.

Minimum down time is the mirror image. "Turned off at t" is u(t−1) − u(t) = 1, and "stay off DT hours" means the *off*-ness, 1−u(n), must be 1 across the window:

  Σ_{n=t}^{t+DT−1} (1 − u(n)) ≥ DT·(u(t−1) − u(t)).

Same logic with on/off swapped. So both min up and min down are linear in the single binary u. This is a *compact* encoding — only the u variables, one inequality per unit per hour.

But now I have to ask the tightness question, because compactness alone isn't the goal. How tight is the LP relaxation of this u-only min-up/down encoding? Here's the worry: when u is allowed to be fractional, the term u(t) − u(t−1) can take fractional values, and the inequality Σ u(n) ≥ UT·(u(t)−u(t−1)) becomes a fairly weak statement about fractional u's — there are fractional points satisfying it that are far from any integer schedule. In fact it's known that if I insist on describing the min-up/down feasible set *exactly* (its convex hull) using only the u variable, I need exponentially many inequalities — the single-binary convex hull is not compactly describable. So the compact u-only version is loose, and the exact u-only version is huge. Both bad. I'm stuck between a weak relaxation and an explosion.

The way out is to add a *redundant* variable that carries information the single u can't express compactly. The natural one: an explicit start-up indicator. Let v(t) = 1 if the unit turns on at hour t, and while I'm at it w(t) = 1 if it turns off at t. These are redundant — they're determined by u — but introducing them lets me *factor* the logic. The link between them is just the arithmetic I was already using:

  u(t) − u(t−1) = v(t) − w(t),    and   v(t) + w(t) ≤ 1

(it can't simultaneously turn on and off). Now watch what min up/down become. The English "if it started within the last UT−1 hours it had better still be on" turns into: the unit is on at t if it started in any of the last UT hours. The cleanest way to say it is that you can have *at most one* start in any UT-long window that ends with the unit still on:

  Σ_{i=t−UT+1}^{t} v(i) ≤ u(t).

If the unit started at any i in that trailing UT-window, the LHS is at least 1, so u(t) is forced to 1 — it's still on, exactly minimum up. And symmetrically for down, using turn-offs:

  Σ_{i=t−DT+1}^{t} w(i) ≤ 1 − u(t).

Here's the payoff, and it's the reason to pay the price of the extra variables: these turn-on/turn-off inequalities are *facets* of the convex hull of the min-up/down polytope. Together with the linking equation and the variable bounds, they give the *ideal* — the tightest possible — description of the min-up/down constraints, and they do it with only O(T) inequalities, not exponentially many. So the redundant v (and w) buy me both tightness *and* compactness for the hardest combinatorial piece. That's the trade I want: a couple of extra variables per unit-hour in exchange for an LP relaxation that, for this sub-problem, equals the integer convex hull. And since u(t)−u(t−1)=v(t)−w(t) lets me eliminate whichever of v,w I don't otherwise need, I can even keep the variable count down — typically I keep v because start-ups carry costs and project out w. The initial conditions get pinned directly: if the unit must be on for its first U hours of history, Σ_{i=1}^{min{U,T}} u(i) = min{U,T}; if it must stay off for D hours, Σ_{i=1}^{min{D,T}} u(i) = 0.

So I have a choice of two encodings of min up/down — the compact single-u one and the tight v-augmented facet one — and the lesson crystallizing here is that the *facet* form is what makes branch-and-cut fast, at the cost of one extra binary track. That's the central engineering knob of the whole formulation.

Now the time-dependent start-up cost, and the v variable I just introduced is going to make this clean too. The physics: cost to start rises with how long the unit's been off. Discretize into a few categories — say hot, warm, cold — indexed s = 1 (hottest) … S (coldest), where category s applies if the unit has been off for at least T_s hours, with costs K_s^SU increasing in s. The naïve modeling temptation is to add a binary that *selects* which category each start falls into — but that's exactly the kind of extra combinatorial decision and extra binaries I'm trying to avoid, and it bloats the model. Let me see if I can get the right cost without a selection binary at all.

Introduce a single continuous auxiliary cost c^SU(t) ≥ 0 per hour, the start-up cost charged at t, and lower-bound it once per category:

  c^SU(t) ≥ K_s^SU·( u(t) − Σ_{i=1}^{T_s} u(t−i) )   for each category s.

Read the bracket. At a genuine start, u(t)=1. The sum Σ_{i=1}^{T_s} u(t−i) counts how many of the previous T_s hours the unit was *on*. If the unit had been off throughout that look-back window — off for at least T_s hours — the sum is 0 and the bracket is 1, so this category-s bound becomes c^SU(t) ≥ K_s^SU. If the unit had been on at some point inside the window, the sum is ≥1 and the bracket is ≤0, so that category's bound is slack. So each category contributes a lower bound that is active exactly when the unit has been off long enough to be in (at least) that category. Now — crucially — c^SU(t) appears with a positive coefficient in the objective I'm *minimizing*. So the LP will drive c^SU(t) down to the largest of its active lower bounds. The categories nest: being off ≥ T_s hours implies off ≥ T_{s'} for any earlier (hotter, cheaper) category, so the colder the unit, the more category-bounds are active, and the *binding* one is the right, more-expensive cold-start cost. The minimization does the category selection for free — no selection binary, just one continuous variable and S linear inequalities per hour. Exactly the compactness I wanted, and it pins the correct time-dependent cost. (If I'm carrying v explicitly I can equally write these in terms of v(t) and the intervening w's; same mechanism.)

Production cost next. The true cost-of-output curve is convex quadratic, a + b·p + c·p². I want a *linear* program inside each node, so I replace the quadratic by a piecewise-linear convex underestimate: split the output band into L segments, let δ_l(t) be the amount produced in segment l, with p(t) = P·u(t) + Σ_l (slope-widths)·δ_l(t) and the segment widths bounded, and charge Σ_l C_l·δ_l(t) with the marginal costs C_l increasing across segments (convexity). And here convexity hands me a gift: because the marginal cost rises segment by segment and I'm minimizing, the solver fills the cheap low segments before the expensive high ones automatically — I do *not* need ordering binaries or special-ordered-set machinery to enforce "fill segment 1 before segment 2." The convex objective enforces it for free. So piecewise cost is just continuous segment variables plus linear bounds. (Compactly, one can even drop the per-segment variables and bound p(t) ≤ (P̄_l − P̄_{l−1})·u(t) per segment as a simpler tightening.)

Ramp limits last — these are what genuinely make the problem inter-temporal in the *continuous* variables, not just the binary ones. Output can't jump too far hour to hour, except that a unit coming on or going off gets a special allowance (it's allowed to come up to its start-up rate SU, or down to its shut-down rate SD). Writing it directly:

  p(t) − p(t−1) ≤ RU·u(t−1) + SU·v(t)     (ramp up: normal rate RU if it was already on, the start-up rate SU on the hour it starts)
  p(t−1) − p(t) ≤ RD·u(t)  + SD·w(t)      (ramp down: rate RD if it stays on, shut-down rate SD on the hour it shuts)

Both linear, both coupling consecutive hours through the same v, w I already have.

And I still need the two system-coupling constraints — the ones Lagrangian relaxation used to dualize, but which I now just keep as hard linear constraints because branch-and-cut handles them directly: demand balance Σ_g p_g(t) = D(t) every hour, and the spinning-reserve requirement. Reserve is cleanest if each unit also reports its *maximum available* power p̄_g(t) (what it could ramp up to if needed), bounded by P̄_g·u_g(t) and by its ramp headroom, and require Σ_g p̄_g(t) ≥ D(t) + R(t).

Stack it all up. The objective is total cost: Σ_t Σ_g [ piecewise production cost + start-up cost c^SU(t) (+ shut-down cost if modeled) ]. The variables: binary u_g(t) (and the start/stop indicators v_g(t), w_g(t)), continuous p_g(t), the production-cost segment variables, and the auxiliary c^SU(t). The constraints: conditional generation band; the facet min-up/down inequalities in v/w; the per-category start-up-cost lower bounds; the piecewise production-cost relations; the ramp-up/ramp-down couplings; and the hourly demand balance and reserve. That whole thing is a single mixed-integer linear program. I hand it to a branch-and-cut solver, which branches on the u's and bounds with the LP relaxation — and *because* I used the tight facet form of min up/down and the no-extra-binary start-cost trick, that LP relaxation is strong, so the tree stays manageable and the solver proves a small gap fast. That tightness is the difference between solving a real fleet in minutes and not solving it at all.

Let me write it in a concrete modeling layer. Pyomo over a branch-and-cut backend (CBC/Gurobi). I'll keep the three binary tracks u, v, w because they give me the tight constraints, model startup cost by categories with a per-category indicator that the objective minimizes over, use the λ-style piecewise production cost, and assemble exactly the constraints I just derived.

```python
import pyomo.environ as pyo

def build_uc(data):
    gens   = data["thermal_generators"]    # each: Pmin, Pmax, UT, DT, RU, RD, SU, SD,
                                            #       startup categories [(lag, cost)...],
                                            #       piecewise [(mw, cost)...], init state
    T      = data["time_periods"]
    demand = data["demand"]; reserve = data["reserves"]

    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=list(gens))
    m.T = pyo.RangeSet(1, T)

    # --- variables ---
    m.p = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)   # output above minimum
    m.r = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)   # reserve (headroom) provided
    m.u = pyo.Var(m.G, m.T, within=pyo.Binary)             # on/off
    m.v = pyo.Var(m.G, m.T, within=pyo.Binary)             # turn-on
    m.w = pyo.Var(m.G, m.T, within=pyo.Binary)             # turn-off
    # startup-category indicator and piecewise-segment fill
    m.d  = pyo.Var(((g,s,t) for g in m.G for s in startups(gens[g]) for t in m.T),
                   within=pyo.Binary)
    m.lam = pyo.Var(((g,l,t) for g in m.G for l in pieces(gens[g]) for t in m.T),
                    within=pyo.UnitInterval)
    m.cprod = pyo.Var(m.G, m.T)                            # production cost above minimum

    # --- objective: production + min-output running cost + start-up cost ---
    def obj(m):
        return sum(
            m.cprod[g,t]
            + gens[g]["piecewise"][0]["cost"] * m.u[g,t]                  # cost at P_min when on
            + sum(su["cost"] * m.d[g,s,t] for s,su in enumerate(gens[g]["startup"]))
            for g in m.G for t in m.T)
    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    for g in m.G:
        gen = gens[g]
        Pmin, Pmax = gen["Pmin"], gen["Pmax"]
        for t in m.T:
            # state-transition logic:  u(t) - u(t-1) = v(t) - w(t)
            prev = gen["u0"] if t == 1 else m.u[g,t-1]
            m.add_component(f"logic_{g}_{t}",
                pyo.Constraint(expr = m.u[g,t] - prev == m.v[g,t] - m.w[g,t]))

            # tight (facet) minimum up time:  sum of turn-ons in trailing UT window <= u(t)
            UT = min(gen["UT"], T)
            if t >= UT:
                m.add_component(f"minup_{g}_{t}",
                    pyo.Constraint(expr =
                        sum(m.v[g,k] for k in range(t-UT+1, t+1)) <= m.u[g,t]))
            # tight minimum down time:  turn-offs in trailing DT window <= 1 - u(t)
            DT = min(gen["DT"], T)
            if t >= DT:
                m.add_component(f"mindn_{g}_{t}",
                    pyo.Constraint(expr =
                        sum(m.w[g,k] for k in range(t-DT+1, t+1)) <= 1 - m.u[g,t]))

            # a start at t is exactly one of its categories
            m.add_component(f"startsel_{g}_{t}",
                pyo.Constraint(expr =
                    m.v[g,t] == sum(m.d[g,s,t] for s,_ in enumerate(gen["startup"]))))

            # generation upper limit (with start/shut ramp tightening); lower band via P_min*u
            m.add_component(f"gmax_{g}_{t}",
                pyo.Constraint(expr =
                    m.p[g,t] + m.r[g,t] <= (Pmax-Pmin)*m.u[g,t]
                                           - max(Pmax-gen["SU"],0)*m.v[g,t]))

            # ramps couple consecutive hours
            if t > 1:
                m.add_component(f"rampup_{g}_{t}",
                    pyo.Constraint(expr = m.p[g,t]+m.r[g,t]-m.p[g,t-1] <= gen["RU"]))
                m.add_component(f"rampdn_{g}_{t}",
                    pyo.Constraint(expr = m.p[g,t-1]-m.p[g,t] <= gen["RD"]))

            # piecewise-linear convex production cost (segments fill cheap-first automatically)
            pc = gen["piecewise"]
            m.add_component(f"psel_{g}_{t}",
                pyo.Constraint(expr =
                    m.p[g,t] == sum((pc[l]["mw"]-pc[0]["mw"])*m.lam[g,l,t]
                                    for l,_ in enumerate(pc))))
            m.add_component(f"csel_{g}_{t}",
                pyo.Constraint(expr =
                    m.cprod[g,t] == sum((pc[l]["cost"]-pc[0]["cost"])*m.lam[g,l,t]
                                        for l,_ in enumerate(pc))))
            m.add_component(f"onsel_{g}_{t}",
                pyo.Constraint(expr =
                    m.u[g,t] == sum(m.lam[g,l,t] for l,_ in enumerate(pc))))

    # --- system coupling: demand balance and reserve each hour ---
    def balance(m, t):
        return sum(m.p[g,t] + gens[g]["Pmin"]*m.u[g,t] for g in m.G) == demand[t-1]
    m.balance = pyo.Constraint(m.T, rule=balance)
    def reserve_rule(m, t):
        return sum(m.r[g,t] for g in m.G) >= reserve[t-1]
    m.reserve = pyo.Constraint(m.T, rule=reserve_rule)
    return m

def solve(m, solver="cbc", mipgap=0.01):
    opt = pyo.SolverFactory(solver)
    return opt.solve(m, options={"ratioGap": mipgap}, tee=True)   # branch-and-cut
```

The causal chain, start to finish: the goal is least-cost hourly generation meeting demand; fix the on/off pattern and what's left is convex economic dispatch, which is easy — so all the hardness lives in the binary on/off decisions u and the thermal rules that couple them across time. I write the conditional generation band and the demand/reserve balance as linear constraints; the combinatorial rules — minimum up/down — I linearize, first compactly in u alone, then notice that the compact form has a loose LP relaxation while an exact u-only description blows up, so I add a redundant turn-on (and turn-off) indicator that factors the logic into facet inequalities that are simultaneously tight and O(T); the time-dependent start-up cost I charge through a continuous auxiliary lower-bounded per category, letting the cost-minimizing objective pick the right category with no extra selection binary; the production cost I make piecewise-linear and convex so segments fill cheapest-first without ordering variables; ramps couple consecutive outputs. The result is one mixed-integer linear program, and because the min-up/down piece is in its tight facet form, the LP relaxation that branch-and-cut solves at every node is strong — strong bounds, a shallow tree, a provable optimality gap in operational time, which is exactly what the decompose-and-repair approaches could never certify.
