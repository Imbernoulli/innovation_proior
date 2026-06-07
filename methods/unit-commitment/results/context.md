# Context: Scheduling a fleet of thermal generators over a day

## Research question

A power system operator must decide, hour by hour over a planning horizon (typically the next 24–48 hours), which thermal generating units to run and how much each should produce, so that total generation meets the forecast electricity demand in every hour at the least possible cost. The decision is not just "how much from each unit" — it is also "which units are switched on at all," and the two are entangled: a unit that is off produces nothing, a unit that is on must produce at least its technical minimum, and switching a unit on incurs a start-up cost and commits it to run for a while.

If the on/off pattern were fixed in advance, the remaining problem — split the demand among the running units to minimize fuel cost — is the classic *economic dispatch*, a smooth convex problem solved in milliseconds. What makes the full problem hard is that the on/off pattern is itself a decision, and a constrained one: a unit cannot be cycled freely. Once started it must stay up for a minimum number of hours; once shut it must stay down for a minimum number of hours; how much a start costs depends on how long the unit has been cold; and a unit's output cannot jump arbitrarily between consecutive hours (ramp limits). These rules couple decisions across time and make the on/off choices a combinatorial problem layered on top of the easy continuous one.

The operational stakes are large: this scheduling problem is solved every day for the wholesale electricity market, the schedule must be produced within roughly ten to fifteen minutes of solver time for fleets of hundreds to thousands of units, and small percentage improvements in cost translate into very large absolute savings across a national grid. The question this landscape poses: how to write the combinatorial commitment problem so that a general-purpose optimizer can actually solve it to provable near-optimality at that scale and speed.

## Background

**Economic dispatch is the easy core.** Fix which units are committed. Each unit g has a convex (typically quadratic, a_g + b_g p + c_g p²) cost as a function of its output p_g, and must lie in its band P_g ≤ p_g ≤ P̄_g. Minimizing Σ_g cost(p_g) subject to Σ_g p_g = D is a convex program whose optimum satisfies the equal-incremental-cost condition: all marginal units run at the same marginal cost λ (the system "price"). This is solvable quickly and exactly. The entire difficulty of the larger problem lives outside this core, in the binary on/off layer.

**Why the commitment layer is combinatorial.** Let u_g(t) ∈ {0,1} be the on/off status of unit g in hour t. The generation band becomes conditional — P_g u_g(t) ≤ p_g(t) ≤ P̄_g u_g(t), so an off unit is forced to zero output and an on unit into its band. On top of this sit four temporally-coupling realities of thermal plant:
- **Minimum up time** UT_g: once a unit is started, thermal and mechanical stresses mean it must run at least UT_g consecutive hours before it may be shut.
- **Minimum down time** DT_g: once shut, it must stay off at least DT_g hours before being restarted.
- **Time-dependent start-up cost**: a boiler that has only just cooled is cheap to reheat (a "hot" start); one that has been off for many hours is expensive (a "cold" start). The start cost is a nondecreasing step function of the number of hours the unit has been off.
- **Ramp limits** RU_g, RD_g: output can change by at most a bounded amount between consecutive hours; start-up and shut-down impose their own rate limits SU_g, SD_g.

Each of these is a constraint that links hour t to earlier hours, so the commitment decisions cannot be made period-by-period independently — the problem is one coupled combinatorial program over the whole horizon. With G units and T hours there are on the order of 2^{GT} commitment patterns; brute force and naïve dynamic programming over commitment states both blow up (the per-period state space alone is 2^G).

**The historical solution method — Lagrangian relaxation.** From the 1980s into the 2000s the dominant practical approach exploited a structural observation: the only constraints that tie *different* units together are the system-wide ones (demand balance and reserve); every other constraint (limits, min up/down, ramps, start cost) involves a single unit. Dualize the coupling constraints with multipliers λ(t), and the relaxed Lagrangian *separates* into G independent single-unit subproblems, each a small dynamic program over that unit's own states, solved cheaply. A subgradient iteration updates λ(t) toward demand balance. This made large fleets tractable, but it has a structural weakness: because the commitment variables are integer, the problem is nonconvex, so there is a **duality gap** — the best dual bound lies strictly below the true optimum, and the schedule recovered from the relaxed subproblems is generally *infeasible* (the hours don't balance) and must be patched by heuristics, yielding a schedule of unknown sub-optimality and no tight certificate.

**The shift to mixed-integer programming.** The alternative is to write the entire commitment problem as a single mixed-integer linear program and hand it to a general branch-and-cut solver. This became the industry approach in the mid-2000s (US system operators transitioned away from Lagrangian-relaxation heuristics toward MILP around 2005), driven by maturing commercial solvers; the estimated cost savings from better schedules are very large (on the order of billions of dollars annually across the US grid, per retrospective accounts). The decisive technical issue, established repeatedly in the optimization literature, is that *how* the MILP is written matters enormously. Two formulations with identical integer-feasible sets can have very different **LP relaxations** (the problem with the binaries allowed to take fractional values). Branch-and-cut prunes its search tree using LP-relaxation bounds; a formulation whose LP relaxation hugs the convex hull of the integer-feasible set (a *tight* formulation) yields strong bounds and a shallow tree, while a loose one forces the solver to enumerate enormously. But a tighter description usually needs more variables and constraints, which slows each node — so practical formulation design is a **tightness-versus-compactness** trade-off, judged empirically against a solver.

**What was known about tightening the pieces.** Several structural results predate the compact MILP target here. Garver (1962) already modeled a generator with three binary state variables — on (u), turned-on (v), turned-off (w) — and with piecewise-linear production costs. Lee et al. (2004) showed that, using the single on/off variable alone, the convex hull of the minimum up/down-time constraints needs exponentially many inequalities (though it can be separated in polynomial time). Malkin (2003) and Rajan & Takriti (2005) independently showed that *adding the turn-on variable* collapses this to an O(T) facet description — the tightest compact encoding of min up/down. Arroyo & Conejo (2000) gave linear ramp-up/ramp-down and shut-down constraints. Nowak & Römisch (2000) gave a discrete (hot/cold) representation of time-dependent start-up cost. These are the building blocks a compact, reasonably-tight MILP would assemble.

## Baselines

**Priority-list commitment.** Order the units by average full-load cost (cheapest "base-load" first), and in each hour commit units down the list until capacity covers demand plus reserve. Trivially fast and transparent. But it ignores min up/down times, start-up costs, and ramps; it makes each hour's decision myopically; and it offers no optimality guarantee — it can be far from least cost, especially when start-up costs and time-coupling matter.

**Dynamic programming over commitment states.** Treat the per-hour commitment vector as a state and find the least-cost path over the horizon by Bellman recursion. Exact in principle and able to honor time-coupling, but the state space is exponential in the number of units (2^G commitment patterns per hour), so it is intractable beyond a handful of units; truncation heuristics restore tractability only by sacrificing optimality.

**Lagrangian relaxation (LR).** The pre-MILP workhorse. Relax the coupling demand/reserve constraints with multipliers; the Lagrangian decomposes into independent single-unit subproblems, each solved by a small dynamic program; iterate the multipliers by subgradient steps to drive the system constraints toward satisfaction. Scales to large fleets because the work is per-unit. Its gaps: a nonzero **duality gap** for this nonconvex problem (the dual bound undershoots the true optimum); a relaxed solution that is typically primal-**infeasible** and needs heuristic recovery (Lagrangian heuristics, list-scheduling repairs); and sensitivity to subgradient step-size and repair tuning. It delivers a bound and a feasible-ish schedule, not a provably near-optimal one.

**Earlier MILP formulations.** Before the compact target here, MILP encodings of UC existed but were heavy — they used more binary variables (e.g. separate binaries to *select* the start-up cost category, or to index commitment transitions) and more constraints than necessary, and/or relaxations that were loose. The result was MILPs that either had weak LP bounds (slow branch-and-cut) or large node sizes, limiting them to small systems. The open gap: a formulation with *fewer* binaries and constraints whose LP relaxation is still tight enough that a commercial branch-and-cut solver handles realistic fleets within operational time limits — precisely modeling the time-dependent start-up cost and the inter-temporal min up/down and ramp constraints as linear inequalities.

## Evaluation settings

The natural yardstick is a set of thermal-fleet test systems: a small canonical 10-unit system (replicated to 20, 40, 60, 100 units to scale up), and larger fleets of hundreds of units, each unit specified by its cost curve (a_g, b_g, c_g or piecewise points), min/max output, min up/down times, ramp rates, start-up rates, hot/cold start-up costs and thresholds, and initial on/off status. Each instance gives an hourly demand profile and a spinning-reserve requirement over a 24-hour (sometimes 36–48-hour) horizon. The metrics of interest are total scheduling cost, the branch-and-cut solve time (and number of nodes) to reach a target optimality gap, and the size of the formulation (counts of binary variables and constraints, and the LP-relaxation bound as a proxy for tightness). The solver is a general-purpose mixed-integer programming engine (branch-and-cut, e.g. CPLEX/Gurobi/CBC). A natural point of comparison is a Lagrangian-relaxation schedule and earlier, heavier MILP encodings, run on the same instances.

## Code framework

A pre-existing algebraic-modeling stack (an MILP modeling layer such as Pyomo over a branch-and-cut solver) supplies the primitives: declare index sets, declare continuous and binary variables over those sets, write linear constraints and a linear objective, and call a solver. The economic-dispatch core — a continuous LP that splits demand among fixed-on units — is the easy, already-understood slot. The empty stubs are the commitment binaries and the *linearization* of each combinatorial rule: how to force a started unit to stay up (and a shut unit to stay down) for its minimum time using linear inequalities, how to charge the correct time-dependent start cost without a combinatorial category search, how to represent the convex production cost piecewise, and how to couple consecutive-hour outputs by ramp limits — then hand the assembled model to branch-and-cut.

```python
import pyomo.environ as pyo

def build_uc(gens, T, demand, reserve):
    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=list(gens))
    m.T = pyo.RangeSet(1, T)

    # power output of each unit each hour (continuous; the easy economic-dispatch variable)
    m.p = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)

    # --- the combinatorial layer: empty stubs ---

    def commitment_vars(m):
        # TODO: binary on/off status u[g,t] (and any auxiliary state variables
        #       that make the time-coupling constraints linear and tight)
        pass

    def generation_limits(m):
        # TODO: an off unit produces 0, an on unit lies in [P_g, Pbar_g]
        pass

    def min_up_down(m):
        # TODO: a unit started in hour t must stay on >= UT_g hours;
        #       a unit shut in hour t must stay off >= DT_g hours.
        #       Express as LINEAR inequalities in the commitment variables,
        #       as tight as possible (small LP-relaxation gap).
        pass

    def startup_cost(m):
        # TODO: charge a time-dependent start cost (hot vs cold) that depends on
        #       how many hours the unit has been off, as linear constraints —
        #       without a separate combinatorial search over start categories.
        pass

    def production_cost(m):
        # TODO: represent the convex production cost piecewise-linearly.
        pass

    def ramp_limits(m):
        # TODO: bound |p[g,t] - p[g,t-1]| by ramp rates, with start/shut allowances.
        pass

    def system_constraints(m):
        # demand balance and reserve each hour (the coupling constraints)
        def balance(m, t):
            return sum(m.p[g, t] for g in m.G) == demand[t]
        m.balance = pyo.Constraint(m.T, rule=balance)
        # TODO: reserve requirement using max-available-power per unit
        pass

    def objective(m):
        # TODO: total = production cost + start-up cost (+ shut-down cost), summed
        pass

    commitment_vars(m); generation_limits(m); min_up_down(m)
    startup_cost(m); production_cost(m); ramp_limits(m)
    system_constraints(m); objective(m)
    return m

def solve(m, solver="cbc", mipgap=0.01):
    opt = pyo.SolverFactory(solver)
    return opt.solve(m, tee=True)   # branch-and-cut
```
