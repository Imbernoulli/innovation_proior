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

**The historical solution method — Lagrangian relaxation.** From the 1980s into the 2000s the dominant practical approach exploited a structural observation: the only constraints that tie *different* units together are the system-wide ones (demand balance and reserve); every other constraint (limits, min up/down, ramps, start cost) involves a single unit. Dualize the coupling constraints with multipliers λ(t), and the relaxed Lagrangian *separates* into G independent single-unit subproblems, each a small dynamic program over that unit's own states, solved cheaply. A subgradient iteration updates λ(t) toward demand balance, and heuristic repairs are applied to the recovered schedule to restore feasibility. This made large fleets tractable.

**The shift to mixed-integer programming.** The alternative is to write the entire commitment problem as a single mixed-integer linear program and hand it to a general branch-and-cut solver. This became the industry approach in the mid-2000s (US system operators transitioned away from Lagrangian-relaxation heuristics toward MILP around 2005), driven by maturing commercial solvers; the estimated cost savings from better schedules are very large (on the order of billions of dollars annually across the US grid, per retrospective accounts). The decisive technical issue, established repeatedly in the optimization literature, is that *how* the MILP is written matters enormously. Two formulations with identical integer-feasible sets can have very different **LP relaxations** (the problem with the binaries allowed to take fractional values). Branch-and-cut prunes its search tree using LP-relaxation bounds; a formulation whose LP relaxation hugs the convex hull of the integer-feasible set (a *tight* formulation) yields strong bounds and a shallow tree, while a loose one forces the solver to enumerate enormously. But a tighter description usually needs more variables and constraints, which slows each node — so practical formulation design is a **tightness-versus-compactness** trade-off, judged empirically against a solver.

**Prior work on the individual pieces.** A body of literature studies how to encode the individual thermal constraints as MILP. Garver (1962) modeled a generator with binary state variables and with piecewise-linear production costs. Lee et al. (2004) studied the convex hull of the minimum up/down-time constraints expressed in the single on/off variable alone, characterizing how many inequalities a complete description requires. Arroyo & Conejo (2000) gave linear ramp-up/ramp-down and shut-down constraints. Nowak & Römisch (2000) gave a discrete (hot/cold) representation of time-dependent start-up cost. These formulation studies are the inherited prior art on the individual constraints.

## Baselines

**Priority-list commitment.** Order the units by average full-load cost (cheapest "base-load" first), and in each hour commit units down the list until capacity covers demand plus reserve. Each hour's decision is made independently based on the sorted list.

**Dynamic programming over commitment states.** Treat the per-hour commitment vector as a state and find the least-cost path over the horizon by Bellman recursion. This respects time-coupling and is exact in principle; the state space has 2^G commitment patterns per hour. Truncation heuristics reduce the per-period search to a manageable subset of states.

**Lagrangian relaxation (LR).** The pre-MILP workhorse. Relax the coupling demand/reserve constraints with multipliers; the Lagrangian decomposes into independent single-unit subproblems, each solved by a small dynamic program; iterate the multipliers by subgradient steps to drive the system constraints toward satisfaction. The work scales with G because it is done per-unit. The relaxed solution is restored to feasibility through Lagrangian heuristics and list-scheduling repairs.

**Earlier MILP formulations.** MILP encodings of UC existed prior to this work, using binary variables to select the start-up cost category or to index commitment transitions, along with associated constraint sets. These formulations varied in their LP-relaxation tightness and in the number of variables and constraints they introduced.

## Evaluation settings

The natural yardstick is a set of thermal-fleet test systems: a small canonical 10-unit system (replicated to 20, 40, 60, 100 units to scale up), and larger fleets of hundreds of units, each unit specified by its cost curve (a_g, b_g, c_g or piecewise points), min/max output, min up/down times, ramp rates, start-up rates, hot/cold start-up costs and thresholds, and initial on/off status. Each instance gives an hourly demand profile and a spinning-reserve requirement over a 24-hour (sometimes 36–48-hour) horizon. The metrics of interest are total scheduling cost, the branch-and-cut solve time (and number of nodes) to reach a target optimality gap, and the size of the formulation (counts of binary variables and constraints, and the LP-relaxation bound as a proxy for tightness). The solver is a general-purpose mixed-integer programming engine (branch-and-cut, e.g. CPLEX/Gurobi/CBC). A natural point of comparison is a Lagrangian-relaxation schedule and earlier, heavier MILP encodings, run on the same instances.

## Code framework

A pre-existing algebraic-modeling stack (an MILP modeling layer such as Pyomo over a branch-and-cut solver) supplies the primitives: declare index sets, declare continuous and binary variables over those sets, write linear constraints and a linear objective, and call a solver. The economic-dispatch core — a continuous LP that splits demand among fixed-on units — is the easy, already-understood slot. The empty stubs are the commitment binaries and the *linearization* of each combinatorial rule: how to force a started unit to stay up (and a shut unit to stay down) for its minimum time, how to charge the correct time-dependent (hot/cold) start cost, how to represent the convex production cost, and how to couple consecutive-hour outputs by ramp limits — then hand the assembled model to branch-and-cut.

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
        # TODO: the binary on/off decision variables for the commitment layer
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
        #       how many hours the unit has been off, as linear constraints.
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
