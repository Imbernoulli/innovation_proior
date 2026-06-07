# Synthesis — Thermal Unit Commitment, Carrión–Arroyo tight MILP

## Sources actually retrieved & read this run
1. **Primary**: Carrión & Arroyo 2006, "A Computationally Efficient Mixed-Integer Linear Formulation for the Thermal Unit Commitment Problem," IEEE Trans. Power Syst. 21(3):1371–1378. Full text behind IEEE paywall; its *exact equations* reproduced/cataloged in the Knueven review below (which explicitly treats CA-2006 as the base formulation and reproduces each CA constraint with attribution), cross-checked against the canonical CA min-up form returned by web search and against the PGLib Pyomo model. Gap flagged: original PDF not obtained; equations taken from faithful secondary (Knueven et al.) + canonical implementations.
2. **Background / lineage**: Padhy 2004 "Unit Commitment — A Bibliographical Survey," IEEE Trans. Power Syst. 19(2):1196–1205 (the survey establishing the field, methods taxonomy: priority list, dynamic programming, Lagrangian relaxation, MILP). Garver 1962 (3-binary u,v,w + piecewise costs — the ancestor). Lee et al. 2004 (1-bin min up/down convex hull is exponential, separable in poly time). Rajan & Takriti 2005 / Malkin 2003 (the O(T) facet min-up/down with turn-on variable). Arroyo & Conejo 2000 (ramping + shutdown limits). Nowak & Römisch 2000 (discrete startup-cost categories). Lagrangian relaxation as the pre-MILP industry approach (PJM switched to MILP in 2005; ~$5B/yr US savings — O'Neill 2017, cited in Knueven).
3. **Third-party explainer**: Knueven, Ostrowski & Watson, "On Mixed Integer Programming Formulations for the Unit Commitment Problem," INFORMS J. Comput. 32(4):857–876 (2020), Sandia open-access SAND2019-13440J (full text PDF read, pp.1–23) + the same authors' FERC tutorial slides (W1-A-1-Knueven.pdf, read). These catalog CA-2006 as the base and walk tight-vs-compact, 1-bin vs 3-bin, lineage of each constraint.
4. **Code**: PGLib-UC reference Pyomo MILP model (`power-grid-lib/pglib-uc/uc_model.py`) — runnable ConcreteModel with u,v,w binaries, piecewise λ production costs, startup categories d[g,s,t], ramp/min-up-down constraints. This is the canonical implementation the final code mirrors.

## The problem (first-principles object)
Schedule a fleet of thermal generators over T hourly periods to meet (forecast) demand D(t) at minimum total cost.
- min Σ_t Σ_g [ production cost of p_g(t) + start-up cost + (shut-down cost) ]
- s.t. power balance Σ_g p_g(t) = D(t) (+ reserve Σ ≥ R(t))
- each generator: if on, P_g ≤ p_g(t) ≤ P̄_g; if off, p_g(t)=0.
Continuous-only (fixed on/off) → economic dispatch → convex (LP/QP), easy: equal-incremental-cost / λ. The hardness is the **on/off decisions** + the temporal coupling they create: minimum up time UT (once on, stay on ≥UT h), minimum down time DT, start-up cost depends on how long off, ramp limits couple p(t) to p(t−1). These make it a large mixed-integer program (binaries u_g(t) ∈ {0,1}). NP-hard combinatorial.

## Lineage / why each prior step falls short
- **Priority list**: rank units by avg full-load cost, commit cheapest until demand met. Fast, but ignores min-up/down, startup, ramps → far from optimal, no optimality guarantee.
- **Dynamic programming** over commitment states: optimal but state space 2^G per period → curse of dimensionality, intractable for real fleets.
- **Lagrangian relaxation (LR)** — the pre-MILP workhorse (1980s–2000s, PJM until 2005). Dualize the *coupling* constraints (demand balance, reserve) with multipliers λ(t); the relaxed problem **decouples per generator** into G single-unit subproblems, each solved by DP over that unit's states (cheap). Update λ by subgradient on the dual. But: (1) **duality gap** — UC is nonconvex (integers), so the dual optimum < primal optimum; the dual solution is usually *infeasible* (demand not met) and needs heuristic repair, giving a suboptimal schedule with no tight bound; (2) tuning the subgradient/repair heuristics is finicky. LR gives a bound + a feasible-ish schedule, not a provably near-optimal one.
- **MILP + branch-and-cut** (this method's frame): write the *whole* UC as one MILP and hand it to a general solver (CPLEX/Gurobi). Branch-and-bound on the binaries; the LP relaxation at each node gives a bound; cutting planes tighten it; modern solvers prove small optimality gaps. The catch that makes-or-breaks it: the **formulation**. Two MILPs with the same feasible integer set can have wildly different LP relaxations; a **tight** one (LP relaxation close to the integer convex hull) prunes the B&B tree fast, a loose one explodes. So the contribution is a *tight & compact* MILP encoding of the combinatorial constraints.

## The derivation (insight → formula)
### Binary commitment & the easy parts
u_g(t)∈{0,1} = on/off. Generation limits couple p to u so "off ⇒ 0 output, on ⇒ within band":
  P_g u_g(t) ≤ p_g(t) ≤ P̄_g u_g(t).             (linear, exact)
Power balance Σ_g p_g(t) = D(t); reserve Σ_g p̄_g(t) ≥ D(t)+R(t) with p_g ≤ p̄_g ≤ P̄_g u_g(t).

### Min up/down — the combinatorial core, made linear
Naïve "if it just turned on, it must stay on UT periods" is a logical implication, not linear. CA's published form (single-binary u): split the horizon into initial / middle / final blocks.
**Min-up (CA-2006 form), middle:** for t = (block after initial must-on) … T−UT+1,
  Σ_{n=t}^{t+UT−1} u(n) ≥ UT·(u(t) − u(t−1)).
Reading: if u(t)−u(t−1)=1 (a turn-on at t), the RHS = UT, forcing all UT periods from t on to be 1; if no turn-on (RHS ≤0) the constraint is slack. Final block (t > T−UT+1): Σ_{n=t}^{T} (u(n) − (u(t)−u(t−1))) ≥ 0. Initial block: force u=1 for the residual must-on hours from history.
**Min-down** symmetric with (1−u): Σ_{n=t}^{t+DT−1} (1−u(n)) ≥ DT·(u(t−1) − u(t)).
This is the *compact* CA encoding. It is correct but **not tight** — its LP relaxation is loose.

**The tighter alternative (Garver 3-binary, Rajan–Takriti facets)** — what tightness costs and buys. Add v(t)=turn-on, w(t)=turn-off, redundant but enabling. Logic: u(t) − u(t−1) = v(t) − w(t), and v(t)+w(t) ≤ 1. Then min up/down become the **facet** inequalities (Malkin 2003 / Rajan & Takriti 2005):
  Σ_{i=t−UT+1}^{t} v(i) ≤ u(t)         (can't have been turned on within the last UT−1 h unless still on)
  Σ_{i=t−DT+1}^{t} w(i) ≤ 1 − u(t)     (symmetric for down)
These are *facets of the up/down polytope* → LP relaxation = integer convex hull for the min-up/down sub-problem → the ideal (tightest) compact formulation. Initial conditions: Σ_{i=1}^{min{U,T}} u(i)=min{U,T}; Σ_{i=1}^{min{D,T}} u(i)=0.

### Start-up cost (time-dependent, hot/cold), linear
Cost of starting depends on how long the unit has been off (cold start costs more). Discretize into S categories s=1(hottest)…S(coldest), each with threshold T_s offline hours and cost K_s^SU (increasing). Auxiliary continuous c^SU(t) ≥ 0 with, for each category s,
  c^SU(t) ≥ K_s^SU·( u(t) − Σ_{i=1}^{T_s} u(t−i) ).
At a turn-on, u(t)=1; the sum Σ_{i=1}^{T_s} u(t−i) counts on-hours in the last T_s periods. If the unit was off for ≥T_s of them, that bracket =1 → the category-s lower bound on c^SU(t) is active. The objective *minimizes* c^SU(t), so the LP sets it to the **largest binding** RHS = the correct (cheapest applicable, since longer-off ⇒ later category with higher K but the brackets nest so the tightest active one wins) category cost — no extra binaries for category selection. (3-bin variant: c^SU(t) ≥ K_s^SU v(t) − Σ K... using w's; same idea.)

### Piecewise-linear production cost
Real cost is quadratic c_g(p)=a u + b p + c p². Convexity ⇒ replace by L-segment piecewise-linear convex underestimate. Garver/λ form: p_g(t) = P_g u(t) + Σ_l (P̄_l − P̄_{l−1}) δ_l(t) with 0≤δ_l(t)≤... and cost Σ_l C_l δ_l(t); convexity means the solver fills cheap segments first automatically — no SOS/ordering binaries needed. CA used the simpler segment-bound form p_l(t) ≤ (P̄_l − P̄_{l−1}) u(t).

### Ramp limits (Arroyo–Conejo 2000)
  p(t) − p(t−1) ≤ RU·u(t−1) + SU·v(t)     (ramp-up; relaxed to SU on a start-up)
  p(t−1) − p(t) ≤ RD·u(t)  + SD·w(t)      (ramp-down; SD on a shut-down)
Couple consecutive periods — this is what makes UC genuinely temporal, not period-separable.

### Why tight matters (the whole point)
Branch-and-cut on the binaries: each node solves the LP relaxation for a bound. If the LP feasible region hugs the integer convex hull (tight), the bound is strong, fractional solutions are rare, the tree is shallow → fast. A loose compact model has a big LP–IP gap → many nodes → slow. CA-2006's contribution: fewer binaries & constraints than predecessors **and** a reasonably tight encoding of startup/min-up-down — the practical sweet spot that made full-MILP UC tractable for solvers, displacing Lagrangian relaxation in practice. Subsequent work (Rajan–Takriti facets, Morales-España ramps, Knueven startup) pushed tightness further; the engineering lesson is the **tight↔compact trade-off**, evaluated empirically.

## Code framework (pre-method scaffold ↔ final code correspondence)
Scaffold: a generic MILP harness — sets (gens, periods), an LP-easy economic-dispatch slot, and empty stubs for: commitment binaries, min-up/down linearization, startup-cost linearization, piecewise cost, ramp coupling, balance/reserve, objective, solve. Final code = PGLib-style Pyomo ConcreteModel filling exactly those stubs (u/v/w, λ piecewise, d[g,s,t] startup categories, ramp & limit constraints, demand/reserve, branch-and-cut solve via cbc/gurobi).

## Unsourced / flagged
- Original CA-2006 PDF not directly obtained (IEEE paywall + RG/aggregator dead ends after budgeted search). All CA equations taken from the Knueven et al. review (which reproduces CA-2006 as the base formulation with attribution), the FERC tutorial, the canonical web-returned CA min-up form, and cross-checked against the PGLib Pyomo model. Gap flagged here and in final report.
- No cost numbers / solver timings invented. The "$5B/yr" and "PJM 2005" are from Knueven/O'Neill (context) and used only as field-state background, not as method results.
