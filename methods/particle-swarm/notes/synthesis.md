# Synthesis — Particle Swarm Optimization

## Sources (three-source bottom line)
1. PRIMARY: Kennedy & Eberhart 1995, "Particle Swarm Optimization," Proc. IEEE ICNN, pp.1942-1948. Read in full (refs/kennedy1995_pso.pdf). Plus Shi & Eberhart 1998 "A Modified Particle Swarm Optimizer" (inertia weight) — equations recovered from the inertia-strategies survey (refs/inertia_strategies_nabic.pdf) which reproduces eqs (1)-(3) of the 1995/1998 forms.
2. BACKGROUND: Reynolds 1987 boids (separation/alignment/cohesion, local info, emergence); Heppner & Grenander 1990 cornfield/roost bird model; E.O. Wilson 1975 sociobiology (social sharing of information); Millonas 1994 five principles of swarm intelligence; GA/EP as contemporary alternatives; derivative-free optimization of multimodal black boxes. Clerc & Kennedy 2002 constriction/stability (for the parameter stability bound).
3. EXPLAINER: pyswarms docs/docstrings (canonical numpy implementation); Bansal et al. 2011 "Inertia Weight Strategies in PSO" (survey reproducing the equations and the 0.9→0.4 linear-decreasing recipe, c1=c2=2).

## Key facts / equations (all sourced)
- 1995 ORIGINAL velocity update (no inertia, factor 2):
  vx[][] = vx[][] + 2*rand()*(pbest - present) + 2*rand()*(gbest - present)
  position: present += vx. (Kennedy 1995, §3.6)
- The 2 makes the stochastic factor mean 1, so agents "overfly" the target about half the time (Kennedy 1995 §3.6).
- Etiology chain (Kennedy 1995 §3): nearest-neighbor velocity matching → flock settles to unanimous direction (collapse) → add "craziness" (random perturbation) for lifelike variation → Heppner cornfield vector (a roost/food attractor) eval=sqrt((x-100)^2)+sqrt((y-100)^2) → each agent remembers its own best (pbest) and the group best (gbest) → adjust velocity toward both → eliminate craziness and nearest-neighbor matching (work fine without them; slightly faster) → "acceleration by distance": replace sign test with the actual difference (pbest-present) → simplified version with factor 2.
- FAILED variants tried & rejected (Kennedy 1995 §3.7): (a) collapse both terms to one toward the MIDPOINT of pbest/gbest — converges on that midpoint even when it isn't an optimum → "two stochastic kicks are necessary"; (b) explorers/settlers two-agent-type version → no improvement; (c) REMOVE momentum (vx = 2rand(pbest-x)+2rand(gbest-x), no carried-over velocity) → "quite ineffective at finding global optima" → momentum/inertia is load-bearing.
- INERTIA (Shi & Eberhart 1998, eq (3) in survey):
  v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x);  x = x + v
  Large w → global search/exploration; small w → local search/exploitation. Motivation: eliminate the need for Vmax. Linear-decreasing w from 0.9 to 0.4 over the run is the recommended recipe (Bansal survey, ref [6]); c1=c2=2 default.
- Vmax: 1995/pre-inertia PSO clamped |v|<=Vmax to stop velocity explosion (a constraint controlling global exploration). Inertia w was introduced partly to replace Vmax (survey).
- STABILITY/constriction (Clerc & Kennedy 2002): with φ=c1+c2, constriction χ=2/|2−φ−√(φ²−4φ)| for φ>4; common choice φ=4.1 → χ≈0.7298, c1=c2=2.05. χ damps velocity to guarantee convergence (prevents explosion). The constriction form is algebraically equivalent to inertia with w=χ and c_i'=χ·c_i; w<1 with bounded c1+c2 is the stability condition.
- Millonas five principles: proximity, quality, diverse response, stability, adaptability (Kennedy 1995 §4).
- pbest = "autobiographical memory / simple nostalgia"; gbest = "publicized knowledge / group norm" (Kennedy 1995 §3.3).
- p_increment ≈ g_increment (cognitive ≈ social) best; high p_increment → excessive wandering; high g_increment → premature convergence to local minima (Kennedy 1995 §3.3). This is the explore/exploit balance.

## Canonical code structure (pyswarms)
- Swarm holds: position (N,D), velocity (N,D), pbest_pos, pbest_cost, best_pos (D,), best_cost, options{c1,c2,w}.
- compute_pbest: mask where current_cost < pbest_cost; update.
- compute_velocity: cognitive = c1*U(0,1,(N,D))*(pbest_pos - position); social = c2*U*(best_pos - position); v = w*velocity + cognitive + social; optional clamp.
- compute_position: position += velocity; optional bounds.
- Star topology compute_gbest: best_pos = pbest_pos[argmin(pbest_cost)] if min < best_cost.
- Loop (GlobalBestPSO.optimize): for i in iters: cost=f(position); pbest=compute_pbest; best=compute_gbest; v=compute_velocity; x=compute_position. Return (best_cost, best_pos).
- generate_velocity: U(min,max) default (0,1). generate_swarm: uniform in bounds.
- gbest = star topology (every particle sees the single global best). lbest = ring topology (each sees neighborhood best) — slower spread, more robust to local minima.

## Benchmarks (pre-method, settings only)
Multimodal black-box test functions: Sphere (unimodal sanity), Rastrigin, Griewank, Ackley, Rosenbrock; search boxes per survey Table 2. Schaffer f6 (Kennedy 1995, very multimodal). NN weight training (XOR 2-3-1, 13 params; Fisher Iris). Metric: function value / error, iterations to criterion. No outcomes.

## Design-decision → why table
- POPULATION of candidates (vs single point): black box, no gradient, multimodal → parallel sampling escapes local minima; social sharing of best gives an evolutionary advantage (Wilson). 
- TWO attractors (pbest + gbest): collapse to one (midpoint) converges to a non-optimum → both stochastic kicks needed. pbest = individual memory (cognitive); gbest = collective memory (social).
- The difference vector (pbest−x), (gbest−x) (vs sign test): magnitude-proportional pull, smoother, easier to understand, better performance ("acceleration by distance").
- Independent per-dimension r1,r2 (random weights): the two stochastic kicks are necessary; randomness lets the swarm "overfly"/search between known-good regions.
- factor 2 / c1=c2=2: makes rand mean 1 so overshoot ~half the time; equal c1,c2 balances explore/exploit (high p→wander, high g→premature).
- INERTIA w (momentum): removing momentum makes PSO ineffective at global optima; w scales the carried velocity → big w explores, small w exploits; linear 0.9→0.4 anneals explore→exploit; replaces Vmax.
- Vmax / clamp / constriction: velocity can explode (positive feedback) → clamp |v| or use χ<1 (φ=c1+c2>4) to guarantee convergence.
- gbest star vs lbest ring: star = fast info spread, risk premature; ring = slower, more robust.
