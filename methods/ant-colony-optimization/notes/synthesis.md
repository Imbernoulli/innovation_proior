# Synthesis — Ant Colony Optimization (Ant System / ACS)

## Pain point / research question
- TSP: find min-length closed tour over n cities. NP-hard. Need a general-purpose constructive metaheuristic, not a TSP-specialist (so it ports to ATSP, QAP, JSP).
- Existing tools: exact (branch&bound) blows up; tailored heuristics (2-opt, Lin-Kernighan, nearest-neighbor + insertion) are TSP-specific; general metaheuristics SA, TS work but are trajectory-based (one solution at a time) and TSP-shaped only via neighborhood moves.
- Want: a population-based constructive search where many simple agents cooperate via a shared, slowly-changing memory; positive feedback to find good solutions fast; a mechanism to avoid premature lock-in.

## Background facts (pre-method, sourced)
- **Real-ant foraging / stigmergy.** Deneubourg double-bridge (Deneubourg et al. 1990, "Self-organizing exploratory pattern of the Argentine ant", J. Insect Behavior 3(2):159): ants deposit pheromone, follow it probabilistically; on unequal branches they converge to the SHORT branch via differential path length + autocatalysis. Mechanism = stigmergy (Grassé): indirect communication through modifications of the environment. The first ant to reach food on the short branch returns first, so the short branch accrues pheromone FASTER -> positive feedback -> all ants converge to it.
- **Autocatalysis / positive feedback.** A self-reinforcing process: more pheromone -> higher choice probability -> more pheromone. Unbounded -> explosion, so a limiting mechanism (evaporation) is needed.
- **TSP & constructive heuristics.** Nearest-neighbor: greedily go to closest unvisited city; cheap but myopic — early greedy choices force terrible late edges (closing the tour costs a lot). η_ij = 1/d_ij is the natural "visibility"/greedy force.
- **SA, TS** as the general-purpose comparators (trajectory metaheuristics).

## Primary derivation — Ant System (Dorigo, Maniezzo, Colorni 1996)
- Graph (N,E). m ants. τ_ij(t) = trail on edge. η_ij = 1/d_ij = visibility (static).
- Each ant builds a tour using tabu list (forbid revisiting). Transition probability (Eq 4):
  p^k_ij(t) = [τ_ij]^α [η_ij]^β / Σ_{l∈allowed_k} [τ_il]^α [η_il]^β  for j ∈ allowed_k, else 0.
  - α = importance of trail; β = importance of visibility. α=0 -> stochastic multi-greedy (visibility only); β=0 -> only pheromone.
- After all m ants finish a cycle (n iterations), update trail (Eq 1):
  τ_ij(t+n) = ρ·τ_ij(t) + Δτ_ij,  Δτ_ij = Σ_k Δτ^k_ij  (Eq 2)
  - ρ = persistence, (1−ρ) = evaporation, 0≤ρ<1.
  - **ant-cycle** (Eq 3): Δτ^k_ij = Q/L_k if ant k used edge (i,j), else 0. (GLOBAL info: deposit ∝ 1/tour-length.)
  - **ant-density** (Eq 5): Δτ^k_ij = Q if ant goes i->j (per step). (local, distance-independent)
  - **ant-quantity** (Eq 6): Δτ^k_ij = Q/d_ij per step. (local, shorter edges favored)
  - ant-cycle wins because it uses GLOBAL solution quality. ant-density/quantity use only local info and do worse.
- τ_ij(0) = c (small positive constant).
- Complexity O(NC·n²·m); with m∝n, O(NC·n³).
- Default params: α=1, β=1, ρ=0.5, Q=100. Best for ant-cycle: α=1, β=5, ρ=0.5. m=n.
- **Stagnation**: all ants follow same tour; happens for high α (trail dominates, premature lock-in). Low α -> too random, no good solutions. Sweet spot in the middle (α≈1, β∈{1..5}). This is the explore/exploit dial.
- **Elitist strategy**: add e·Q/L* to best-so-far tour edges (e elitist ants). Optimal range of e; too many -> premature exploitation of suboptimal.
- **Why it works (Sec VIII)**: a lone greedy ant (α=0) makes good early edges but bad forced late edges (tour-closure constraint). With many ants, good sub-paths get traversed by many ants -> much trail; bad forced edges traversed by few -> little trail. Superimposition of effects extracts good components. Evaporation lets the system forget early greedy bias and exploit accumulated global info.

## ACS (Dorigo & Gambardella 1997) — the efficiency refinements
Three changes vs AS:
1. **Pseudo-random-proportional rule (Eq 3)**: sample q∈[0,1]; if q≤q0, EXPLOIT: s = argmax_u [τ(r,u)][η(r,u)]^β; else biased exploration via Eq(1) (with α implicitly 1). q0 dials exploit vs explore directly.
2. **Global update only on best-so-far tour (Eq 4)**: τ(r,s) ← (1−α)τ(r,s) + α·Δτ(r,s), Δτ=1/L_gb on best edges. Concentrates search near best tour.
3. **Local update (Eq 5)**, applied as each ant traverses an edge: τ(r,s) ← (1−ρ)τ(r,s) + ρ·τ0. Effect: just-used edges lose pheromone -> next ants in same iteration avoid them -> ants don't all collapse onto one path (diversity within an iteration). τ0=(n·Lnn)^−1, Lnn = nearest-neighbor tour length.
- Params: β=2, q0=0.9, α=ρ=0.1, m=10, τ0=(n·Lnn)^−1.

## Code (grounded in ppoffice/ant-colony-tsp aco.py)
- Graph(cost_matrix, rank): pheromone init = 1/(rank*rank).
- ACO(ant_count, generations, alpha, beta, rho, q, strategy): strategy 0=ant-cycle, 1=ant-quality, 2=ant-density.
- _Ant: tabu list, allowed list, eta=1/d. _select_next: roulette over [τ]^α[η]^β. _update_pheromone_delta: cycle Q/total_cost, quality Q, density Q/d_ij.
- ACO._update_pheromone: τ *= rho (persistence), then += Σ ant deltas. solve: per gen build ants, each builds tour, track best, update.
- NOTE: code's `rho` = persistence (multiplier), matches AS τ←ρτ+Δτ. Keep this naming.

## Design-decision → why
- Why η=1/d (visibility)? greedy force: prefer near cities; gives decent tours before any pheromone exists (cold start).
- Why multiply [τ]^α·[η]^β (not add)? product makes both factors necessary (an edge needs both pheromone AND closeness); exponents independently tune each. Additive would let one swamp the other and lose the AND.
- Why deposit ∝ 1/L (ant-cycle) not constant? reinforcement ∝ solution quality -> short tours bias the trail more -> selection pressure toward good solutions. Constant/per-step (density/quantity) ignores global quality, does worse.
- Why evaporation (1−ρ)? without it trail accumulates unbounded (autocatalysis explodes) and old/bad edges keep their early pheromone forever -> lock-in; evaporation forgets bad early choices and bounds trail.
- Why ρ≈0.5? balance: forget enough to escape early greedy bias but retain enough accumulated global info.
- Why α≈1, not large? large α -> trail dominates -> stagnation (all ants same tour); small α -> ignores learned info, ~random. Middle = explore/exploit balance.
- Why m=n? experimentally linear relation; enough ants to cover, not so many it's wasteful.
- ACS: why q0 rule? gives a direct exploit/explore knob, sharper than tuning α.
- ACS: why best-only global update? focus pheromone reinforcement near the best tour -> faster convergence on large problems (AS reinforces every tour incl. mediocre ones).
- ACS: why local update lowers pheromone? counteracts the best-only update's tendency to collapse all ants onto the best tour; makes traversed edges less attractive within an iteration so ants diversify.
- ACS: why τ0=(n·Lnn)^−1? sets initial pheromone near the level a good tour would deposit, so local update can only lower it toward a meaningful floor; ties the scale to the problem.
