# LNS / ALNS synthesis (from retrieved sources)

## Sources read
- Shaw 1998 (CP'98): origin of LNS. Move = remove a set of customer visits, reinsert via CP branch-and-bound + Limited Discrepancy Search. Relatedness-based removal. Dynamic neighborhood size. Accept improving only (descent).
- Ropke & Pisinger 2006 (Transportation Sci, pre-print): ALNS. Multiple competing destroy/repair heuristics, roulette-wheel selection by adaptive weights, segment-based weight update, SA acceptance. Shaw/Random/Worst removal; greedy + regret-k insertion; noise.
- Schrimpf et al. 2000 (J Comp Phys): Ruin & Recreate. Independent same idea — large moves + SA/TA/Great-Deluge acceptance. Radial/Random/Sequential ruin + best insertion. Why: discontinuous / heavily-constrained landscapes where small moves get stuck or can't reach admissible neighbors.
- Pisinger & Ropke 2010 (Handbook of Metaheuristics): clean synthesis. VLSN framing (Ahuja et al.), neighborhood-search formalism, LNS = implicit neighborhood via destroy+repair, ALNS general framework (per-iteration EMA weight update).
- N-Wouda/ALNS python library: canonical code. RouletteWheel update ω = θ·ω + (1−θ)·s_j.

## Pain point / research question
Industrial VRP / scheduling. Classical local search uses small moves (2-opt, relocate, swap) — neighborhood O(n^2). On tightly-constrained problems (time windows, capacity, pickup-before-delivery) these get trapped in local optima and can barely move: most neighbors are infeasible, so the search can't walk from one admissible basin to another. Want a method that escapes local optima on constrained combinatorial problems and meshes with constraint propagation.

## Key derivation chain
1. Neighborhood-search formalism (handbook): X feasible solutions, c cost, N(x)⊆X. Steepest descent: x ← argmin_{x'∈N(x)} c(x'). Local optimum: c(x) ≤ c(x') ∀ x'∈N(x).
2. Bigger neighborhood ⇒ better local optima (fewer, deeper). 2-opt/relocate are O(n^2) = small. Want exponentially large N, but can't enumerate it.
3. VLSN (Ahuja et al.): neighborhoods that grow exponentially or are too large to search explicitly. Three classes: variable-depth (Lin-Kernighan), network-flow improvement, polynomially-searchable restrictions. LNS fits none cleanly but is VLSN.
4. The move: relax (destroy) part of the solution — remove q elements — then re-optimize (repair) over the relaxed part. One destroy+repair = one step into an implicitly-defined huge neighborhood. For CVRP n=100, q=15: C(100,15)=2.5e17 removal choices alone.
5. Why "large": with a big hole you have freedom to reach feasible reconfigurations a small move can't; with a small hole the best reinsertion is back where it came from (Shaw's independence argument) — wasted.
6. Removal choice: remove RELATED elements (Shaw relatedness eq), so reinsertion can actually swap them; remove unrelated ⇒ independent reinsertions ⇒ no gain.
7. Repair: optimal (CP branch-and-bound, LDS — Shaw) OR fast heuristic (greedy / regret-k — Ropke-Pisinger). Optimal repair only yields improving/equal moves ⇒ poor diversification; heuristic repair injects bad moves that diversify.
8. Acceptance: Shaw = descent (improving only). Schrimpf / Ropke-Pisinger = SA: accept worse with prob exp(−(c(x')−c(x))/T), T cooled geometrically T←c·T. Escapes local optima.
9. Adaptivity (ALNS): which destroy + which repair is best is instance- and phase-dependent. Maintain a weight per operator, pick by roulette wheel: P(j) = w_j / Σ w_i. Score recent success, update weights toward recent scores.

## EXACT formulas (verbatim from sources)
- Shaw relatedness (no side constraints, eq in Shaw §2.1): r(i,j) = 1 / ( c_ij + V_ij ), where c_ij normalized travel cost in [0,1], V_ij = 1 if same vehicle else 0. (Lower cost & same vehicle ⇒ MORE related ⇒ larger r.) NOTE the 2006 paper uses an inverse convention R(i,j) where LOWER R = more related.
- Shaw removal selection: pick a seed at random; repeatedly pick random already-removed r, rank remaining by relatedness, pick the one at index floor(rand^D · |L|) (Shaw uses D determinism, rand∈[0,1)); 5≤D≤15 reasonable. (2006: p≥1, index floor(y^p·|L|), low p = more random.)
- Shaw neighborhood-size control: start q=1; if α consecutive non-improving moves, q+=1; cap 30.
- Ropke-Pisinger relatedness R(i,j) (eq 17): R = φ(d_{A(i),A(j)}+d_{B(i),B(j)}) + χ(|T_{A(i)}−T_{A(j)}|+|T_{B(i)}−T_{B(j)}|) + ψ|l_i−l_j| + ω(1 − |K_i∩K_j|/min(|K_i|,|K_j|)). Lower R = more related.
- Worst removal: cost(i,s) = f(s) − f_{−i}(s); remove high-cost requests, randomized via index floor(y^p·|L|) over descending-cost list.
- Random removal = Shaw with p=1.
- Greedy insertion: Δf_{i,k} = best insertion cost of i into route k (∞ if infeasible); c_i = min_k Δf_{i,k}; insert argmin_i c_i. (regret-1)
- Regret-k: insert argmax_i ( Σ_{j=1..k} (Δf_{i,x_{ij}} − Δf_{i,x_{i1}}) ), where x_{ij} = route with j-th lowest cost. regret-2: c*_i = Δf_{i,x_{i2}} − Δf_{i,x_{i1}}.
- Roulette selection (eq 20): P(j) = w_j / Σ_{i=1}^k w_i. Insertion chosen independently of removal.
- Adaptive weight update (2006, segment-based): w_{i,j+1} = w_{ij}(1−r) + r·(π_i/θ_i). π_i = total score of heuristic i in last segment; θ_i = #times i was used in last segment; r = reaction factor ∈[0,1]. Segment = 100 iterations; scores reset to 0 each segment.
- Scores: σ1 if new global best; σ2 if accepted, unvisited, AND better than current; σ3 if accepted, unvisited, AND worse than current. (Only reward unvisited solutions, tracked via hash table.)
- Handbook per-iteration variant (eq 1,2): score ψ = max(ω1 new best, ω2 better than current, ω3 accepted, ω4 rejected), ω1≥ω2≥ω3≥ω4≥0. Update only the two USED operators: ρ_a ← λρ_a + (1−λ)ψ, ρ_b ← λρ_b + (1−λ)ψ. λ∈[0,1] decay. This is the form in the N-Wouda library: ω = θω + (1−θ)s_j.
- SA acceptance: accept x' if c(x')≤c(x); else accept with prob exp(−(c(x')−c(x))/T). T starts at T_start, T←c·T each iter, 0<c<1. T_start set so a w%-worse-than-initial solution is accepted with prob 0.5 ⇒ T_start = −w·c(x0) / ln(0.5) (with γ-term zeroed).
- Schrimpf ruin: Radial (seed + A−1 nearest neighbors, A≤⌈F·N⌉), Random (A random nodes), Sequential (A consecutive on one route). Recreate = best insertion. Threshold-accepting cooling T = T0·exp(−ln2·x/α), half-life α=0.1.

## Ancestors / baselines
- Steepest descent / local search with k-exchange (2-opt Croes/Lin; relocate, swap). Limitation: small neighborhood, traps in local optima, can't move under tight constraints.
- Lin-Kernighan: variable-depth — sequentially chains edge swaps, partial search of deep neighborhood. Ancestor of "go deeper to escape" idea but still edge-local and TSP-specific.
- Simulated Annealing (Kirkpatrick 1983) / Threshold Accepting (Dueck-Scheuer) / Great Deluge (Dueck): acceptance rules that take worse moves to escape local optima. LNS/ALNS borrow these as the accept step.
- VLSN (Ahuja et al. 2002): umbrella — exponentially large neighborhoods searched implicitly. LNS = a VLSN where the neighborhood is defined by destroy+repair.
- CP-based reinsertion (Shaw): branch-and-bound with propagation + LDS to do an (near-)optimal repair.

## Design decisions → why
- Large vs small move: small move's best reinsertion ≈ original position under tight constraints ⇒ stuck; large move opens feasible reconfigurations. Tradeoff: fewer moves/sec.
- Remove RELATED not random: unrelated removals reinsert independently (no interchange) ⇒ equivalent to several tiny moves ⇒ wasted cost (repair scales with q). Related removals enable swaps.
- Degree of destruction q: too small ⇒ lose large-neighborhood benefit; too large ⇒ degrades to re-solve-from-scratch (slow / heuristic-repair gives junk). Shaw: ramp q up on stagnation. RP: random q in a range.
- Heuristic vs optimal repair: optimal repair only produces improving/equal moves ⇒ trapped in valleys unless huge destroy; heuristic repair's "mistakes" diversify. Greedy postpones hard requests ⇒ regret-k look-ahead fixes that.
- SA acceptance vs descent: descent (Shaw) traps; SA accepts worse moves early to cross barriers.
- Multiple operators + adaptivity: no single removal/insertion is best across instances/phases; roulette by learned weights auto-tunes the mix; reward unvisited (diversifying) moves, not just improving ones.
- Reward unvisited only: prevents inflating weights for operators cycling back to already-seen solutions.
- Update only used operators (per-iter form); segment form averages π_i/θ_i so a rarely-used lucky operator isn't over-credited.
- Noise on insertion cost: myopic greedy always makes locally-best move; noise randomizes so SA's sampling explores.
