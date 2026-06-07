# Tabu Search — synthesis

## Pain point
Combinatorial optimization min c(x), x in X (discrete). Hill-climbing/steepest-descent: from x, pick s in S(x) (neighborhood) that improves c; stop when no improving move. Chief limitation: stops at a LOCAL optimum that may be far from global. Also: if you allow non-improving moves to escape, plain descent will immediately walk back downhill into the same local optimum → cycling (2-cycle between local opt and its best neighbor). So the two coupled failure modes of descent: (1) entrapment at local optima, (2) cycling once you try to leave.

## Core idea (Glover 1986, 1989/1990)
- Tabu search = metaheuristic superimposed on a base heuristic (e.g., 2-opt local search). At each step pick the BEST available move (not just improving) from S(x) minus tabu set T. Even when c increases (least disimprovement). This lets it walk out of a local optimum.
- To stop it from cycling straight back, FORBID recently made moves' reversals: T = {s^{-1} : s = s_h for h > k - t}. The reverse of any of the last t moves is "tabu". t = tabu tenure. Update at each step: T := T - s_{k-t} + s_k^{-1} (circular list).
- Rationale for tabu (recency): likelihood of cycling back to a past solution is inversely related to distance (in #moves) from it; forbidding reversal of last t moves drives progressively away from the last t solution states; for t large enough, return likelihood ~ vanishes.
- Best-move (aggressive) not slow-descent: contrast with simulated annealing's Metropolis cooling. Two reasons: (1) many problems solved optimally by best-available move; (2) local optimality is no barrier procedurally, so spend effort in good regions.
- t tradeoff: smaller t = more latitude/freedom of choice; larger t = stronger anti-cycling. Empirically m≈7 (5–9, up to 12) works remarkably well across applications; "highly stable range." 1986 paper notes coincidence with Miller's magical-number-7 short-term-memory chunk count.
- ATTRIBUTE-based tabu (not full-solution): record partial attributes of a move (e.g., one edge of a 2-opt swap, or (set, weight) pairs), implemented with TABULIST (circular) + TABUSTATE array. Cheaper memory; one attribute can make a whole class tabu. Implication: tabu by attribute may forbid moves that don't actually revisit a solution → too restrictive → need aspiration.

## Aspiration criteria
- Tabu status can over-restrict: it forbids a move because its attributes match a recent move, even if that move would lead to a NEW, better solution. Fix: aspiration criterion overrides tabu when a move is "good enough."
- Simplest, most common: aspiration by objective — if a tabu move yields c(s(x)) < c(x*) (better than the best found so far), accept it anyway (it can't cause a cycle because that solution has never been seen). This is the dominant practical form.
- More refined: A(s,x) aspiration-level function, BEST(q) tables keyed on c(x) or on move attributes; A(z) aspiration list in 1986 (A(z) initially z-1; updated A(z') = min(A(z'), z''-1) so any move that improves z to A(z) achieves a not-previously-seen transition → safe to override). Tabu restriction and aspiration are dual: tabu = admissible if condition does NOT apply; aspiration = admissible if condition DOES apply. Integrated framework with per-attribute tabu lists T_p of different tenures t_p and per-attribute aspiration A(e).
- Three solution-specific anti-cycling levels: prevent move from x to s(x) if (1) s(x) visited before (only full guarantee, expensive); (2) s applied to x before (repetition — works poorly, can reverse immediately); (3) s^{-1} applied to s(x) before (reversal — compatible with the tabu-list structure). Reversal-prevention (3) is what tabu lists do; repetition-prevention (2) noted as NOT competitive.

## Short-term vs long-term memory; intensification/diversification
- Short-term (recency) memory = the tabu list above. Prevents cycling, escapes local optima.
- Intermediate-term memory (intensification): record features common to a set of best solutions; restrict/penalize moves to favor those features → focus search on good regions. E.g., in TSP discard edges never used in good tours, shrink the problem.
- Long-term (frequency) memory (diversification): count how often each attribute (e.g., each edge) appears across all solutions generated; PENALIZE frequently-used attributes to push the search into unexplored territory — purposeful (not random) restart/new region. "Frequency-based tabu criteria in contrast to recency-based." Roughly the reverse of intensification.
- Strategic oscillation: drive search across a boundary (feasibility, or a fixed parameter p) to alternating depths, recording best at each crossing.
- Probabilistic tabu search: accept moves by probability from (S1) attractiveness, (S2) tabu status (shorter residence → lower accept prob), (S3) aspiration. Optional; ensures prob-1 optimality in the limit under mild conditions.

## Part II advanced (finiteness)
- Dynamic tabu list management: C-sequence (successive cancellation) and Reverse Elimination Method (REM) — exact tracking of which attributes must be tabu to guarantee no previously-visited solution is revisited except along a new trajectory; bears on finiteness. Also staged search, structured move sets. MIP applications, neural-net guidance. (Advanced; trace mentions the finiteness motivation and the idea of a growing t / priority drop, not the full REM machinery.)

## Code grounding
Reference impl (olexiim/tsp tabu.py): loop = get_neighbors(state); among neighbors not in tabu pick min cost; move; push onto tabu list with a size limit (circular). That repo uses full-solution tabu. The canonical Glover form for TSP uses 2-opt move attributes (edges) + aspiration-by-objective. My demo: TSP, 2-opt neighborhood, tabu list on the EDGE-PAIR / move attributes with tenure t, aspiration = override if better than global best, track best objective over iterations. Print best objective trajectory. Verify runs.

## In-frame discipline
Do NOT cite the 1986/1989/1990 papers as published artifacts. Name "tabu search" as the thing being built (answer.md). Cite ancestors: hill climbing / steepest descent local search (Lin–Kernighan 2-opt lineage, Lin 1965), simulated annealing (Kirkpatrick et al. 1983, Metropolis 1953) as a contrasting contemporary, Miller 1956 magical-number-7. Hansen steepest-ascent/mildest-descent as a near-ancestor of "best non-improving move". Application result numbers are proposed-method results → excluded.
