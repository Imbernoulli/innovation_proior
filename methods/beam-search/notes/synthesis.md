# Beam search — synthesis

## Lineage (primary: Lowerre 1976 HARPY thesis summary, Stanford mirror)
- CMU speech decoding context (Reddy lab). Two predecessor systems:
  - **Hearsay-I** (Reddy et al. 1973): best-first search of syntactic paths *with backtracking*. Backtracking is costly in a large space.
  - **Dragon** (Baker 1975): Markov network, searches *all paths in parallel, no backtracking*, guarantees the globally optimal path in deterministic time — i.e. the Viterbi/forward dynamic program over the network. Slow (45–200× real time), because it updates every state's probability on every 10 ms frame.
- **HARPY** = combine the two: transition network (no a-priori probs), search **"best few" paths in parallel, no backtracking**, with segmentation. "Best" defined by a **heuristic threshold on the probabilities of the states being searched" → variable number of paths kept; many where confusable, few where obvious. Speedup 10–25× over Dragon (all-paths).
- Key quote (thesis summary p.15): searches only a few "best" paths in parallel; "Best" is defined by a heuristic threshold on the probabilities of the states being searched.
- Note: HARPY's original pruning was a **threshold beam** (keep states within a factor/threshold of the layer's best), not strictly fixed top-W. The fixed-width top-W formulation and the *name* "beam search" came from the Reddy/CMU writeups (~1977). Flag this nuance, don't overstate "top-W" as Lowerre's exact words.
- Locus model of search: prune all but a beam of near-miss alternatives around the best path at each segmental decision point — contains exponential growth without backtracking.

## Background ancestors (clean statements retrieved)
- **BFS**: FIFO frontier, explores all nodes at distance d before depth d+1. Complete; but frontier (and memory) grows exponentially with depth — O(b^d).
- **Best-first / A***: order frontier by f(n)=g(n)+h(n); g = cost so far, h = admissible (lower-bound) heuristic to goal. With admissible h, A* is complete and optimal. But it still may store exponentially many frontier nodes → memory blowup.
- **Branch and bound**: keep best-so-far incumbent; compute a bound on each subproblem; prune any node whose bound can't beat the incumbent. Sound (never prunes the optimum). But worst-case still exponential; BrFS variant can't use incumbent-pruning well, high memory.
- The common pain: all three are *exact/complete* and pay for it in memory/time that's exponential in the worst case. Beam search is the deliberate trade: bound the frontier breadth to a constant W, give up completeness & optimality, buy linear-in-depth memory and time.

## Modern explainers
- Wikipedia: beam search = modification of best-first that reduces memory; BFS to build the tree; at each level generate all successors, sort by heuristic, keep top β (beam width). W=∞ → best-first; W=1 → greedy/hill-climbing. Sacrifices completeness and optimality (a goal can be pruned).
- AIMA local beam search: keep k states, generate all successors of all k, keep best k. Differs from k random restarts: information passes between threads ("come join me in this good region"). Failure mode: all k collapse into one region → expensive hill-climbing. Fix: **stochastic beam search** — pick k successors at random w/ prob increasing in value.
- **Filtered beam search (Ow & Morton 1988)**: two-stage. Cheap *local/priority* evaluation filters the successors down to a **filter width**; then an expensive *global/total-cost* evaluation ranks survivors and keeps **beam width** β. Motivation: the accurate global evaluation is expensive, so don't run it on every successor.
- **Chokudai search (chokudai's blog 2017)**: beam search's diversity failure — top-W states in a layer tend to be tiny variations descending from one or two strong early states; even large W explores few genuinely distinct situations, gets stuck. Chokudai's fix: maintain **one priority queue per depth/turn**; one *pass* = for each depth in order, pop the single best un-expanded state from that depth's queue and push its successors into the next depth's queue; repeat passes until time runs out. Effect: equivalent to running width-1 beam search to the end, then again, then again — each pass injects a fresh distinct lineage, so after k passes you have ~k diverse decent solutions. Anytime (loop until clock); no beam-width tuning. Cost: must keep all states, more memory.

## Code plan
Self-contained python: beam search for **0/1 knapsack** (or single-machine weighted-tardiness?). Knapsack is cleanest: state = prefix of items + chosen set; layer = item index; value = total value so far; evaluation = value + optimistic bound (fractional completion) for ranking. Keep top-W per layer. Print best feasible value. Compare W=1 (greedy-ish) vs larger W and brute-force optimum to show beam approaches optimum as W grows. Must actually run.
