# Tabu Search

## Problem

Minimize c(x) over a discrete feasible set X (travelling salesman, graph coloring, scheduling, partitioning, 0/1 knapsack, quadratic assignment, …). A local-search heuristic walks from a trial solution to a neighbor via *moves* and keeps improving, but steepest descent halts at a *local optimum* that is generically far from global. Naively allowing the best non-improving move to escape only induces *cycling* — the search reverses the move it just made and snaps back. Goal: guide local search past local optima without cycling and without storing every solution visited.

## Key idea

Use **adaptive memory** instead of randomization. At each step take the **best admissible move** in the neighborhood — improving or not — so local optimality is never a barrier. To stop the immediate undo, keep a **tabu list**: forbid the *reversals* of the moves made in the last *t* iterations (the **tabu tenure**). This makes "escape a basin" and "don't fall back into it" the same mechanism. Forbidding the reverse of the last *t* moves pushes the search away from the recent solution states, so short cycles cannot directly re-close over the *t*-step horizon (a likelihood effect, not a hard no-revisit guarantee — exact guarantees need more elaborate dynamic list management). The tenure should be the smallest value that suppresses cycling: tenure beyond that only removes the freedom to chase good moves. Early applications (covering, scheduling, partitioning) found a small fixed list of about 7 (roughly 5–12) works remarkably well; larger neighborhoods such as 2-opt TSP use larger, problem-size-dependent list lengths.

For efficiency the list stores cheap **move attributes** (e.g., an edge of a 2-opt move, a (set, weight) tag) rather than whole moves; a move is tabu if it touches a tabu attribute. Attributes over-forbid, so an **aspiration criterion** overrides the tabu status of a move that is good enough — canonically, accept a tabu move if it beats the best solution found so far. This is safe against cycling because a record-beating solution has never been visited.

Two longer-horizon memories complete the picture. **Short-term (recency) memory** is the tabu list. **Intermediate-term memory** *intensifies*: record features common to the best solutions seen and bias moves to keep them. **Long-term (frequency) memory** *diversifies*: count how often each attribute appears over the whole run and penalize over-used ones to push into unexplored regions. Recency forbids the immediate past to escape now; frequency penalizes the cumulative past to explore later.

## Algorithm

Given problem (P): minimize c(x), x in X, neighborhood S(x), move inverses s⁻¹.

1. Choose initial x in X; set incumbent x* := x, k := 0, tabu list T empty.
2. Build the candidate set S(x). For each candidate move s, mark it admissible if it is **not tabu**, OR it is tabu but satisfies the **aspiration** criterion (e.g., c(s(x)) < c(x*)).
3. If no admissible move exists, stop. Otherwise pick the admissible s minimizing c(s(x)) (best move, improving or not). Set x := s(x), k := k+1.
4. Record the attributes of s as tabu for tenure t (sliding/circular list: add s's reversal-blocking attributes, drop those older than t). If c(x) < c(x*), update x* := x.
5. (Optional) Update frequency counts for diversification; periodically intensify on shared features of best solutions, or diversify by penalizing high-frequency attributes.
6. If the iteration budget (total, or since x* last improved) is exhausted, stop; else return to step 2.

Return x*.

## Code

```python
import random, math

def tour_len(tour, D):
    n = len(tour)
    return sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n))

def two_opt_apply(tour, i, j):
    # neighbor tour: reverse the segment at positions i+1..j
    return tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]

def removed_edges(tour, i, j):
    # the two undirected edges this 2-opt deletes (broken at the segment ends)
    n = len(tour)
    return frozenset((tour[i], tour[i + 1])), frozenset((tour[j], tour[(j + 1) % n]))

def added_edges(tour, i, j):
    # the two undirected edges this 2-opt creates; tabu-ing these blocks the reverse
    n = len(tour)
    return frozenset((tour[i], tour[j])), frozenset((tour[i + 1], tour[(j + 1) % n]))

def tabu_search(D, n, tenure=7, iters=400, seed=0):
    random.seed(seed)
    cur = list(range(n)); random.shuffle(cur)          # any starting solution
    best, best_cost = cur[:], tour_len(cur, D)         # incumbent x*
    tabu = {}                       # an edge -> iteration until which deleting it is forbidden
    history = [best_cost]
    for k in range(1, iters + 1):
        chosen, chosen_cost, chosen_add = None, math.inf, None
        for i in range(n - 1):
            for j in range(i + 2, n):                   # j>=i+2: skip degenerate no-op moves
                if i == 0 and j == n - 1:
                    continue
                cand = two_opt_apply(cur, i, j)
                cost = tour_len(cand, D)
                r1, r2 = removed_edges(cur, i, j)       # this candidate deletes r1, r2
                is_tabu = tabu.get(r1, 0) >= k or tabu.get(r2, 0) >= k  # deleting a recently-added edge?
                aspire = cost < best_cost               # aspiration: never-seen & better
                if is_tabu and not aspire:
                    continue                            # forbidden, and not good enough
                if cost < chosen_cost:                  # best ADMISSIBLE move
                    chosen, chosen_cost = cand, cost
                    chosen_add = added_edges(cur, i, j)
        if chosen is None:                              # all moves forbidden (demo: stop;
            break                                       #  a full version ages out oldest tabus)
        cur = chosen                                    # take it, improving or not
        for e in chosen_add:                            # forbid deleting the just-added edges
            tabu[e] = k + tenure
        if chosen_cost < best_cost:                     # update incumbent x*
            best, best_cost = cur[:], chosen_cost
        history.append(best_cost)
    return best, best_cost, history

if __name__ == "__main__":
    random.seed(1)
    n = 15
    pts = [(random.random(), random.random()) for _ in range(n)]
    D = [[math.dist(pts[a], pts[b]) for b in range(n)] for a in range(n)]
    tour, cost, hist = tabu_search(D, n, tenure=7, iters=300, seed=0)
    print("best cost:", round(cost, 4))
    for k in range(0, len(hist), max(1, len(hist) // 10)):
        print(f"  iter {k:3d}  best {hist[k]:.4f}")
```

Running it prints the best objective decreasing over iterations as the search repeatedly escapes local optima and settles on a short tour. Swapping the neighborhood, the attribute definition, and the aspiration rule retargets the same skeleton to any combinatorial problem expressible as min c(x) over moves.
