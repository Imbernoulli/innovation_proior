A very large class of practical problems is just "minimize $c(x)$ over $x \in X$" where $X$ pins specified components of $x$ to discrete values — travelling salesman, graph coloring, job-shop sequencing, set partitioning, quadratic assignment, multidimensional knapsack, employee scheduling. The interesting instances are NP-hard, and exact methods like cutting planes and branch and bound, while provably correct, are rigid enough that on real-size combinatorial structure they exhaust memory or run past any usable time budget. The pragmatic handle is local search: represent a procedure by its *moves*, where a move $s$ takes one trial solution to a neighbor and the moves applicable at $x$ form the neighborhood $S(x)$. Steepest descent iterates the best improving move — $x := s(x)$ for $s$ with $c(s(x)) < c(x)$ — until none exists, and there it stops. That stopping point is a *local optimum*, and in any combinatorial landscape worth the name it is generically not the global one. The result is hostage to wherever the search happened to start and to the shape of the neighborhood, with nothing to do once improving moves run out. Multi-start descent — a hundred random restarts, keep the best — only throws away everything each run learned, with no transfer between basins and coverage left to luck that is exponentially thin when $X$ is huge. The obvious fix of taking the *least-disimproving* move when descent stalls (Hansen's steepest-ascent/mildest-descent) escapes the basin but then *cycles*: from the neighbor $y$ just reached, the cheapest move is the one straight back to the local optimum $x^*$, and the walk oscillates in a 2-cycle forever. Simulated annealing escapes basins by accepting a worsening move of size $\Delta$ with probability $\exp(-\Delta/\mathrm{Temp})$ under a cooling schedule, but it buys its diversity purely by *forgetting* the past — keeping no record of where it has been, it cannot avoid revisiting solutions, cannot steer toward features common to good solutions, and cannot deliberately diversify away from over-used ones. So escaping local optima and avoiding cycling are not two problems but one: any rule loose enough to leave a basin is, by the same looseness, happy to walk straight back in.

The method I propose is Tabu Search, and its premise is that a search should be guided by *adaptive memory* rather than by randomization. Keep the aggressive primitive — at every step take the *best admissible move* in $S(x)$, improving or not — so that local optimality is never a barrier the search has to detect or react to; the rule "take the best admissible move" is well-defined whether or not an improving move exists, and the search simply passes through local optima. What disciplines this so it does not cycle is the *tabu list*. The trap that closes a cycle is precisely the *reverse* of a move just made: if I applied $s$, the danger is $s^{-1}$ with $s^{-1}(s(x)) = x$. So I forbid reversals. Forbidding only the last reverse is not enough, because a slightly bigger loop — $x^0 \to x^1 \to x^2$ and back to $x^0$ along a different move — sneaks around a single blocked edge. The fix scales: keep forbidden the reverses of the last $t$ moves, where $t$ is the *tabu tenure*. Formally the tabu set is the sliding window
$$T = \{\, s^{-1} : s = s_h,\ h > k - t \,\},$$
with $k$ the current iteration; maintaining it is a circular list of length $t$, adding the reverse of the move just made and dropping the one now $t$ steps old. The selection rule becomes "best move in $S(x)$ not in $T$," and this is what fuses escape and anti-cycling into a single mechanism — over a $t$-step horizon no move may directly undo any of its $t$ predecessors, so short loops cannot re-close. This is a likelihood effect, not a hard no-revisit guarantee; a longer loop could still sneak around the forbidden reverses under the cheap proxy below, and an ironclad guarantee would need more elaborate dynamic list management. I should stress that the tabu list prevents *reversal*, not *repetition*: repetition-prevention ("don't make the same move $s$ again") would still permit applying $s^{-1}$ immediately after $s$ and sliding right back, which is exactly the 2-cycle I am killing — empirically those lists work poorly, and the reason is that they fail to stop the immediate undo.

The tenure $t$ carries a real tension that I resolve deliberately rather than by defaulting to "make it large." Push $t$ small and I forbid almost nothing, keeping the widest latitude to chase genuinely good moves, but I barely suppress cycling — a loop a hair longer than $t$ slips through. Push $t$ large and I crush cycling, but I have forbidden a big chunk of $S(x)$ on every step, starving the search of good moves and forcing long detours of mediocre ones, sometimes into regions chosen only because everything attractive is tabu. So the goal is the *smallest* $t$ that reliably prevents cycling; every unit beyond that is latitude given up. Short cycles need a tenure of a handful of steps, not hundreds, and there is an independent hint about that order of magnitude — Miller's finding that simple recency memory usefully holds about seven chunks, give or take two. On simpler problems (covering, scheduling, partitioning, where each move toggles one element) a fixed tenure around seven, in the 5–12 band, is both anti-cyclic and unconstraining. But the right tenure scales with how the move attributes are built: a coarse attribute like one edge of a 2-opt swap blocks a wider class per unit of tenure than a single named variable flip, so a richer structure such as 2-opt over a large tour wants a longer, problem-size-dependent list. The principle is "smallest $t$ that stops cycling," not the literal value.

A practicality reshapes the design: I do not need the full identity of a move to recognize its reverse. Storing and comparing whole tours for the last $t$ steps across an $O(n^2)$ neighborhood every iteration is heavy, so instead of tabu *moves* I keep tabu *attributes* — for a 2-opt move, an edge it touches — and a move is tabu if it touches a currently-tabu attribute. One recorded attribute is cheap to store, cheap to look up in a small table, and can forbid a whole class of moves at once. That cheapness is also a liability: attribute-based tabu *over-forbids*. Making "don't re-add edge $e$" tabu blocks *every* move that re-adds $e$, including moves that lead somewhere never visited and possibly excellent — the attribute is a coarse proxy for "you're undoing recent history," and it sometimes flags moves that undo nothing. The remedy is an *aspiration criterion*: override a move's tabu status when it is, on the merits, too good to pass up. The cleanest form falls straight out of the worry — if a tabu move would produce a solution better than the best ever found, accept it regardless. This never reintroduces cycling, the only thing tabu was protecting, because a solution strictly better than the all-time incumbent $x^*$ is by definition one never visited, so there is no loop to fear: override tabu when $c(s(x)) < c(x^*)$. Tabu restriction and aspiration are mirror images of one admissibility test — a move is admissible if the tabu condition does not apply, *or* if the aspiration condition does. (A finer general form keys a record $\mathrm{BEST}(q)$ on the objective value $q$ of the solution being left, recording the best transition previously made out of the $q$-level and firing aspiration when a tabu move beats it — detecting transitions new relative to their own objective class — but "better than the incumbent best" is the blunt special case and is what the core method uses.)

Recency memory is only the shortest-horizon use of the underlying idea, that memory, not randomness, is the engine. Two longer-horizon memories complete the picture and pull in opposite directions. Intermediate-term memory *intensifies*: good solutions share features — edges that keep appearing in short tours, assignments that recur — so recording the features common to a pool of the best solutions and biasing moves to keep them concentrates effort in a promising region (with a brutal free version for large problems: discard move-components that never appear in good solutions, shrink the problem, and run far more iterations per second). Long-term *frequency* memory *diversifies*: count over the whole run how often each attribute appears, and *penalize* the frequently-used ones by adding a tax term to the move evaluation, pushing the search to assemble solutions from neglected pieces and into unexplored territory — a purposeful restart reasoned from the search's own history, the opposite of annealing's restart-by-randomness. Recency forbids the immediate past to escape now; frequency penalizes the cumulative past to explore later.

Concretely, for TSP with 2-opt: a move reverses the segment between positions $i+1$ and $j$, deleting the two edges at the segment's ends and adding two new ones. The reverse move I must block would *delete those two newly-added edges* to restore the old ones, so the tabu attributes are the *added* edges, recorded as "do not delete these for the next $t$ iterations" — any later candidate that would delete a still-tabu edge is exactly an attempted undo. This is the easy place to slip: tabu-ing the *deleted* edges would do nothing, because the reverse does not delete them, it re-adds them. One index guard takes $j \ge i+2$, since $j = i+1$ reverses a one-element segment and gives the same tour back, a degenerate no-op that should not compete with real moves. The procedure is steepest descent with two changes: it never stops at a local optimum (it takes the best *admissible* move, worsening if necessary), and it carries the tabu memory that forbids recent reversals under the aspiration override.

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
