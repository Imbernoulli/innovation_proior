## Research question

A great many practical problems reduce to minimizing a cost over a discrete set:

    minimize c(x) subject to x in X,

where X constrains specified components of x to discrete values and c may be linear or nonlinear. Travelling salesman, graph coloring, job-shop sequencing, set partitioning, the quadratic assignment problem, multidimensional knapsack, employee scheduling — all sit here, and the interesting instances are NP-hard. Exact methods (cutting planes, branch and bound) are provably correct but rigidly structured, and that rigidity caps their reach: on combinatorial structure of real size they either exhaust memory or run past any usable time budget. The pragmatic alternative is a heuristic that walks from one trial solution to a neighbor and keeps the best it sees. The question this raises: a neighborhood walk that only ever steps downhill halts at the first solution with no better neighbor — a *local* optimum that can be arbitrarily far from the global one. How can a local-search heuristic be guided to keep exploring past that point?

## Background

**Local search / hill climbing.** Represent a procedure for (P) by its *moves*. A move s is a mapping on a subset X(s) of X taking one trial solution to another; the moves applicable at x form the neighborhood S(x) = {s in S : x in X(s)}. Steepest descent ("hill climbing", with the hill inverted for minimization) iterates: from x, choose s in S(x) with c(s(x)) < c(x), set x := s(x), and stop when no improving move exists. Many elegant algorithms are instances — the simplex method on the extreme points of an LP, for example. The construction is general: change the definition of a move and the same skeleton attacks a new problem.

**Common move structures.** For 0/1 problems, the moves between adjacent vertices of the unit hypercube, s(x) = x ± e_j (flip one variable). For the travelling salesman problem, the **2-opt** move (Lin, 1965; Lin & Kernighan, 1973): delete two non-adjacent edges of a tour and add back the unique two edges that reconnect it, reversing the intervening segment. For mixed-integer programs, a composite move that increments/decrements an integer variable and then re-solves the continuous part by linear programming. For partitioning, a swap exchanging an element of one set with an element of another. Each move has an *inverse* s⁻¹ with s⁻¹(s(x)) = x.

**Steepest-ascent / mildest-descent.** Hansen (1986) studied the "best non-improving move" idea for combinatorial programming: at a local optimum, take the move that increases the objective the least (mildest ascent in a maximization framing) to climb out, rather than restarting.

**Simulated annealing.** The prominent contemporary escape mechanism (Kirkpatrick, Gelatt & Vecchi, 1983; on Metropolis et al., 1953). It accepts a worsening move of size Δ with probability exp(−Δ/Temp), Temp lowered on a cooling schedule. All *improving* moves carry the same status (any one met is accepted); worsening moves start with high acceptance when Temp is large and are progressively extinguished. The premise is a *slow*, randomized descent: with the right schedule the local optimum eventually reached is, with high probability, a good one.

**Adaptive memory.** The contrasting idea is that a search ought to *learn from its own trajectory*: record salient features of recent and past moves and let that record shape which moves are admissible now. The open design question is what to record, for how long, and how the record should constrain choice.

**Magical number seven.** Miller (1956) found human short-term memory holds about seven (plus or minus two) "chunks." A pre-existing fact about how much recent history a simple memory mechanism can usefully carry — a suggestive benchmark for how long a forbidding-record might need to be.

## Baselines

**Steepest descent / hill climbing.** Core idea: iterate improving moves to a local optimum. Math: x := s(x) for s in S(x) with c(s(x)) < c(x); stop when none exists.

**Multi-start descent.** Run descent from many random starts, keep the best.

**Steepest-ascent/mildest-descent (Hansen, 1986).** Core idea: when stuck, take the best (mildest-worsening) move to leave the local optimum instead of restarting.

**Simulated annealing (Kirkpatrick et al., 1983).** Core idea: accept worsening moves probabilistically under a cooling schedule to tunnel out of basins; under a sufficiently slow cooling schedule and standard finite-state ergodicity conditions it converges to the global optimum in the infinite-time limit.

## Evaluation settings

Natural yardsticks are the standard combinatorial benchmarks: travelling-salesman instances with known or best-known tour lengths (cities in the tens to low hundreds), graph coloring, job-shop / flow sequencing, multidimensional 0/1 knapsack and capital-budgeting instances, set-partitioning and clustering problems, and quadratic assignment instances. The metric is solution quality (objective value, or gap to a known optimum/best-known) attained within a fixed computational budget, compared across heuristics; a secondary axis is robustness to the random starting solution and to neighborhood/parameter choices.

## Code framework

The primitives that already exist: a problem with a cost function and a neighborhood generator, plus a base steepest-descent loop. The contribution will fill the marked slot.

```python
import math, random

def tour_len(tour, D):
    n = len(tour)
    return sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n))

def two_opt_apply(tour, i, j):
    # reverse the segment between i+1 and j inclusive -> a neighbor tour
    return tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]

def neighbors(tour):
    # yield (neighbor, the move that produced it) over the 2-opt neighborhood
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if i == 0 and j == n - 1:
                continue
            yield two_opt_apply(tour, i, j), (i, j)

def steepest_descent(D, n, seed=0):
    random.seed(seed)
    cur = list(range(n)); random.shuffle(cur)
    cur_cost = tour_len(cur, D)
    while True:
        best, best_cost = None, cur_cost
        for cand, _move in neighbors(cur):
            c = tour_len(cand, D)
            if c < best_cost:
                best, best_cost = cand, c
        if best is None:          # local optimum: no improving neighbor
            return cur, cur_cost   # <-- and here the search simply stops
        cur, cur_cost = best, best_cost

def search(D, n, iters, seed=0):
    # TODO: like steepest_descent, but do not stop at a local optimum;
    #       keep the best solution seen over a budget of `iters` steps.
    pass
```
