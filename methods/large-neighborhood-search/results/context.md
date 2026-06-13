# Context: escaping local optima on tightly-constrained routing and scheduling

## Research question

Industrial vehicle-routing and scheduling problems — deliver goods to hundreds of customers with a fleet of vehicles, respecting capacities, time windows, pickup-before-delivery precedence, driver rules — are solved routinely and at scale. Exact methods do not scale to the sizes that arise in practice, so the working tool is local search wrapped in a metaheuristic. The concrete pain point: classical local search makes *small* moves (swap two customers, relocate one, reconnect two edges), and on tightly-constrained instances those small moves get trapped. Once the search reaches a locally optimal routing plan, almost every small perturbation is either worse or outright infeasible — moving a single time-windowed customer somewhere else usually violates a window or a capacity bound — so the search cannot walk from one feasible basin of the solution space to another. The goal is a search step that can cross those barriers — reaching feasible plans that the current small-move repertoire cannot — while still always producing a feasible plan and still meshing with the constraint machinery that real problems demand.

## Background

A combinatorial optimization instance is a finite set of feasible solutions `X` with a cost `c: X → ℝ`; the task is `x* = argmin_{x∈X} c(x)`. Neighborhood search equips this with a neighborhood function `N: X → 2^X`. A solution `x` is **locally optimal** with respect to `N` when `c(x) ≤ c(x')` for every `x' ∈ N(x)`. Steepest descent repeatedly replaces `x` by `argmin_{x'∈N(x)} c(x')` until a local optimum is reached. The quality of what you get is governed by the neighborhood: a larger `N` has fewer, deeper local optima, so searching a large neighborhood tends to return better solutions — but it costs more per step.

The neighborhoods in standard use are small. For the TSP the **2-opt** neighborhood (deleting two edges and reconnecting the tour the other way) has size `O(n²)`; for routing the **relocate** neighborhood (move one customer to another position or route) and **swap** are likewise `O(n²)`. These can be searched quickly, which is why they dominate in practice, but `O(n²)` is tiny relative to the solution space, and on heavily-constrained instances the reachable feasible part of such a neighborhood shrinks toward nothing.

It is well documented that on "discontinuous" landscapes — where adjacent solutions can have wildly different cost, and where most neighbors of a feasible schedule are *infeasible* — classical small-move local search and the Monte-Carlo acceptance methods built on it (simulated annealing, threshold accepting, the great-deluge algorithm) struggle: they either stall in a local optimum or get stuck in "slightly infeasible" states maintained only by penalty terms. The observed behavior that motivates a different move operator is exactly this: the harder and more constrained the problem, the more a single-element move is forced back to (almost) where it started, and the search freezes.

A separate strand widened the lens on neighborhood size. **Very Large-Scale Neighborhood (VLSN) search** (Ahuja, Ergun, Orlin, Punnen, 2002) names the class of algorithms whose neighborhood grows exponentially with instance size, or is simply too large to enumerate, and is therefore searched implicitly or by a specialized sub-algorithm. Their taxonomy has three branches: variable-depth methods, network-flow-based improvement, and restrictions to a polynomially-searchable subclass. The intuition VLSN formalizes is that a bigger neighborhood yields fewer, deeper local optima; the open engineering question it leaves is *how* to define and search an exponentially large neighborhood cheaply for a constrained routing problem.

Constraint programming is the other relevant body of technology. Real routing problems are saturated with side constraints, and CP propagation (maintaining earliest/latest service times along a route, ruling out load- or time-infeasible insertions) plus branch-and-bound is exactly the machinery that decides legality and cost of a partial routing plan efficiently. The tension is that CP's tree search and local search's neighborhood walk are different paradigms that do not obviously compose.

## Baselines

**Steepest descent / k-exchange local search.** Move = a `k`-exchange (2-opt for tours; relocate / swap for routes). Core algorithm: enumerate `N(x)`, take the best improving neighbor, repeat to a local optimum. The math is the local-optimality condition `c(x) ≤ c(x') ∀ x'∈N(x)`. Gap: the neighborhood is `O(n²)`, so the local optima are shallow; under tight time-window/capacity constraints most neighbors are infeasible, so the search both traps early and can barely move.

**Lin–Kernighan / variable-depth search.** Instead of a fixed `k`, chain a sequence of edge swaps, building progressively deeper compound moves `N₁, N₂, …, N_k` and searching them partially, so that as many as `n` edges may change between the start tour and the accepted tour. Core idea: escape a shallow local optimum by going *deeper* rather than wider, with a heuristic partial search keeping the cost down. Gap: the move is still edge-local and tailored to the TSP's structure; it does not naturally carry the side constraints (time windows, precedence, heterogeneous fleet) of real routing, and the depth is achieved by a hand-crafted chaining rule specific to the tour structure.

**Simulated annealing / threshold accepting / great deluge.** These are acceptance rules layered on a small-move neighborhood. SA accepts an improving move always and a worsening move of size `Δ = c(x') − c(x) > 0` with probability `exp(−Δ/T)`, lowering the temperature `T` over time; threshold accepting accepts any move not worse than a shrinking threshold; great deluge (for a minimization objective) accepts any candidate whose cost is no higher than a "water line" that is lowered over time. Core idea: take controlled worsening steps to climb out of local optima. Gap: an acceptance rule only decides whether to keep a neighbor the underlying move proposes — if that neighborhood is small and feasible neighbors are scarce, no acceptance rule can manufacture a feasible reconfiguration that the move never generates in the first place.

**CP branch-and-bound for routing.** Treat insertion positions of customers as constrained variables; propagate load and time-window rules to prune illegal positions; branch-and-bound to the minimum-cost completion. Core idea: an (near-)optimal, constraint-aware way to build or complete a routing plan. Gap: solving the whole instance this way does not scale to the sizes that arise in practice.

## Evaluation settings

The natural yardsticks are the classic routing benchmark libraries: Solomon's VRPTW instances (vehicle routing with time windows, customers on a grid with service times and `[a_i, b_i]` windows), and their pickup-and-delivery variants (Li & Lim; Nanry & Barnes), with instances ranging from tens to hundreds of requests. Instances are described by fleet size, vehicle capacity `Q` or `C_k`, customer demands `q_i`, spatial and temporal customer distributions, time-window density and width, and service times. The standard objective is a lexicographic or weighted combination of (1) number of vehicles used, (2) total travel distance, sometimes (3) total route duration, and — when a request bank is allowed — the number of unserved requests. Distances are Euclidean (some older work truncates to one decimal for integer exact methods; double-precision distances are the modern choice). Reported quality is the gap to the best published solution per instance. (Settings only — the comparison numbers a new method would produce are not part of the ground it stands on.)

## Code framework

The pre-method scaffold is a generic iterated-search harness over a routing solution: a solution representation (routes as customer sequences), a feasibility-aware cost, an initial construction heuristic, an acceptance rule borrowed from the SA/TA family, and one big empty slot for the move operator. The contribution is whatever fills `perturb`.

```python
import math, random

class Solution:
    """A routing plan: a list of routes, each a list of customer ids."""
    def __init__(self, routes):
        self.routes = routes
    def copy(self): ...
    def cost(self):
        """Feasibility-aware objective (distance; +penalty / +∞ if infeasible)."""
        ...

def initial_solution(instance) -> Solution:
    """Cheap construction heuristic (e.g. greedy insertion) -> a feasible plan."""
    ...

def accept(candidate: Solution, current: Solution, T: float) -> bool:
    # SA-style acceptance: improving always; worsening with prob exp(-Δ/T).
    delta = candidate.cost() - current.cost()
    if delta <= 0:
        return True
    return random.random() < math.exp(-delta / T)

def perturb(current: Solution, instance) -> Solution:
    # TODO: the move operator we will design.
    # Classical fillings (relocate / 2-opt) make a SMALL change here and trap.
    pass

def search(instance, iters, T0, cooling):
    current = best = initial_solution(instance)
    T = T0
    for _ in range(iters):
        candidate = perturb(current, instance)        # the slot to design
        if candidate.cost() < best.cost():
            best = candidate.copy()
        if accept(candidate, current, T):
            current = candidate
        T *= cooling
    return best
```
