## Research question

The traveling salesman problem (TSP) asks for a minimal-length closed tour visiting each of n cities exactly once. It is NP-hard: the number of tours grows factorially, so exhaustive search is hopeless beyond a few dozen cities, and exact branch-and-bound solves moderate symmetric instances but explodes on hard and asymmetric ones. The practical target is a heuristic that returns a very good tour in reasonable time.

Two families of heuristic exist. The first is TSP-tailored: nearest-neighbor construction, insertion heuristics, and local-search improvers such as 2-opt and Lin-Kernighan; each is engineered around the structure of the symmetric Euclidean TSP. The second is general-purpose metaheuristics — simulated annealing, tabu search — which apply broadly and evolve a single candidate solution along a trajectory of local moves.

The question this landscape poses: how should one organize a heuristic search that finds very good TSP solutions quickly and that can be cast in terms general enough to apply to other combinatorial problems (the asymmetric TSP, the quadratic assignment problem, job-shop scheduling) by supplying a problem-specific component?

## Background

**The TSP and the greedy constructive baseline.** Given cities with pairwise distances d_ij, a constructive heuristic builds a tour incrementally: stand at a city, pick the next city, repeat until all are visited, then close the loop. The simplest rule is nearest-neighbor — always step to the closest unvisited city. It is cheap and gives a decent start. The natural numeric expression of "closeness is desirable" is the visibility η_ij = 1/d_ij — large for near cities, small for far ones. Some such greedy force is what makes a constructive heuristic produce acceptable tours.

**The explore–exploit spectrum.** A search that only exploits accumulated evidence converges fast and concentrates effort where the evidence points; a search that only explores keeps sampling broadly and never settles. Any heuristic of this kind lives somewhere on that spectrum, and where it sits governs whether it converges and what it converges to.

**Constructive vs. trajectory search.** A trajectory-based search holds a single candidate solution and moves it through a neighborhood. A constructive search instead builds a complete solution one element at a time from a greedy heuristic. A search organized around a graph, a constructive step, and a greedy heuristic can, in principle, be restated for any problem expressed in those terms.

## Baselines

**Nearest-neighbor and insertion heuristics.** Greedy construction: from the current city go to the nearest unvisited one (nearest-neighbor), or repeatedly insert the city whose insertion least increases tour length (insertion variants). Fast, deterministic (modulo the start city), and the standard cheap baseline. Nearest-neighbor also supplies a convenient rough estimate L_nn of a good tour's length.

**2-opt and Lin-Kernighan local search.** Improvement heuristics: start from any tour and repeatedly remove k edges and reconnect to shorten it (2-opt exchanges two edges; Lin-Kernighan adaptively chooses the number of edges). Lin-Kernighan is among the strongest TSP solvers known.

**Simulated annealing (SA).** A general-purpose trajectory metaheuristic: from a current solution propose a random neighboring move; accept improving moves always and worsening moves with probability exp(−Δ/T); lower the temperature T on a schedule (e.g. T ← 0.99·T) so the search anneals from exploratory to greedy. General and simple, carrying a single solution along one trajectory governed by the cooling schedule.

**Tabu search (TS).** Another general-purpose trajectory method: always move to the best non-tabu neighbor, maintaining a tabu list of recently visited solutions (or recent moves) to prevent cycling, sometimes overridden by an aspiration criterion. Effective on TSP, single-trajectory, and reliant on neighborhood moves and list-length tuning. Its tabu-list device — forbidding recently used elements — is a useful primitive: a constructive agent likewise needs a per-agent forbidden set to avoid revisiting cities and to guarantee a legal tour.

## Evaluation settings

The natural yardstick is a small symmetric Euclidean TSP with a known good tour — for instance the 30-city Oliver30 instance from the literature — together with larger grid instances (r×r city grids with known optimal lengths) to probe how cost scales with problem size, and an asymmetric instance (such as ry48p) to test transfer to the ATSP. Distances are Euclidean (or a given asymmetric matrix); a common integer-distance convention is used when comparing against tabulated TSP heuristic results. The metric is tour length (shorter is better), reported as best and average over repeated independent runs under a fixed cycle budget (e.g. NC_MAX cycles) or a fixed wall-clock budget. The natural comparators are the TSP-tailored heuristics above (nearest-neighbor, insertion, 2-opt, Lin-Kernighan) and the general-purpose metaheuristics SA and TS, run under matched hardware and time limits. Secondary diagnostics characterize the search dynamics rather than tour quality: the population's tour-length standard deviation per cycle (is the search still spread out?) and the average node branching — the count of next-city options per city still carrying appreciable weight above a small threshold ε (how much of the graph is still in play?).

## Code framework

The available primitives are a fully-connected distance graph, a per-agent forbidden set (the tabu-list device) to keep tours legal, the static visibility η_ij = 1/d_ij, and a generational loop in which a batch of agents each build a complete tour. The empty slots are left for whatever cooperative machinery the design calls for: how an agent picks its next city, what per-edge state (if any) is carried, what survives between generations, and how the population's experience feeds the next round.

```python
import random


class Graph(object):
    def __init__(self, cost_matrix, rank):
        self.matrix = cost_matrix
        self.rank = rank
        # TODO: whatever per-edge state, if any, the design needs.
        self.pheromone = None


class Colony(object):
    def __init__(self, ant_count, generations, alpha, beta, rho, q):
        self.ant_count = ant_count
        self.generations = generations
        self.alpha = alpha          # tunable weighting parameter
        self.beta = beta            # tunable weighting parameter
        self.rho = rho              # tunable parameter in [0,1)
        self.Q = q                  # tunable scale parameter

    def _update_pheromone(self, graph, ants):
        # TODO: update the per-edge state between generations from this batch of agents.
        pass

    def solve(self, graph):
        best_cost = float('inf')
        best_solution = []
        for _ in range(self.generations):
            ants = [_Ant(self, graph) for _ in range(self.ant_count)]
            for ant in ants:
                for _ in range(graph.rank - 1):
                    ant._select_next()
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]  # close the tour
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = list(ant.tabu)
                ant._update_pheromone_delta()
            self._update_pheromone(graph, ants)
        return best_solution, best_cost


class _Ant(object):
    def __init__(self, colony, graph):
        self.colony = colony
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []                                  # cities already visited (legal-tour constraint)
        self.pheromone_delta = []                       # what this agent writes back to the shared state
        self.allowed = [i for i in range(graph.rank)]   # not-yet-visited cities
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j]
                     for j in range(graph.rank)] for i in range(graph.rank)]  # visibility 1/d
        start = random.randint(0, graph.rank - 1)
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        # TODO: choose the next city from `allowed`; remove it from `allowed`,
        #       append to `tabu`, accumulate its edge cost, advance `current`.
        pass

    def _update_pheromone_delta(self):
        # TODO: from the closed tour this agent just built, produce whatever it
        #       contributes back for the next generation.
        pass
```
