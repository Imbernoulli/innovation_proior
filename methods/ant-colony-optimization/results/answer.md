# Ant Colony Optimization: stigmergic constructive search for the TSP

## Problem

Find a short closed tour through n cities with pairwise distances d_ij, visiting each city once. The TSP is NP-hard, so the practical target is a heuristic that returns a very good tour fast and — unlike the TSP-tailored specialists (nearest-neighbor, 2-opt, Lin-Kernighan) — ports to other combinatorial problems (asymmetric TSP, quadratic assignment, job-shop) with only a change of problem-specific heuristic. The single-trajectory metaheuristics (simulated annealing, tabu search) are general but carry one candidate solution along a chain of local moves. The aim here is a *population-based, constructive* search: many simple agents each build a complete tour in parallel and cooperate through a shared, slowly-decaying memory on the edges.

## Key idea

Borrow the foraging mechanism of ants. Each agent ("ant") builds a tour by walking from city to city; the cooperation is *stigmergic* — agents communicate only by depositing **pheromone** τ_ij on the edges they use and reading each other's deposits. Two forces shape the next-city choice, combined multiplicatively so both must be satisfied: the learned trail τ_ij and the greedy **visibility** η_ij = 1/d_ij (large for near cities). For ant k standing at city i, the probability of moving to an unvisited city j ∈ allowed_k is

    p^k_ij = [τ_ij]^α [η_ij]^β / Σ_{l ∈ allowed_k} [τ_il]^α [η_il]^β,   (0 otherwise),

with α weighting the trail and β the closeness. α = 0 gives stochastic multi-start nearest-neighbor; β = 0 follows the trail blind to distance; the useful regime is in between (defaults α ≈ 1, β ≈ 5).

After every ant has completed a tour, the trail is updated **once per cycle**:

    τ_ij ← ρ · τ_ij + Δτ_ij,   Δτ_ij = Σ_k Δτ^k_ij,

where ρ ∈ [0,1) is the fraction of trail that *persists* (so 1−ρ evaporates), and an ant that used edge (i,j) in its completed tour deposits a **quality-weighted** amount

    Δτ^k_ij = Q / L_k   (and 0 on edges it did not use),

with L_k the length of ant k's tour. Short tours deposit more, so the shared memory is biased by global solution quality; the local variants (constant Q per step, or Q/d_ij per step) are only local edge-reinforcement rules and do not encode completed-tour quality. Evaporation is the brake on the autocatalytic positive feedback: it bounds the trail and lets the colony forget lucky early choices. The trail is initialized to a small positive constant c so visibility drives the first cycle and [τ]^α is never identically zero. An optional **elitist** bonus reinforces the best-so-far tour's edges each cycle by e·(Q/L*).

Why it converges on good tours: a lone greedy ant is structurally bad at the *end* of its tour (greedy-early forces long late edges), but its good early sub-paths are short edges that many ants, started from different cities, independently agree on. Superimposing m ants reinforces those shared good fragments while each ant's idiosyncratic bad forced edges get little pheromone and evaporate away. α, β, ρ (and elitist e) set the explore–exploit operating point: too much exploitation and every ant retraces one tour (stagnation, tour-length spread → 0); too little and the search never concentrates.

## Algorithm

    initialize τ_ij = c for all edges; best ← none
    for each cycle (until NC_MAX or stagnation):
        for each of m ants:
            place ant on a random start city; tabu ← {start}
            while not all cities visited:
                pick next city j from allowed by p^k_ij = [τ_ij]^α[η_ij]^β / Σ_l [τ_il]^α[η_il]^β
                add j to tabu; accumulate d; advance
            close the tour (last → first); compute L_k
            if L_k < best.cost: best ← this tour
            record this ant's deposit Δτ^k_ij = Q/L_k on each used edge
        for all edges: τ_ij ← ρ · τ_ij + Σ_k Δτ^k_ij   (evaporate, then deposit)
    return best

## Code

```python
import random


class Graph:
    def __init__(self, cost_matrix, rank):
        self.matrix = cost_matrix
        self.rank = rank
        # shared trail; small positive floor so visibility drives the first cycle
        # and [tau]^alpha is never identically zero
        self.pheromone = [[1 / (rank * rank) for _ in range(rank)] for _ in range(rank)]


class Colony:
    def __init__(self, ant_count, generations, alpha=1.0, beta=5.0, rho=0.5, q=100.0):
        # alpha: weight on trail   beta: weight on visibility
        # rho:   fraction of trail that PERSISTS each cycle (1-rho evaporates)
        # q:     deposit scale Q
        self.ant_count = ant_count
        self.generations = generations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = q

    def _update_pheromone(self, graph, ants):
        # evaporate, then fold in every ant's deposit: tau <- rho*tau + sum_k delta_k
        for i in range(graph.rank):
            for j in range(graph.rank):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    def solve(self, graph):
        best_cost = float('inf')
        best_solution = []
        for _ in range(self.generations):
            ants = [_Ant(self, graph) for _ in range(self.ant_count)]
            for ant in ants:
                for _ in range(graph.rank - 1):
                    ant._select_next()                                       # build the tour
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]    # close the loop
                if ant.total_cost < best_cost:                               # track best-so-far
                    best_cost = ant.total_cost
                    best_solution = list(ant.tabu)
                ant._update_pheromone_delta()                                # quality-weighted deposit
            self._update_pheromone(graph, ants)                             # evaporate + deposit, once per cycle
        return best_solution, best_cost


class _Ant:
    def __init__(self, colony, graph):
        self.colony = colony
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []                                    # cities visited (legal-tour constraint)
        self.pheromone_delta = []                         # this ant's write-back
        self.allowed = [i for i in range(graph.rank)]     # not-yet-visited cities
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j]
                     for j in range(graph.rank)] for i in range(graph.rank)]  # visibility 1/d
        start = random.randint(0, graph.rank - 1)         # different start per ant
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        # desirability of each allowed edge = [tau]^alpha * [eta]^beta, normalized to a probability
        denom = 0.0
        for j in self.allowed:
            denom += self.graph.pheromone[self.current][j] ** self.colony.alpha \
                * self.eta[self.current][j] ** self.colony.beta
        probabilities = [0.0 for _ in range(self.graph.rank)]
        for j in self.allowed:
            probabilities[j] = self.graph.pheromone[self.current][j] ** self.colony.alpha \
                * self.eta[self.current][j] ** self.colony.beta / denom
        # roulette-wheel sample over the allowed cities
        selected = self.allowed[-1]
        rand = random.random()
        for j, p in enumerate(probabilities):
            rand -= p
            if rand <= 0:
                selected = j
                break
        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected

    def _update_pheromone_delta(self):
        # what this ant deposits on each edge of its completed closed tour
        self.pheromone_delta = [[0 for _ in range(self.graph.rank)] for _ in range(self.graph.rank)]
        for i, j in zip(self.tabu, self.tabu[1:] + self.tabu[:1]):
            self.pheromone_delta[i][j] = self.colony.Q / self.total_cost
```

A default colony uses m = n ants (one per city, so a cycle costs O(n³) and lays a statistically meaningful trail), α = 1, β = 5, ρ = 0.5, and Q = 100 as the deposit scale. The same construction-plus-shared-memory template transfers to any problem for which one can define a graph, a constructive step, and a greedy heuristic η.
