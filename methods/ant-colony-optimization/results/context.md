# Context: cooperative constructive search for the traveling salesman

## Research question

The traveling salesman problem (TSP) asks for a minimal-length closed tour visiting each of n cities exactly once. It is NP-hard: the number of tours grows factorially, so exhaustive search is hopeless beyond a few dozen cities, and exact branch-and-bound, while it solves moderate symmetric instances, explodes on hard and asymmetric ones. The practical answer is a heuristic that returns a very good tour in reasonable time.

Two families of heuristic exist, and each leaves a gap. The first is TSP-tailored: nearest-neighbor construction, insertion heuristics, and local-search improvers such as 2-opt and Lin-Kernighan. They are fast and strong, but they are *bespoke* — each is engineered around the structure of the symmetric Euclidean TSP and does not transfer, with only minimal change, to a related but differently-structured problem (the asymmetric TSP, the quadratic assignment problem, job-shop scheduling). The second is general-purpose metaheuristics — simulated annealing, tabu search — which apply broadly but evolve a *single* candidate solution along a trajectory of local moves.

The question this landscape poses: is there a *general-purpose, population-based, constructive* search — many simple agents building tours in parallel and cooperating through a shared memory — that finds very good TSP solutions quickly, ports to other combinatorial problems with only a change of problem-specific heuristic, and does not collapse onto a single suboptimal answer?

## Background

**The TSP and the greedy constructive baseline.** Given cities with pairwise distances d_ij, a constructive heuristic builds a tour incrementally: stand at a city, pick the next city, repeat until all are visited, then close the loop. The simplest rule is nearest-neighbor — always step to the closest unvisited city. It is cheap and gives a decent start, but it is myopic: greedy early choices use up the short edges, so the agent is later *forced* to take long edges, and closing the tour back to the start is often very expensive. The natural numeric expression of "closeness is desirable" is the visibility η_ij = 1/d_ij — large for near cities, small for far ones. Any constructive method needs such a greedy force to produce acceptable tours before any learned information exists.

**Real-ant foraging and stigmergy.** Ethologists studied how nearly blind ants find shortest paths between nest and food. The mechanism is pheromone: a walking ant lays a chemical trail and, at a junction, chooses a direction with probability biased by the trail intensity it senses. This is *stigmergy* — indirect communication through persistent modifications of the shared environment rather than direct agent-to-agent messaging. In the controlled double-bridge experiment (Deneubourg et al., 1990, *The self-organizing exploratory pattern of the Argentine ant*), a nest and a food source are joined by two branches. When the branches are unequal, the colony converges to the *shorter* one. The cause is differential path length: an ant taking the short branch reaches the food and returns sooner, so the short branch accumulates pheromone *faster*; later ants then prefer it, deposit more, and the bias compounds.

**Autocatalysis (positive feedback) and why it must be bounded.** The compounding above is autocatalytic — more pheromone on an edge raises the probability of choosing it, which deposits still more pheromone. Positive feedback of this kind discovers good options rapidly, but if nothing opposes it the trail grows without limit and the system locks onto whatever it found first, good or bad. A self-reinforcing search therefore needs a counter-force that lets the accumulated bias decay, so early accidental choices can be forgotten and the search keeps probing alternatives.

**The explore–exploit tension.** A search that only exploits accumulated evidence converges fast but risks premature lock-in on a suboptimal answer (stagnation: every agent ends up tracing the same tour). A search that only explores never concentrates effort where evidence points and wastes its budget. A usable method needs a tunable balance: enough exploitation to home in, enough exploration to keep finding better tours, with knobs that move the operating point between the two.

**Why a population of constructive agents.** A single trajectory-based search holds one solution. If instead many simple agents each build a complete tour in parallel and write what they learn into a shared, slowly-decaying memory on the edges, then (i) the unit of cooperation is a persistent global structure rather than a single current point, (ii) the memory can encode "edges that appeared in many short tours" as a bias for the next round, and (iii) the same construction-plus-shared-memory template can be reused for any problem for which one can define a graph, a constructive step, and a greedy heuristic. The open engineering questions are the exact form of the probabilistic choice rule, what quantity gets deposited and how the deposit reflects solution quality, and how the decay and the choice rule are tuned so the colony converges without stagnating.

## Baselines

**Nearest-neighbor and insertion heuristics.** Greedy construction: from the current city go to the nearest unvisited one (nearest-neighbor), or repeatedly insert the city whose insertion least increases tour length (insertion variants). Fast, deterministic (modulo the start city), and the standard cheap baseline. The gap: myopic — early greedy edges force bad late edges and an expensive tour closure — and purely TSP-specific. Nearest-neighbor also supplies a convenient rough estimate L_nn of a good tour's length.

**2-opt and Lin-Kernighan local search.** Improvement heuristics: start from any tour and repeatedly remove k edges and reconnect to shorten it (2-opt exchanges two edges; Lin-Kernighan adaptively chooses the number of edges). Lin-Kernighan is among the strongest TSP solvers known. The gap: they are pure local-search improvers tied to TSP edge-exchange neighborhoods, with no constructive or learning component, and do not transfer to problems without that neighborhood structure.

**Simulated annealing (SA).** A general-purpose trajectory metaheuristic: from a current solution propose a random neighboring move; accept improving moves always and worsening moves with probability exp(−Δ/T); lower the temperature T on a schedule (e.g. T ← 0.99·T) so the search anneals from exploratory to greedy. General and simple, but it carries a single solution along one trajectory and its behavior hinges on the cooling schedule.

**Tabu search (TS).** Another general-purpose trajectory method: always move to the best non-tabu neighbor, maintaining a tabu list of recently visited solutions (or recent moves) to prevent cycling, sometimes overridden by an aspiration criterion. Effective on TSP, but again single-trajectory and reliant on neighborhood moves and list-length tuning. (Its tabu-list device — forbidding recently used elements — is a useful primitive: a constructive agent likewise needs a per-agent forbidden set to avoid revisiting cities and to guarantee a legal tour.)

## Evaluation settings

The natural yardstick is a small symmetric Euclidean TSP with a known good tour — for instance the 30-city Oliver30 instance from the literature — together with larger grid instances (r×r city grids with known optimal lengths) to probe how cost scales with problem size, and an asymmetric instance (such as ry48p) to test transfer to the ATSP. Distances are Euclidean (or a given asymmetric matrix); a common integer-distance convention is used when comparing against tabulated TSP heuristic results. The metric is tour length (shorter is better), reported as best and average over repeated independent runs under a fixed cycle budget (e.g. NC_MAX cycles) or a fixed wall-clock budget. The natural comparators are the TSP-tailored heuristics above (nearest-neighbor, insertion, 2-opt, Lin-Kernighan) and the general-purpose metaheuristics SA and TS, run under matched hardware and time limits. Secondary diagnostics characterize the search dynamics rather than tour quality: the population's tour-length standard deviation per cycle (is the colony still exploring?) and the average node branching — the count of edges per city whose trail exceeds a small threshold ε (how much of the graph is still in play?).

## Code framework

The available primitives are a fully-connected distance graph, a per-agent forbidden set (the tabu-list device) to keep tours legal, the static visibility η_ij = 1/d_ij, and a generational loop in which a batch of agents each build a complete tour. The empty slots are: the shared edge memory and its initialization, the probabilistic choice rule each agent uses to pick its next city, what each agent contributes back to the shared memory after building a tour, and how that memory decays and is updated between generations.

```python
import random


class Graph(object):
    def __init__(self, cost_matrix, rank):
        self.matrix = cost_matrix
        self.rank = rank
        # TODO: a shared per-edge memory the agents read and write,
        #       plus its initial value.
        self.memory = None


class Colony(object):
    def __init__(self, ant_count, generations, alpha, beta, rho, q, strategy):
        self.ant_count = ant_count
        self.generations = generations
        self.alpha = alpha          # weight on the shared memory term
        self.beta = beta            # weight on the visibility term
        self.rho = rho              # how much memory persists between generations
        self.Q = q                  # scale of an agent's contribution
        self.update_strategy = strategy

    def _update_memory(self, graph, ants):
        # TODO: decay the shared memory, then fold in every agent's contribution.
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
                ant._compute_contribution()
            self._update_memory(graph, ants)
        return best_solution, best_cost


class _Ant(object):
    def __init__(self, colony, graph):
        self.colony = colony
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []                                  # cities already visited (legal-tour constraint)
        self.contribution = []                          # what this agent writes back to shared memory
        self.allowed = [i for i in range(graph.rank)]   # not-yet-visited cities
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j]
                     for j in range(graph.rank)] for i in range(graph.rank)]  # visibility 1/d
        start = random.randint(0, graph.rank - 1)
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        # TODO: choose the next city probabilistically from the shared memory
        #       and the visibility; remove it from `allowed`, append to `tabu`,
        #       accumulate its edge cost, advance `current`.
        pass

    def _compute_contribution(self):
        # TODO: build this agent's write-back to the shared memory from the
        #       tour it just constructed (reflecting how good the tour is).
        pass
```
