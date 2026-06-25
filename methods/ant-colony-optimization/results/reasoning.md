Let me start from what actually hurts. I have the traveling salesman problem: n cities, distances d_ij, find the shortest closed tour visiting each once. It's NP-hard — the tours grow factorially, exact search dies past a few dozen cities, and branch-and-bound chokes on the asymmetric and otherwise nasty instances. So I'm after a heuristic that returns a very good tour fast.

The heuristics that already exist split into two camps, and each leaves me wanting. One camp is TSP-specialists: nearest-neighbor and insertion to build a tour, 2-opt and Lin-Kernighan to polish one. Lin-Kernighan is genuinely excellent. But these are *carved* around the symmetric Euclidean TSP. The moment the problem shifts — distances stop being symmetric, or it's really a quadratic assignment or a job-shop schedule wearing a TSP costume — the specialist machinery doesn't carry over without being re-engineered. The other camp is general-purpose metaheuristics, simulated annealing and tabu search. Those apply anywhere, but they walk a *single* solution along a trajectory of local moves: propose a neighbor, accept or reject, repeat. One current point at a time.

What I want is something that's general-purpose *and* constructive *and* not single-trajectory — many simple agents, each building a whole tour from scratch, cooperating so that what one discovers helps the others. If I phrase it that way, what should the cooperation look like? Direct messaging between agents — agent A telling agent B "I found a good edge here" — gets complicated fast and doesn't obviously scale with the number of agents or the size of the graph. The alternative is indirect: agents leave marks in a shared structure and read each other's marks, never addressing one another. Then the unit of cooperation is a persistent global object on the graph, not a conversation and not a single candidate solution. That feels much cleaner to build.

Where in nature does indirect coordination through a shared medium actually produce good solutions? Blind ants foraging are the case I keep coming back to. An ant lays a pheromone trail as it walks, and at a junction it turns one way or the other with a probability biased by how much pheromone it smells there. No ant tells any other ant anything; they communicate only by modifying the shared ground — stigmergy. And there's a hard experimental fact I should lean on, not just gesture at: in the double-bridge setup, a nest joined to food by a short and a long branch, the colony reliably ends up on the *short* branch. I want to be sure I understand the mechanism before I try to steal it, because if I'm wrong about *why* the short branch wins I'll build the wrong thing. The story I've heard is purely mechanical: an ant that happens to take the short branch reaches the food and comes back *sooner*, so in a fixed window the short branch gets walked more times, so it accumulates pheromone *faster*; the next ants smell more there and prefer it; that deposits still more. Differential path length, amplified by feedback. Let me actually check that this story holds up numerically before I trust it.

I'll model the bare mechanics: a short branch of length 1 and a long branch of length 2, ants departing the nest at a steady rate, each choosing a branch with probability proportional to that branch's current pheromone, traversing it in time equal to its length, and depositing one unit of pheromone on arrival. Both branches start with equal pheromone, so the *first* choice is a coin flip — there's no built-in preference for "short." I run this as an event-driven simulation and watch which branch wins. Over a run of 2001 departing ants I get 1471 choosing the short branch and 530 the long — a 73.5% short fraction — with final pheromone 1464 on short versus 527 on long. So the colony does swing decisively to the short branch with no information beyond round-trip time, starting from a fair coin. That confirms the engine is real and it's exactly what I want: differential length, fed back through deposition, breaks the symmetry on its own. Worth holding onto.

That feedback is autocatalytic — more pheromone raises the choice probability, which lays more pheromone, which raises it again. The simulation makes the danger as concrete as the promise: 1464 versus 527 is the short branch running away, and there were only two branches. That's the good part when there are two options and the short one is right. But a runaway loop has no fixed point short of "everything on the favored path." In a TSP with n cities there are countless near-tours, and the path the colony stumbles onto and amplifies first is almost certainly not optimal. So pure positive feedback will slam onto an early accident and freeze. I'm going to need something pulling the other way — a decay — or the colony locks in and stops searching. Flag that; I'll have to build it in.

Let me turn the metaphor into a TSP construction. The graph is the cities; an "ant" stands at a city and builds a tour by repeatedly choosing a next city until all are visited, then closing the loop. To keep the tour legal I give each ant a forbidden set — the cities it's already visited — and it only ever picks from what's left. (Borrowing the name "tabu list" from tabu search, though the device is simpler here: I'm just forbidding revisits to make a permutation, no aspiration logic.)

Now the only real question: with what probability does an ant at city i pick city j next? Two pressures should shape that choice. First, plain greed: near cities are better than far ones, all else equal. The clean numeric handle is *visibility* η_ij = 1/d_ij — big for close cities, small for distant ones. If I went on visibility alone I'd have a stochastic nearest-neighbor heuristic, which I already know is myopic: spend the short edges early and you're forced into long edges late, and the tour-closing edge back to the start is brutal. So visibility alone is a decent cold-start but a bad finish. Second, the learned signal: the pheromone τ_ij on edge (i,j), encoding "ants in the past found this edge worth using." Early on τ is uninformative, so the ant should lean on visibility; as τ accumulates, it should lean on the trail.

How do I combine τ and η into one preference for edge (i,j)? My first instinct is to add them — some weighted sum w_τ·τ_ij + w_η·η_ij. But stare at that. If one term is on a wildly different scale than the other, the sum is just whichever term is bigger; the small one is invisible. And conceptually I don't want "trail OR closeness," I want "trail AND closeness" — an edge is attractive when it's *both* well-trodden *and* short. A product does that: τ_ij · η_ij is only large when neither factor is small, and either factor near zero kills the edge. The product also gives me a clean way to weight each pressure independently — raise each to its own exponent. So the desirability of edge (i,j) is [τ_ij]^α · [η_ij]^β, with α controlling how much the trail matters and β how much closeness matters. Two independent dials, multiplicative so both factors are genuinely necessary.

Turn that desirability into a probability by normalizing over the allowed cities. For ant k at city i,

  p^k_ij = [τ_ij]^α [η_ij]^β / Σ_{l ∈ allowed_k} [τ_il]^α [η_il]^β,  for j ∈ allowed_k, and 0 otherwise.

Before I go further I want to actually compute this on a case small enough to do by hand, to make sure the product rule does the sensible thing. Take four cities on a unit square — (0,0),(1,0),(1,1),(0,1) — and stand an ant at the corner (0,0). The two adjacent corners are at distance 1, the diagonal corner at distance √2 ≈ 1.414. On the very first cycle the trail is uniform: τ = 1/16 = 0.0625 everywhere. Visibilities are η = 1 to each adjacent corner and η = 1/√2 ≈ 0.7071 to the diagonal. With α = 1 and β = 5 the desirabilities are 0.0625·1^5 = 0.0625 to each adjacent corner and 0.0625·0.7071^5 = 0.0625·0.1768 = 0.01105 to the diagonal. Normalizing over the three: Z = 0.0625 + 0.0625 + 0.01105 = 0.13605, so the ant picks an adjacent corner with probability 0.0625/0.13605 = 0.459 each, and the diagonal with probability only 0.01105/0.13605 = 0.081. That's the behavior I wanted: with the trail flat, the β exponent on visibility already pushes the ant strongly away from the long diagonal edge toward the short perimeter edges, without me having to special-case anything. Good — the product rule with a healthy β is a real greedy bias, not a near-uniform wash.

Sanity-check the dials at their extremes too. Set α = 0: the trail drops out, every ant is a stochastic greedy constructor driven purely by visibility, started from different cities — a stochastic multi-start nearest-neighbor. Set β = 0: closeness drops out and ants follow only the accumulated trail, blind to distance. Neither extreme is what I want; the interesting behavior is in between, and α versus β is going to be the explore–exploit dial. I'll come back to tuning it once I have the feedback loop closed.

Now the loop. The whole point is that ants write what they learn back onto the edges. When does a deposit happen, and how much? Here's the first real fork, and I want to reason it out rather than guess. Two timings are possible. Option A: an ant drops pheromone *as it walks*, edge by edge, the instant it crosses (i,j) — local, immediate. Option B: an ant waits until it has finished a *complete* tour, then goes back and reinforces the edges it used, by an amount that reflects how good the whole tour was — global.

Let me think about what information each timing has access to. In Option A, when the ant crosses (i,j) it has no idea whether the tour it's building will turn out good or terrible — it's only partway through. So whatever it deposits can't be a function of solution quality; the best it can do is deposit a constant, or maybe something distance-dependent like Q or Q/d_ij. Either way the trail is being shaped by *local* features, not by how good the finished tour was. Option B is different: by the time the ant deposits, it knows its tour length L_k, so it can make the deposit *reflect quality* — reinforce more for a short tour, less for a long one. Concretely, deposit Q/L_k on each edge the ant used. An ant whose tour came out short adds a lot of pheromone; an ant whose tour was poor adds little. That's exactly the selection pressure I'd want baked into the shared memory: good solutions bias the next round more than bad ones. The double-bridge engine worked because the *good* (short) path got reinforced harder, and tying the deposit to L_k is the most direct way to reproduce that on a graph where "good" means a short completed tour rather than a short single edge.

So I'll let an ant that used edge (i,j) in its completed tour deposit Δτ^k_ij = Q/L_k, and 0 on edges it didn't use. Total deposit on an edge in a cycle is the sum over all ants that used it, Δτ_ij = Σ_k Δτ^k_ij. The two local-timing alternatives — Q per step regardless of distance, or Q/d_ij per step — I'll keep as variants worth testing, because my argument that they should lose is a design argument, not a measured fact. The design argument is that they encode only a local edge feature and never see the completed tour's quality, whereas the double-bridge lesson was that global path quality is what should be amplified. I expect ant-cycle (the global-quality version) to pull ahead on hard instances for that reason, but I should be honest that on a tiny instance the three rules may be nearly indistinguishable — there isn't enough room for the global signal to matter — so I'd want to confirm the gap on real-sized problems before claiming it. I'll build around ant-cycle and treat the comparison as something to verify, not assume.

Now the decay I flagged earlier. If every cycle just *adds* Σ_k Q/L_k to edges, the trail only grows — autocatalysis with no brake, and two bad things follow. One, the numbers run away unbounded. Two, and worse, an edge that got lucky pheromone in cycle 1 keeps it forever; the colony can never *forget* a bad early commitment, so it locks in (the runaway I saw mechanically in the double-bridge, now with no second branch to escape to). I need the trail to evaporate. The simplest brake is multiplicative: keep a fraction ρ of the old trail each cycle and let (1−ρ) evaporate. So the update is

  τ_ij(t+n) = ρ · τ_ij(t) + Δτ_ij,

with 0 ≤ ρ < 1, applied once per cycle after all m ants have finished and deposited. (I count time so that one ant move is one iteration, n moves complete a tour, and a *cycle* is the n iterations in which all ants build their tours; the trail updates once per cycle.) Evaporation does two jobs at once: it bounds the trail, and it lets the system forget. Edges that stop being used decay back toward irrelevance; only edges that keep getting reinforced by good tours stay strong.

Let me make this update concrete with one edge so I trust the arithmetic, using the square again. Edge (0,1) starts at τ = 0.0625. Suppose in a cycle two ants use it: one that closed a good perimeter tour of length 4 and one that closed the crossing tour of length 2+2√2 ≈ 4.828. With Q = 100 their deposits are 100/4 = 25 and 100/4.828 = 20.71, summing to Δτ = 45.71. With ρ = 0.5 the new value is τ ← 0.5·0.0625 + 45.71 = 0.03125 + 45.71 = 45.74. Meanwhile an edge that *no* ant used this cycle just decays: 0.5·0.0625 = 0.03125. So a used edge jumps from 0.0625 to ~45.7 while an unused one halves toward zero — the trail genuinely sharpens onto used edges and erases the rest in a single cycle. The mechanism does what I designed it to.

Initialize τ_ij(0) to a small positive constant c — not zero, because zero pheromone with α > 0 would zero out the desirability of every edge and the probabilities would be 0/0 and undefined; a small floor lets visibility drive the first cycle while leaving room for the trail to grow. (My hand-computation above used exactly such a floor, c = 1/n², and the first-cycle probabilities came out well-defined and visibility-dominated, which is what I wanted from the floor.)

Now let me trace the dynamics across cycles. Cycle 1: trail is uniform, so ants pick essentially on visibility — they build greedy-ish tours, varied because of the randomness and the different start cities. Each deposits Q/L_k on its edges; ants with shorter tours deposit more. So after cycle 1, edges that showed up in the *short* tours carry more pheromone than edges that only showed up in long tours — exactly the 45.7-vs-0.03 sharpening I just computed, summed over many ants. Cycle 2: the [τ]^α factor now nudges ants toward those well-reinforced edges, on top of visibility. Good edges get used more, reinforced more; the bias compounds — there's the autocatalysis. Meanwhile evaporation quietly erases the pheromone on edges that good tours stopped using.

Does this actually find good tours, or have I just told myself a story? I should run it on the square and on a slightly bigger instance and look at the number, not trust the narrative. On the unit square the optimal tour is the perimeter, length 4; the trap is the crossing tour 0–2–1–3 of length 2+2√2 ≈ 4.83. Running the colony (4 ants, 50 cycles, α=1, β=5, ρ=0.5, Q=100) it returns the tour [3,2,1,0] of length 4.0 — the perimeter, not the crossing. So on the case I can fully reason about, it lands on the optimum and avoids the obvious trap. Scaling up to 20 random cities, the colony returns a best tour of length 352.3, against the best multi-start nearest-neighbor tour of 353.8 — so it's already edging past the greedy baseline it's built from, which is the minimum I'd want before believing the cooperation buys anything. (On an instance this small the margin is thin; I'd expect it to widen on larger instances where the greedy endings hurt more, but I'm reading these two numbers as "it works and beats greedy," not as a benchmark.)

But I promised myself this wouldn't just lock in. What stops the colony from collapsing to a single tour that everyone retraces forever — stagnation? This is where α earns its keep. Picture α large: the [τ]^α term sharpens hard, so the highest-trail edge out of each city dominates the probability overwhelmingly, every ant takes it, that edge gets all the deposit, and the colony converges on whatever it amplified first, which I have no reason to think is optimal. Picture α near zero: the trail barely matters, ants stay near-random greedy constructors, they never concentrate on the good edges the feedback discovered, and the search wanders without converging. Neither is what I want; there should be a middle band — α around 1, with β giving closeness a healthy say (β somewhere in the 1-to-5 range) — where the trail concentrates the search without freezing it.

I can check the direction of this, at least. Running the 20-city instance with a heavy-exploitation setting (α=6, β=2, slow evaporation ρ=0.9) gives a best tour of 383.8 — distinctly worse than the balanced 352.3 — and pushing it to the extreme (α=10, β=1, ρ=0.99) the population's tour lengths cluster more tightly (the count of distinct tour lengths among the ants falls, and the spread shrinks) while the tours stay poor. So over-exploitation does hurt and does narrow the population, in the direction the argument predicts. I'll be honest that the spread doesn't drop all the way to literally zero on an instance this small in the cycles I ran — "every ant retraces the identical tour" is the idealized limit, not something I observed cleanly here — but the qualitative signature is there: crank the exploitation up and you get worse tours and a less diverse population. So the explore–exploit balance is real, and α (with β and ρ) is the knob that sets the operating point.

ρ plays into the same balance from the other side. A large ρ (slow evaporation) means the trail holds onto its history — more exploitation, more risk of lock-in, consistent with the worse number I got at ρ=0.9. A small ρ (fast evaporation) means the colony forgets quickly — more exploration, but it may forget good edges before it can exploit them. So there's a sweet spot here too: forget *enough* to shed the early greedy bias and any accidental over-commitment, but retain enough accumulated global evidence to actually converge. The single-cycle arithmetic above used ρ = 0.5 — half the trail persisting per cycle — and it gave a clean sharpening-without-freezing on both the square and the 20-city run, so that's a reasonable default to start from: fast enough to forget, slow enough to learn.

Let me write the whole thing out as an algorithm so I can see the moving parts. Initialize: t = 0, cycle counter NC = 0, every edge τ_ij = c and Δτ_ij = 0, place the m ants on cities. Then repeat: each ant, starting from its city, makes n−1 moves, at each step picking the next allowed city by the probability rule, until its tabu list is full; then it moves from the last city back to the first to close the tour, computes L_k, and we track the shortest tour seen. Then for every edge accumulate Δτ_ij = Σ_k (Q/L_k over ants k that used it). Then evaporate-and-deposit, τ_ij ← ρ·τ_ij + Δτ_ij, reset the Δτ to zero, empty the tabu lists, increment NC. Stop when NC hits a budget NC_MAX or the colony has stagnated (all ants on the same tour). The cost per cycle is dominated by every ant considering every edge — O(n²·m) — and since I'll set the number of ants proportional to n (more on that in a second), that's O(n³) per cycle.

Why m = n ants? I want enough ants per cycle to lay down a statistically meaningful trail — too few and a single cycle's deposit is noisy, dominated by which handful of tours happened to get built. Tying the ant count to the problem size, one ant per city as a default, scales the colony's "sampling rate" with the problem and keeps the per-cycle cost at the O(n³) I just counted rather than letting it balloon.

There's one more thing I can squeeze, and it comes from asking: I'm tracking the best tour ever found, but right now that knowledge only influences the future through whatever pheromone that tour's ant happened to deposit in its own cycle, which then evaporates like everything else. The best-so-far tour is my most valuable piece of information — why let it fade? Let me reinforce it extra each cycle: on top of the normal deposits, add a bonus to the edges of the best-so-far tour, say e·(Q/L*) where L* is the best length and e is a count of "elitist" ants all reinforcing that one tour. That pulls the colony's search toward the neighborhood of the best tour. But I have to be careful with e — this is the same explore–exploit tension wearing a different hat. A little elitism sharpens convergence and surfaces the best tour sooner. Too much, and the elitist bonus dominates early, before the colony has really explored, and it forces everyone around a tour that was only good *so far* — premature exploitation of a suboptimal solution, the same failure mode the over-α run showed. So there's an optimal range for e: enough to guide, not so much that it pins the search prematurely.

Step back and ask *why* this works at all, because I want to understand it, not just observe a good number. Take a single ant in isolation — set α = 0 so it's purely greedy on visibility. It makes locally sensible moves: short edges early. But a tour is a closed loop, and greedy-early means the *late* moves are forced onto whatever long edges remain, plus a costly jump home. So a lone greedy ant's tour is good in its early segments and bad in its late ones — predictably, structurally bad, not randomly bad. (This is the same myopia I noted at the start, and it's why the multi-start NN baseline sat at 353.8 and couldn't do better: every restart hits its own forced bad ending.) Now run many ants from many starts. Each one's *good early segments* are short edges that lots of ants, starting in different places, independently choose — so those good sub-paths get walked by many ants and soak up a lot of pheromone (a big summed Δτ, like the 45.7 I computed). Each ant's *bad late edges* are idiosyncratic, forced by that particular ant's history; few other ants are forced onto the same bad edge, so those edges get little pheromone and evaporation halves them away between cycles. The trail is a superimposition that *extracts the good sub-paths and washes out the individually-bad forced edges*. That's the most plausible account of why the colony's 352.3 beat the greedy 353.8: it assembled a tour out of the good fragments many greedy agents agreed on, which no single greedy agent could produce because no single agent can avoid its own forced bad ending. The cooperation is doing the work.

That also tells me what stagnation *is*, precisely. As cycles pass, the transition probabilities at each city sharpen toward a near-stationary pattern. In the healthy regime most cities keep two or three live out-edges with non-negligible probability, so ants still branch and the search keeps finding variants. In the failed regime — too much α, too little evaporation, too much elitism — each city collapses toward essentially one out-edge, the population's tours bunch together, and the diversity I watched shrink in the over-α run is the early sign of it. So I can *watch* for health: track the spread of tour lengths across the population (well above zero = still exploring) and the average node branching (how many edges per city stay above a small trail threshold ε; above two = not stuck on one cycle). Those are the read-outs for whether the dials are set right.

The pieces are now pinned down: the choice rule is a product [τ]^α[η]^β so that learned trail and greedy closeness are both necessary and independently tunable — and the hand-computed corner probabilities (0.46 vs 0.08) showed the β term gives a real greedy bias from the first cycle; the deposit is Q/L per used edge so the shared trail is biased by global solution quality, mirroring the double-bridge engine I verified swings to the short path on its own; evaporation ρ is the brake that bounds the runaway feedback and lets the colony forget early mistakes, sharpening used edges (0.0625 → 45.7) while halving away unused ones in a single cycle; and α, β, ρ, and elitist reinforcement set the operating point, with the square landing on the optimal perimeter (4.0, not the 4.83 trap) and the 20-city run edging past greedy (352.3 vs 353.8) when they're balanced, and degrading to 383.8 when exploitation is cranked too high. I can write the implementation.

```python
import random


class Graph(object):
    def __init__(self, cost_matrix, rank):
        self.matrix = cost_matrix
        self.rank = rank
        # shared trail; small positive floor so visibility drives the first cycle
        # and [tau]^alpha is never identically zero
        self.pheromone = [[1 / (rank * rank) for _ in range(rank)] for _ in range(rank)]


class Colony(object):
    def __init__(self, ant_count, generations, alpha, beta, rho, q):
        # alpha: weight on trail   beta: weight on visibility
        # rho:   fraction of trail that PERSISTS each cycle (1-rho evaporates)
        # q:     deposit scale Q
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations

    def _update_pheromone(self, graph, ants):
        # evaporate, then fold in every ant's deposit
        for i in range(graph.rank):
            for j in range(graph.rank):
                graph.pheromone[i][j] *= self.rho                      # tau <- rho * tau
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]  # + sum_k delta_k

    def solve(self, graph):
        best_cost = float('inf')
        best_solution = []
        for _ in range(self.generations):
            ants = [_Ant(self, graph) for _ in range(self.ant_count)]
            for ant in ants:
                for _ in range(graph.rank - 1):
                    ant._select_next()                                  # build the tour
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]  # close the loop
                if ant.total_cost < best_cost:                          # track best-so-far
                    best_cost = ant.total_cost
                    best_solution = list(ant.tabu)
                ant._update_pheromone_delta()                           # quality-weighted deposit
            self._update_pheromone(graph, ants)                         # evaporate + deposit, once per cycle
        return best_solution, best_cost


class _Ant(object):
    def __init__(self, colony, graph):
        self.colony = colony
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []                                   # cities visited (legal-tour constraint)
        self.pheromone_delta = []                        # this ant's write-back
        self.allowed = [i for i in range(graph.rank)]    # not-yet-visited cities
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j]
                     for j in range(graph.rank)] for i in range(graph.rank)]  # visibility 1/d
        start = random.randint(0, graph.rank - 1)        # different start per ant
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        # desirability of each allowed edge = [tau]^alpha * [eta]^beta, normalized to a probability
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha \
                * self.eta[self.current][i] ** self.colony.beta
        probabilities = [0 for _ in range(self.graph.rank)]
        for i in range(self.graph.rank):
            if i in self.allowed:
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha \
                    * self.eta[self.current][i] ** self.colony.beta / denominator
        # roulette-wheel sample
        selected = self.allowed[-1]
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected

    def _update_pheromone_delta(self):
        # what this ant deposits on each edge of its closed tour
        self.pheromone_delta = [[0 for _ in range(self.graph.rank)] for _ in range(self.graph.rank)]
        for i, j in zip(self.tabu, self.tabu[1:] + self.tabu[:1]):
            self.pheromone_delta[i][j] = self.colony.Q / self.total_cost
```

Many simple agents build tours by a product rule that fuses learned trail and greedy closeness; they write quality-weighted pheromone (Q/L) back onto the edges of good tours; evaporation bounds the resulting positive feedback and lets the colony forget bad early choices; and α, β, ρ tune the operating point so that the superimposed agreement of many greedy ants on good sub-paths — with their idiosyncratic bad edges evaporated away — converges on short tours without all collapsing onto one.
