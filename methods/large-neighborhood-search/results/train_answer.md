We need to produce good routing and scheduling plans at industrial scale — deliver to hundreds of customers with a fleet, respecting capacities, time windows, and pickup-before-delivery precedence — and we need them fast. Exact methods do not reach these sizes, so the working tool is local search wrapped in a metaheuristic: hold a feasible plan, perturb it, keep the perturbation when it helps, repeat. The entire behaviour of that loop rides on a single component, the move operator, so that is where I look. The moves in standard use are tiny: 2-opt cuts two tour edges and reconnects the other way, relocate pulls one customer out and drops it elsewhere, swap exchanges two. Each touches a constant number of elements, so the neighbourhood — the set of plans reachable in one step — has size on the order of $n^2$. That smallness is the appeal, because $N(x)$ can be enumerated and the best improving neighbour taken quickly: steepest descent sets $x \leftarrow \arg\min_{x'\in N(x)} c(x')$ until $c(x) \le c(x')$ for all $x' \in N(x)$. But $n^2$ is microscopic against the real solution space, so those local optima are shallow, and — the part that actually bites — on a tightly-constrained instance the *feasible* fraction of that already-tiny neighbourhood collapses toward nothing. Try to relocate one time-windowed customer to a better slot and almost every destination pushes some downstream customer past its deadline or blows a capacity bound; the move is rejected on feasibility before cost matters, and the only legal destinations sit right back where the customer already was. The search freezes. It is not at the optimum — it is stranded in a feasible basin from which every small feasible step keeps it inside, because the landscape is discontinuous: neighbours differ wildly and most neighbours of a feasible schedule are infeasible.

The reflex is to take worse moves sometimes — simulated annealing, threshold accepting, great deluge — controlled uphill steps to climb out of local optima. But that repairs the wrong half. An acceptance rule decides whether to *keep* a neighbour the move proposes; it cannot *create* a neighbour the move never generates. The barrier between feasible basins here is not a cost barrier I can pay for with $\exp(-\Delta/T)$ — it is a feasibility barrier, with no feasible single-step path across it at all. Variable-depth search à la Lin–Kernighan goes big by chaining many edge swaps into one deep compound move, which is the right spirit, but its mechanics are edge-local and hand-tailored to the TSP; they do not carry time windows and precedence, and the depth comes from a bespoke chaining rule. So acceptance is necessary but not sufficient, and the thing I must change is the move itself.

I propose Large Neighborhood Search, and its self-tuning extension Adaptive Large Neighborhood Search. The single move is destroy-and-repair (ruin and recreate): from the current plan, *remove* a set of $q$ customers entirely — lift them out, shortcutting the routes where they sat — and then *re-insert* them from scratch in the best way I can find. This keeps feasibility for free: after removal I hold a smaller but perfectly feasible partial plan, and reinsertion only ever places a customer where constraints allow, so the result is feasible again. The implied neighbourhood is enormous — every choice of which $q$ to remove times every legal reinsertion — for 100 customers removing 15 that is already $\binom{100}{15} \approx 2.5\times10^{17}$ removal choices — and I never enumerate it; I *sample* it, picking one removal set and doing one good reinsertion. The move is large because the hole is large, which is exactly what lets it reach feasible reconfigurations no sequence of small feasible moves can: it rearranges enough at once to clear the feasibility barrier in one shot.

The size of the hole has to be chosen with care, and the governing insight is *relatedness*. Remove a single customer and, under tight constraints, its best feasible reinsertion is overwhelmingly right back where it came from — the relocate trap again. Remove two unrelated customers, far apart in space and time, and their reinsertions do not interact: each returns near its own old slot, gaining nothing two separate one-customer moves would not. The payoff appears only when the removed customers *compete for the same space* — remove two that are close and currently near each other, and reinsertion can put them back swapped or interleaved, a genuine interchange no single feasible relocate could reach. So remove *related* customers, and no more than necessary, since every extra removed customer makes reinsertion slower. To make "related" precise I want a measure large when customers are close and large when they share a vehicle (sharing a vehicle matters because emptying a route entirely is the only way to drop the vehicle count). With $c_{ij}$ the normalized travel cost in $[0,1]$ and $V_{ij}=0$ if $i$ and $j$ are on the same vehicle and $1$ otherwise,
$$R(i,j) = \frac{1}{c_{ij} + V_{ij}}.$$
Small $c_{ij}$ drives the denominator down and $R$ up; same vehicle drops the penalty term entirely and pushes $R$ higher still — both toward more related. The full pickup-and-delivery variant simply enriches this into a lower-is-more-related weighted distance combining travel distance, time-window proximity, load similarity, and vehicle-compatibility, $R(i,j) = \phi\,(d_{A(i),A(j)}+d_{B(i),B(j)}) + \chi\,(|T_{A(i)}-T_{A(j)}|+|T_{B(i)}-T_{B(j)}|) + \psi\,|l_i-l_j| + \omega\,(1 - |K_i\cap K_j|/\min(|K_i|,|K_j|))$, but proximity-plus-vehicle is the core. To choose the removal set without locking onto one cluster, I grow it from a random seed: repeatedly take a random already-chosen member, rank the remaining customers by relatedness to it, and pick index $\lfloor |L|\cdot \text{rand}^{D}\rfloor$ with $\text{rand}\in[0,1)$ and $D\ge 1$. With $D=1$ the index is uniform (relatedness ignored, plain random removal); as $D\to\infty$ it always picks index 0 (the most related, fully deterministic); a moderate $D$ biases toward related customers while keeping enough wobble to explore.

Reinsertion is where the constraint machinery earns its keep, and there is a real fork. The original LNS repairs *optimally*: treat each removed customer as a variable whose values are legal insertion arcs, propagate load and time-window rules to prune illegal arcs, and branch-and-bound to the minimum-cost completion — the evaluate-this-large-move step is itself a little constraint-programming search over the hole, with limited discrepancy search capping the tree (order insertions cheapest-first, count a discrepancy each time you go against that ordering, and explore only within a discrepancy budget) so one pathological reinsertion cannot hold the run hostage, and with most-constrained-customer-first variable ordering and cheapest-cost value ordering. This is powerful but heavyweight. The decisive observation comes once I pair the move with simulated-annealing acceptance — accept $x'$ from $x$ whenever $c(x')\le c(x)$, else with probability $\exp(-(c(x')-c(x))/T)$, cooling $T \leftarrow c\,T$ — to escape large-move local optima. Putting the removed customers back exactly where they came from is always a feasible completion of the hole, so an *optimal* repair returns something no worse than that: the candidate is improving-or-equal, never strictly worse, for any hole size. Optimal repair therefore engineers *out* the very worsening candidates SA needs to diversify with. So I flip it and use a fast, deliberately imperfect repair. The cheapest, greedy (regret-1), repeatedly inserts $\arg\min_i c_i$ with $c_i = \min_k \Delta f_{i,k}$, where $\Delta f_{i,k}$ is the best feasible insertion cost of $i$ into route $k$. Greedy is myopic and postpones the hard customers until the routes are full, so I generalize to regret-$k$ look-ahead: insert $\arg\max_i \sum_{j=1}^{k}(\Delta f_{i,x_{ij}} - \Delta f_{i,x_{i1}})$, choosing the customer I would *regret* most for not placing now — the one whose good slots I would pay the most to keep — where $x_{ij}$ is the route with $i$'s $j$-th cheapest insertion (with an edge case: if a customer can be inserted into fewer than $m-k+1$ routes it cannot form a full $k$-deep regret, so insert whichever such most-constrained customer fits in the fewest routes, ties by best insertion cost). These imperfect completions are often worse than what was torn down, and that is exactly the point: those worse candidates are what SA accepts to cross a barrier, and the repair is faster than a CP tree besides. The start temperature is calibrated rather than guessed: from $\exp(-\Delta/T_{\text{start}})=0.5$ with $\Delta = w\cdot c(x_0)$, a $w\%$-worse solution is accepted with probability $0.5$ when $T_{\text{start}} = w\cdot c(x_0)/\ln 2$.

What makes ALNS work is that no single destroy/repair pairing wins across instances, geometries, or phases of a run — and even worst-removal (remove customers with large $cost(i,s)=f(s)-f_{-i}(s)$, the ones that look misplaced) pulls on a different rope than relatedness-removal (the easily-interchangeable ones). So carry several of each and let the search learn the mix. Pick a removal and a repair independently by roulette wheel, operator $j$ with probability $w_j/\sum_i w_i$. After each iteration the chosen operators are scored by the fate of their candidate: $\sigma_1$ if it set a new global best, $\sigma_2$ if it was accepted and improved the current solution, $\sigma_3$ if it was accepted while worse — that last case is the diversification that escapes local optima, and the tuned example $(\sigma_1,\sigma_2,\sigma_3)=(33,9,13)$ deliberately rewards it most. A candidate is credited only if its solution is *unvisited* (tracked by a hash table), so cycling back to seen solutions earns nothing; since one iteration fires both a removal and a repair, both used operators receive the same score. Weights then update per segment of, say, 100 iterations: with $\pi_i$ the total score and $\theta_i$ the use-count of operator $i$ over the segment, and reaction factor $r\in[0,1]$,
$$w_{i,j+1} = w_{i,j}\,(1-r) + r\,\frac{\pi_i}{\theta_i},$$
an exponential moving average of recent per-use success — $r=0$ freezes the initial weights, $r=1$ lets only the last segment decide. Operators producing new-bests and accepted moves climb and get picked more; wasteful ones decay. One last refinement keeps the deterministic greedy/regret repair from returning the same completion of a given hole: perturb each evaluated insertion cost $C$ by noise drawn from $[-maxN, maxN]$, using $C' = \max(0, C+\text{noise})$ with $maxN = \eta\cdot\max_{i,j} d_{ij}$ scaled to the instance, so the repair sometimes takes the second-best move — exactly the sampling SA wants — and "noise" versus "no noise" can themselves be two more competing operators the weights tune. The whole loop is destroy a related chunk, repair it, accept by Metropolis, and re-weight: simulated annealing over an implicitly-exponential neighbourhood that tunes its own neighbourhoods.

```python
import math, random

random.seed(0)

# ---- a small CVRP instance: depot at 0, customers 1..N with demands ----
N = 20                      # customers (depot is index 0)
CAP = 30                    # vehicle capacity
coords = [(50, 50)] + [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(N)]
demand = [0] + [random.randint(3, 10) for _ in range(N)]

def dist(a, b):
    (xa, ya), (xb, yb) = coords[a], coords[b]
    return math.hypot(xa - xb, ya - yb)

def route_cost(route):                       # depot -> ... -> depot
    nodes = [0] + route + [0]
    return sum(dist(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))

def solution_cost(routes):
    return sum(route_cost(r) for r in routes)

def load(route):
    return sum(demand[c] for c in route)

# ---- initial feasible solution: greedy sweep into capacity-bounded routes ----
def initial():
    custs = list(range(1, N + 1))
    random.shuffle(custs)
    routes, cur = [], []
    for c in custs:
        if load(cur) + demand[c] <= CAP:
            cur.append(c)
        else:
            routes.append(cur); cur = [c]
    if cur: routes.append(cur)
    return routes

# ---- destroy operators: each returns (partial_routes, removed_customers) ----
def random_removal(routes, q):
    planned = [c for r in routes for c in r]
    removed = set(random.sample(planned, min(q, len(planned))))
    partial = [[c for c in r if c not in removed] for r in routes]
    return [r for r in partial if r], list(removed)

def worst_removal(routes, q, p=3):
    work = [r[:] for r in routes]                        # cost(i,s)=f(s)-f_{-i}(s) changes
    planned = sum(len(r) for r in work)                  # as the route changes, so the
    target = min(q, planned)                             # cost list is recomputed each step
    removed = []
    while len(removed) < target:
        contrib = []
        for ri, r in enumerate(work):
            nodes = [0] + r + [0]
            for k, c in enumerate(r):
                saved = (dist(nodes[k], nodes[k + 1]) + dist(nodes[k + 1], nodes[k + 2])
                         - dist(nodes[k], nodes[k + 2]))
                contrib.append((saved, ri, k, c))
        contrib.sort(reverse=True)                        # descending cost
        y = random.random()
        idx = min(int((y ** p) * len(contrib)), len(contrib) - 1)
        _, ri, k, c = contrib[idx]                        # bias toward high-cost end
        work[ri].pop(k)
        removed.append(c)
    return [r for r in work if r], removed

# ---- repair: best feasible insertion cost of customer c into a given route ----
def best_insertion(route, c):
    if load(route) + demand[c] > CAP:
        return math.inf, None
    nodes = [0] + route + [0]
    best, pos = math.inf, None
    for k in range(len(nodes) - 1):
        delta = dist(nodes[k], c) + dist(c, nodes[k + 1]) - dist(nodes[k], nodes[k + 1])
        if delta < best:
            best, pos = delta, k
    return best, pos

def insert_options(routes, c):                           # sorted (delta, route_idx, pos)
    opts = []
    for ri, r in enumerate(routes):
        d, pos = best_insertion(r, c)
        if pos is not None:
            opts.append((d, ri, pos))
    opts.append((dist(0, c) + dist(c, 0), -1, 0))         # open a fresh route
    opts.sort()
    return opts

def greedy_repair(routes, removed):                      # regret-1
    routes = [r[:] for r in routes]
    pool = removed[:]
    while pool:
        best = None
        for c in pool:
            d, ri, pos = insert_options(routes, c)[0]
            if best is None or d < best[0]:
                best = (d, ri, pos, c)
        d, ri, pos, c = best
        if ri == -1: routes.append([c])
        else:        routes[ri].insert(pos, c)
        pool.remove(c)
    return routes

def regret2_repair(routes, removed):                     # maximize regret c*_i
    routes = [r[:] for r in routes]
    pool = removed[:]
    while pool:
        best = None
        for c in pool:
            opts = insert_options(routes, c)
            d1 = opts[0][0]
            d2 = opts[1][0] if len(opts) > 1 else d1
            key = (d2 - d1, -d1)                          # ties -> lowest insertion cost
            if best is None or key > best[0]:
                best = (key, opts[0], c)
        _, (d, ri, pos), c = best
        if ri == -1: routes.append([c])
        else:        routes[ri].insert(pos, c)
        pool.remove(c)
    return routes

# ---- adaptive operator selection (roulette wheel + segmented weight update) ----
destroy_ops = [random_removal, worst_removal]
repair_ops  = [greedy_repair, regret2_repair]
dw = [1.0, 1.0]                                           # destroy weights
rw = [1.0, 1.0]                                           # repair weights
S1, S2, S3 = 33, 9, 13                                    # scores: best / better / accepted
R = 0.1                                                   # reaction factor r
SEGMENT = 100                                             # iterations per segment

def roulette(weights):
    t = random.random() * sum(weights)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc >= t:
            return i
    return len(weights) - 1

# ---- the search: destroy + repair + SA acceptance, weights learned online ----
def alns(iters=4000):
    cur = best = initial()
    cur_c = best_c = solution_cost(cur)
    T = 0.1 * best_c / math.log(2)                        # ~10%-worse accepted w.p. 0.5
    cooling = 0.9985
    seen = {tuple(tuple(r) for r in cur)}                # initial solution already visited
    traj = []
    d_score = [0.0] * len(dw); d_used = [0] * len(dw)     # per-segment statistics
    r_score = [0.0] * len(rw); r_used = [0] * len(rw)
    for it in range(iters):
        di, ri = roulette(dw), roulette(rw)
        d_used[di] += 1; r_used[ri] += 1
        q = random.randint(2, max(2, N // 4))             # degree of destruction
        partial, removed = destroy_ops[di](cur, q)
        cand = repair_ops[ri](partial, removed)
        cand_c = solution_cost(cand)

        key = tuple(tuple(r) for r in cand)
        score = 0
        if cand_c < best_c - 1e-9:
            best, best_c, score = cand, cand_c, S1        # new global best -> S1
            traj.append((it, best_c))
        delta = cand_c - cur_c
        accepted = delta <= 0 or random.random() < math.exp(-delta / T)
        if accepted:
            if key not in seen:                           # reward only unvisited
                score = max(score, S2 if delta < 0 else S3)
            cur, cur_c = cand, cand_c
        seen.add(key)

        d_score[di] += score; r_score[ri] += score        # accumulate pi_i over segment
        if (it + 1) % SEGMENT == 0:                        # w_{i,j+1} = w_ij(1-r)+r*pi_i/theta_i
            for i in range(len(dw)):
                if d_used[i] > 0:
                    dw[i] = dw[i] * (1 - R) + R * (d_score[i] / d_used[i])
            for i in range(len(rw)):
                if r_used[i] > 0:
                    rw[i] = rw[i] * (1 - R) + R * (r_score[i] / r_used[i])
            d_score = [0.0] * len(dw); d_used = [0] * len(dw)
            r_score = [0.0] * len(rw); r_used = [0] * len(rw)
        T *= cooling
    return best, best_c, traj

best, best_c, traj = alns()
print("improving objective trajectory (iter, best cost):")
for it, c in traj:
    print(f"  iter {it:5d}   best = {c:8.2f}")
print(f"\nfinal routes: {len(best)} vehicles, cost {best_c:.2f}")
print(f"destroy weights {['%.2f' % w for w in dw]}, "
      f"repair weights {['%.2f' % w for w in rw]}")
```
