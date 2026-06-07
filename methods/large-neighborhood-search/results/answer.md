# Large Neighborhood Search (LNS) and Adaptive LNS (ALNS)

## Problem

Solve large-scale, tightly-constrained combinatorial optimization problems — especially vehicle routing and scheduling (VRP, VRPTW, pickup-and-delivery, crew scheduling) — for which exact methods do not scale and classical small-move local search (2-opt, relocate, swap) gets trapped. On constrained instances the trap is a *feasibility* barrier, not merely a cost barrier: once at a feasible local optimum, almost every small move is infeasible (violating a time window or capacity), so the search cannot reach a different feasible basin and freezes. No acceptance rule alone fixes this, because the rule can only filter neighbours the move operator proposes.

## Key idea

Replace the small move with a **large move = destroy + repair** (a.k.a. "ruin and recreate"). One step removes a sizeable set of `q` elements (customers/requests) from the current solution and re-optimizes the resulting hole by reinserting them. The neighbourhood — every choice of which `q` to remove × every legal reinsertion — is exponentially large (for 100 customers, removing 15 gives `C(100,15) ≈ 2.5×10¹⁷` removal choices) and is **sampled**, never enumerated. A large hole gives the freedom to reach feasible reconfigurations no small feasible move can.

Three design refinements make it work:
- **Remove *related* elements** (spatially close / same route), so reinsertion can genuinely interchange them; removing unrelated elements just reinserts them independently and gains nothing.
- **Acceptance from simulated annealing**, so the search escapes large-move local optima: accept `x'` if `c(x') ≤ c(x)`, else with probability `exp(−(c(x')−c(x))/T)`, cooling `T ← c·T`.
- **Adaptivity (ALNS):** no single destroy/repair operator is best across instances or phases, so carry several, pick by roulette wheel from learned weights, and update those weights toward each operator's recent success. Under SA acceptance a *fast, imperfect* repair (greedy / regret-k) is often preferable to an optimal one: an optimal repair never returns a candidate worse than the current solution (putting the removed elements back is always a feasible completion), so it gives SA no worsening move to diversify with — whereas a heuristic repair is faster and its imperfect completions supply exactly those diversifying candidates.

## Algorithm

Maintain best `x_b`, current `x`. Destroy operators `Ω⁻` with weights `w⁻`, repair operators `Ω⁺` with weights `w⁺`.

1. **Initialize**: feasible `x` (cheap construction); `x_b = x`; all weights `= 1`.
2. **Select** a destroy `d` and a repair `r` by roulette wheel: `P(j) = w_j / Σ_i w_i` (destroy and repair chosen independently).
3. **Destroy + repair**: `x_t = r(d(x))` — remove `q` elements, reinsert them.
   - *Removal*: **Shaw/related removal** — the original simple relatedness is `R(i,j) = 1/(c_ij + V_ij)` (`c_ij` normalized travel cost, `V_ij = 0` if same vehicle else `1`; higher `R` = more related); grow the set from a random seed, each step ranking the rest by relatedness and picking index `⌊|L|·rand^D⌋` (`D` controls determinism). The full PDPTW variant uses a lower-is-more-related weighted distance `R(i,j) = φ(d_{A(i),A(j)}+d_{B(i),B(j)}) + χ(|T_{A(i)}−T_{A(j)}|+|T_{B(i)}−T_{B(j)}|) + ψ|l_i−l_j| + ω(1 − |K_i∩K_j|/min(|K_i|,|K_j|))` combining distance, time, load, and vehicle-compatibility terms. Also **random removal** (= `D=1`) and **worst removal** (remove high `cost(i,s) = f(s) − f_{−i}(s)`).
   - *Reinsertion*: **greedy** (regret-1) inserts `argmin_i c_i`, `c_i = min_k Δf_{i,k}`; or **regret-k** inserts `argmax_i Σ_{j=1}^{k}(Δf_{i,x_{ij}} − Δf_{i,x_{i1}})` (look-ahead so hard requests aren't postponed; if a request fits in fewer than `m−k+1` routes, insert the one with the fewest feasible routes, ties by best insertion cost). Repair can also be optimal CP branch-and-bound + limited discrepancy search (original LNS).
4. **Accept** (simulated annealing): `x ← x_t` if `c(x_t) ≤ c(x)`, else with probability `exp(−(c(x_t)−c(x))/T)`. Update `x_b` if `c(x_t) < c(x_b)`. Cool `T ← c·T`. (`T_start` calibrated so a `w`%-worse solution is accepted w.p. 0.5 ⇒ `T_start = w·c(x₀)/ln 2`.)
5. **Score** the used operators: `σ₁` if new global best, `σ₂` if accepted-and-improving, `σ₃` if accepted-but-worse (only for *unvisited* solutions, tracked via a hash table). Both used operators get the same score. (Tuned example: `(σ₁,σ₂,σ₃) = (33, 9, 13)` — note `σ₃ > σ₂`, rewarding diversifying acceptance.)
6. **Adapt weights** at the end of each segment (e.g. 100 iterations): with `π_i` the total score and `θ_i` the use-count of operator `i` last segment, and reaction factor `r ∈ [0,1]`,
   `w_{i,j+1} = w_{i,j}(1 − r) + r·(π_i / θ_i)`.
   Reset segment statistics. (Per-iteration variant: `w ← λw + (1−λ)ψ` on the two used operators.)
7. Repeat until the stop criterion (iteration/time limit); return `x_b`.

## Code

A self-contained ALNS for a small CVRP: random + worst removal, greedy + regret-2 repair, roulette-wheel adaptive weights with the segmented update, SA acceptance. Prints the improving objective trajectory.

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
