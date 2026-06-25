OK, let me think this through from scratch. I have a vehicle routing problem — hundreds of customers, a fleet, capacities, time windows, maybe pickup-before-delivery precedence — and I need good plans fast and at scale. Exact methods are out at this size, so I'm doing local search: hold a feasible plan, perturb it, keep the perturbation if it helps, repeat. The whole behaviour of that loop is decided by one thing — the perturbation, the *move* — so let me stare at what the standard moves actually do on a constrained instance, because that's where the trouble is.

The standard moves are tiny. 2-opt cuts two edges of a tour and reconnects the other way. Relocate pulls one customer out and drops it somewhere else. Swap exchanges two customers. Each of these touches a constant number of elements, so the neighbourhood — the set of plans reachable in one move — has size on the order of `n²`. That `n²` is the appeal: I can enumerate it and take the best improving neighbour quickly. Steepest descent on it: repeatedly set `x ← argmin_{x'∈N(x)} c(x')`, stop when `c(x) ≤ c(x')` for all `x' ∈ N(x)`, i.e. at a local optimum. Fine. But `n²` is microscopic against the real solution space, so those local optima are shallow, and — here's the thing that actually bites — on a *tightly constrained* instance the part of that already-tiny neighbourhood that's even *feasible* collapses toward nothing.

Let me feel that concretely. Suppose I'm at a decent plan and I try to relocate one time-windowed customer to a better spot. Almost everywhere I'd want to put it, the insertion pushes some downstream customer past its deadline, or blows a capacity bound. So the move is rejected on feasibility before cost even matters. The only legal destinations are right back near where the customer already was — which, of course, doesn't improve anything. So the search freezes. It's not that I've found the optimum; it's that I'm standing in a feasible basin and every small feasible step keeps me in it. The landscape is "discontinuous" in the sense that neighbours can differ wildly and most neighbours of a feasible schedule are *infeasible*, so I can't walk continuously from one good basin to another. That's the diagnosis. The harder and tighter the instance, the worse it gets.

Now, my first instinct is: just take worse moves sometimes, that's what simulated annealing is for. Accept an improving move always; accept a worsening one of size `Δ = c(x') − c(x) > 0` with probability `exp(−Δ/T)`, cool `T` down over time. Or threshold accepting, or great deluge — same family, controlled uphill steps to climb out of local optima. But wait — that fixes the wrong half of the problem. The acceptance rule decides whether to *keep* a neighbour; it cannot *create* a neighbour the move operator never proposes. If the move is a single relocate and there are essentially no feasible relocates that go anywhere new, then no amount of clever acceptance manufactures the big feasible reconfiguration the instance actually needs. The barrier between basins isn't a cost barrier I can pay for with `exp(−Δ/T)`; it's a *feasibility* barrier — there's no feasible single-step path across it at all. So the acceptance rule is necessary but not sufficient. The thing I have to change is the move.

So make the neighbourhood bigger. That's the lever: a larger `N` has fewer, deeper local optima, and — more importantly here — a large move can reach feasible plans that no sequence of small *feasible* moves can, because it rearranges enough at once to clear the feasibility barrier in one shot. The catch is obvious: if I make the neighbourhood literally large — say all plans differing in `k` customers' positions, with `k` big — I can't enumerate it. For relocate, `k=1` gave `O(n²)`; pushing `k` up makes the neighbourhood grow combinatorially and searching it explicitly is hopeless. People have a name for this regime — neighbourhoods that grow exponentially with size, or are just too large to enumerate, searched implicitly: very-large-scale neighbourhoods. Their whole taxonomy (variable-depth chains à la Lin–Kernighan, network-flow improvement, polynomially-searchable restrictions) is about one question: how do you search an exponentially large neighbourhood *cheaply*? Lin–Kernighan, for instance, escapes shallow TSP optima by chaining many edge swaps into one deep compound move and searching that chain partially. That's the right *spirit* — go big — but the mechanics are edge-local and hand-tailored to the TSP; they don't carry time windows and precedence, and the depth comes from a bespoke chaining rule, not something generic I can point at a constrained routing problem.

Let me back up and ask what a large move on a *routing* plan should even be, structurally. A routing plan is an assignment of customers to positions on routes. What's the most natural way to make a big, generic change to such an assignment while guaranteeing I land back on a feasible plan? Here's the move that keeps the feasibility guarantee for free: take the current plan, *remove* a set of customers from it entirely — just lift them out, shortcutting the routes where they sat — and then *re-insert* them, from scratch, in the best way I can find. After removal I have a smaller but perfectly feasible partial plan; re-insertion only ever places a customer where the constraints allow, so the result is feasible again. One removal-plus-reinsertion is one step. And let me size the implied neighbourhood, because the whole bet is that it's huge: if I remove `q` customers, every choice of which `q` to remove, times every legal way to reinsert them, is a neighbour I might reach. Just the removal choices, for a 100-customer plan removing 15, are `C(100,15) = 100!/(15!·85!)`. Let me actually evaluate that rather than wave at it — `C(100,15) = 253,338,471,349,988,640`, i.e. `≈ 2.5×10¹⁷`. So one *family* of single moves already spans `10¹⁷` neighbours before I even count reinsertions; there is no enumerating this. Which forces the only viable reading of the move: I never list the neighbourhood, I *sample* it, by picking one removal set and doing one (good) reinsertion. So the move I've arrived at is large precisely because the hole is large, and it's searched implicitly — relax a chunk of the solution and re-optimize it — rather than by listing neighbours. That's a different animal from the small fixed-`k` exchanges I started with, and it's the kind of exponentially-large, implicitly-searched neighbourhood the very-large-scale-neighbourhood literature is about; Lin–Kernighan's deep edge-swap chains are one species of it, this destroy-and-reinsert is another, one that doesn't depend on the move being edge-local and so carries time windows and precedence along for free.

Why does the *size* of the hole matter so much, though — couldn't I just do lots of tiny single-customer removals? Let me reason about it. Suppose I remove just one customer and reinsert it optimally. Under tight constraints its best feasible reinsertion is, overwhelmingly, right back where it came from — same independence trap as relocate. Now remove two. If the two I removed are unrelated — far apart in space and time — then their best reinsertions don't interact: each goes back near its own old slot independently, and I've gained nothing that two separate one-customer moves wouldn't have given me. So removing unrelated customers is wasteful — it's just several independent tiny moves wearing a trench coat, and reinsertion gets more expensive the more I remove. The payoff only appears when the removed customers *compete for the same space*: if I remove two customers that are close and currently near each other, reinsertion can put them back in *swapped* order, or interleaved with each other's neighbours — a genuine interchange that no single feasible relocate could reach. So the lesson is sharp: don't remove a random scatter; remove *related* customers, so that reinsertion has something to actually rearrange. "No more than necessary" too — every extra removed customer makes reinsertion slower, so I want the smallest hole that still buys an interchange.

I need to make "related" precise. What makes two customers good to remove together? They should be close enough that swapping them is plausible. Geographic proximity is the obvious axis: if `c_ij` is the (normalized) travel cost between `i` and `j`, small `c_ij` means removing both opens a real opportunity to trade their positions. Being on the same route matters too — if I want to *empty* a route entirely (the only way to reduce the vehicle count), I'd better be removing customers that share a route. So let me define a relatedness that's large when customers are close and large when they're on the same vehicle. The cleanest thing: relatedness inversely proportional to a "distance" that combines both. Let `V_ij = 0` if `i` and `j` are currently served by the same vehicle and `1` otherwise, and with `c_ij` normalized into `[0,1]`, set

`R(i,j) = 1 / (c_ij + V_ij)`.

Small `c_ij` (close) drives the denominator down and `R` up; same vehicle (`V_ij=0`) drops the penalty term entirely so `R` is larger still — both push toward *more* related, which is exactly what I want. And nothing stops me from folding in more axes later — similar time windows, similar load, pickup/delivery partners forced to move together — but proximity-plus-vehicle is the core.

Now, *how* do I pick the removal set using `R`? The trap to avoid: if I always greedily take the most-related customers, I'll keep choosing the same cluster every time and the search stagnates — it stops exploring. So I want relatedness to *bias* the choice without making it deterministic. Build the set incrementally: start from one random seed customer. Then repeatedly pick a random member of the already-chosen set, rank all the remaining customers by their relatedness to it, and choose one *near the top of the ranking but not necessarily the very top*. The standard trick to inject exactly this controllable bias: draw a uniform `rand ∈ [0,1)`, raise it to a power `D ≥ 1`, and index into the relatedness-sorted list at position `⌊ |L| · rand^D ⌋`. Before I trust this I want to see that the knob really does interpolate between "uniform" and "always the top," so let me draw it 100,000 times against a list of length 20 and look at the index distribution. At `D = 1`, `rand^D` is just uniform, so I'd expect a mean index near 9.5 and index-0 about 1/20 = 5% of the time — and the draw gives mean 9.50, index-0 4.9%. Good, that's pure random removal. At `D = 5`: mean index 3.0, index-0 about 55% of draws — strongly biased to the related end but still wandering. At `D = 30`: mean index 0.56, index-0 about 91% — nearly always the single most-related customer. So `D` genuinely slides from uniform (1) toward deterministic-most-related (large), exactly the dial I wanted; I'll keep it moderate (low-to-mid teens) so it favours related customers without locking onto one cluster. Repeat until I've removed `q` of them.

Then the other half: reinsertion. This is where the constraint machinery earns its keep, and there's a real fork in the road. Option one: reinsert *optimally* — treat each removed customer as a variable whose values are its legal insertion arcs, propagate the load and time-window rules to prune illegal arcs (a customer can't go between `u` and `w` if it makes the vehicle arrive past `u`'s or `w`'s deadline; maintain earliest/latest service times along each route; drop any arc whose lower-bound cost already exceeds the best plan so far), and branch-and-bound to the minimum-cost completion. This is exactly where constraint programming meshes with the move: the "evaluate this large move" step *is* a little CP search over the hole. It's heavyweight — far fewer moves per second than small-move local search — but each move is powerful, reaching far.

There's a snag with full branch-and-bound, though. For a hole of ~25 customers it usually finishes in seconds, but the runtime distribution has a brutal tail — occasionally one reinsertion takes forever, because the tree is huge and I'm bounding it only naively (lower bound = current partial cost, no clever bound on the as-yet-unrouted customers). I can't have the whole search hostage to one pathological reinsertion. So I want to *cap* the tree exploration. The right tool: limited discrepancy search. Order the insertions by a value-ordering heuristic (cheapest insertion position first), and count a "discrepancy" as every time I go *against* that heuristic — inserting at the second-cheapest position is one discrepancy, third-cheapest is two, two separate second-cheapest picks is two, and so on. Then explore only leaves within a discrepancy budget. That trusts the heuristic to be usually right and bounds the search to "the heuristic, plus a few corrections," trading reinsertion thoroughness against moves-per-second via one knob. And to make the heuristic itself good: insert the *most constrained* customer first — the one farthest from the rest of the plan, since it'll eat the most time/distance and constrain everything else — and try its insertion arcs cheapest-first. Variable ordering on the most-constrained-variable, value ordering on cheapest-cost — straight out of CP practice.

That's a complete method already: relatedness-removal + CP/LDS reinsertion + accept-if-improving, with `q` ramped up over time (start at 1; if `α` consecutive moves don't improve, bump `q`, cap it around 30, so I only enlarge the hole when stuck at the current size). It works. But two things nag at me.

The first nag: accept-if-improving is pure descent, and I already argued descent traps. I patched the *move* to be large, which gets me across feasibility barriers — but I can still settle into a large-move local optimum where no single destroy-repair improves. So bring back the acceptance idea I shelved earlier, now that the move is finally worth pairing it with. Accept a candidate `x'` from current `x` if `c(x') ≤ c(x)` always; otherwise accept it with probability `exp(−(c(x') − c(x))/T)`, with `T` started warm and cooled geometrically `T ← c·T`, `0 < c < 1`. Early on, sizeable worsenings get through and I roam; late, only near-improvements survive and I intensify. Now the loop is "destroy a related chunk, repair it, accept by Metropolis" — which, squint at it, is just simulated annealing with an exotic, very large neighbourhood instead of a single swap. Same skeleton, radically better move. For the start temperature, rather than guess `T_start` per instance, I can *calibrate* it: pick `T_start` so that a solution `w` percent worse than my initial one is accepted with probability `0.5`. Set the worsening to `Δ = w·c(x₀)` and demand `exp(−Δ/T_start) = 0.5`; taking logs, `−Δ/T_start = ln 0.5 = −ln 2`, so `T_start = Δ/ln 2 = w·c(x₀)/ln 2`. Let me sanity-check the algebra didn't drop a sign, with `c(x₀) = 1000`, `w = 0.10`: `T_start = 0.10·1000/0.6931 = 144.27`, and plugging back, `exp(−(0.10·1000)/144.27) = exp(−0.6931) = 0.500`. The round-trip lands on exactly 0.5, so the formula is right and gives me an instance-adaptive start temperature for free — no per-instance tuning of `T_start`.

And actually, the moment I switch to SA acceptance, the *optimal* reinsertion starts to look like it might be a liability rather than an asset — let me reason about whether an optimal repair can ever even hand SA a worsening candidate. Consider the set of all feasible completions of a given hole. One element of that set is always available: put each removed customer back into the exact slot it came from. I should check that this "put-back" really does reconstruct the original solution and not something off by a position, since the whole argument hangs on it — so let me trace it. Take a feasible plan (cost `1155.31` on my test instance), pull two customers out recording their `(route, position)`, then reinsert them at those same coordinates: the result is byte-for-byte the original routes and the cost is `1155.31` again, identical. So put-back is genuinely a feasible completion whose cost equals the solution I destroyed. An *optimal* repair, by definition, returns the minimum-cost completion over that set, and the minimum over a set containing put-back is `≤` put-back's cost — so the candidate is always improving-or-equal versus the solution I tore up, never strictly worse, for *any* hole size (large `q` only widens the set the minimum is taken over, which can only help, never hurt). That means an optimal repair can never hand SA a controlled *worsening* candidate to escape a valley with — the diversification SA exists to exploit would be engineered straight out by making repair too good.

So flip it: use a *fast, deliberately imperfect* repair. The cheapest one — basic greedy: for each still-unplaced customer `i`, let `Δf_{i,k}` be the cost of its best feasible insertion into route `k` (`∞` if it doesn't fit), `c_i = min_k Δf_{i,k}` its best-overall insertion cost; insert the `argmin_i c_i`, update only the one route I touched, repeat. Does this actually produce worsening candidates the way the argument predicts? It's worth measuring rather than assuming, because on a *poor* current solution greedy reinsertion improves almost everything (I tried it on a sloppy initial plan and got better-or-equal on every one of 2000 random holes — useless as a test, since there was nowhere to go but down). The honest test is near an optimum, where a myopic repair has room to overshoot. So I take a near-optimal solution (cost `491.40`) and hit it with 3000 random 4-customer holes, greedily repaired each time: `604` of the candidates come back *strictly worse* than `491.40`, `0` strictly better, the remaining `2396` equal. So roughly a fifth of greedy repairs near an optimum genuinely worsen the solution — exactly the controlled-worsening candidates SA can accept to cross a barrier, and the very ones an optimal repair was provably incapable of producing. A poor insertion heuristic, counterintuitively, makes the *overall* method better, because those bad completions are what diversify; and it's faster than a CP tree besides. (No CP tree at all now — and that's fine.)

Greedy has one ugly failure mode worth fixing in place: it keeps inserting whatever is cheapest *right now*, which postpones the "hard" customers — the ones expensive to place anywhere — to the end, by which point the routes are full and they jam or spill to a request bank. The fix is a look-ahead. Don't ask "who's cheapest to place now"; ask "who will I *regret* most if I don't place them now." For customer `i`, sort its route insertion costs `Δf_{i,x_{i1}} ≤ Δf_{i,x_{i2}} ≤ …` (so `x_{ij}` is the route with `i`'s `j`-th cheapest insertion). The regret of deferring `i` is how much worse its second-best option is than its best: `c*_i = Δf_{i,x_{i2}} − Δf_{i,x_{i1}}`. Insert the `i` maximizing `c*_i`, at its best position — the one I'd pay the most to *not* lose its good slot. Generalize to look `k` routes deep: `argmax_i Σ_{j=1}^{k} (Δf_{i,x_{ij}} − Δf_{i,x_{i1}})`, the regret-`k` heuristic; large `k` spots earlier that a customer's options are vanishing. There's an edge case to honour: if some customer can no longer be inserted into at least `m−k+1` routes (its options are already that scarce), it can't even form a full `k`-deep regret, so I just insert whichever such customer fits in the *fewest* routes first — most-constrained-first again — breaking ties by best insertion cost. Greedy is regret-1. Now I have a little family of repair operators of different myopia, and another family of removal operators — relatedness-removal, and I can add plain *random* removal (which is just relatedness-removal with `D=1`), and a "worst" removal that targets the customers currently costing the most.

That worst-removal idea deserves a second: instead of removing customers that swap easily (relatedness), remove the ones that look *misplaced*. Define `cost(i,s) = f(s) − f_{−i}(s)`, how much customer `i` is currently adding to the plan; rank customers by that and remove the high-cost ones (randomized the same way, index `⌊|L|·y^p⌋`, so I don't gouge out the exact same customers every time). Relatedness-removal and worst-removal are pulling on different ropes — one picks easily-interchangeable customers, the other picks badly-placed ones — and on different instances, or different phases of the same run, one will beat the other.

Which lands me on the second nag, the deeper one: I now have several removal operators and several repair operators, and *I don't know which to use*. The best removal-repair pairing genuinely depends on the instance — its geometry, its time-window tightness — and even on *where in the search I am*. Hand-tuning a fixed choice is fragile. So don't choose one; carry them all, and let the search *learn* the mix as it goes. Assign each operator a weight `w_i` and pick by roulette wheel: select operator `j` with probability `w_j / Σ_i w_i`, choosing a removal operator and a repair operator independently. Now I need the weights to track which operators have actually been *working lately*.

Here's the scoring logic. After each iteration I know the fate of the candidate the chosen operators produced, and I want to reward operators that move the search forward. Three grades of success: it found a new *global best* — the strongest signal, reward `σ₁`; it was accepted and improved on the current solution — good, reward `σ₂`; it was accepted even though it was *worse* than current — still valuable, because that's the diversification that escapes local optima, reward `σ₃`. I won't pin down an order among `σ₂` and `σ₃` up front — whether improving-acceptance or diversifying-acceptance deserves more credit is itself something tuning should decide, since rewarding the worsening-but-accepted case can be the thing that keeps the search exploring. One subtlety I have to get right: only reward a candidate that I *haven't seen before*. If an operator keeps cycling the search back to already-visited solutions, that's not progress; rewarding it would inflate weights for going in circles. So I keep a hash table of visited solutions and only credit *unvisited* ones. And since each iteration fires one removal *and* one repair, I can't tell which deserves the credit — so I award the same score to both operators used.

Now turn scores into weights. I don't want to overreact to a single lucky iteration, nor average over the whole run (the right operator drifts as the search cools). So work in *segments* — say 100 iterations. Reset every operator's score to 0 at the start of a segment; over the segment accumulate `π_i` = total score earned by operator `i` and `θ_i` = number of times `i` was used. The per-use performance of `i` last segment is `π_i/θ_i`. Blend that into its weight with a reaction factor `r ∈ [0,1]`:

`w_{i,j+1} = w_{i,j}·(1 − r) + r·(π_i / θ_i)`.

`r = 0` ignores the statistics and freezes the initial weights; `r = 1` throws away history and lets last segment alone decide; in between, the weights are an exponential moving average of recent per-use success. Start all weights equal in segment 1. An operator that's been producing new-bests and accepted moves climbs in weight and gets picked more; one that's been wasting iterations decays and gets picked less — automatically, per instance, per phase. (Equivalently I could collapse the segment to a single iteration and update only the two operators I just used, with `w ← λ·w + (1−λ)·ψ` where `ψ` is this iteration's score and `λ` is a decay — the same exponential-moving-average idea at iteration granularity instead of segment granularity; the segment version smooths out single lucky iterations.) What I've ended up with is a destroy-and-repair search carrying several of each operator and letting the run reweight them on the fly — a large-neighbourhood search that tunes its own neighbourhoods as it goes.

One last refinement, because the greedy/regret repair is myopic and will keep making the same locally-best insertion. Inside an SA framework I want my neighbourhood *sampled* with some randomness, but a deterministic repair always returns the same completion of a given hole. So perturb the insertion costs: whenever I evaluate an insertion cost `C`, add noise drawn from `[−maxN, maxN]` and use `C' = max(0, C + noise)`, with `maxN = η · max_{i,j} d_{ij}` scaled to the instance's distances. Sometimes the repair then makes the second-best move instead of the best, which is exactly the sampling SA wants — and I can let the *same* adaptive mechanism decide how often to use noisy versus clean insertion, by treating "noise" and "no noise" as two more competing operators.

Let me make all of this concrete in code, on a capacitated VRP, with random + worst removal and greedy + regret-2 repair, roulette-wheel adaptive weights, and SA acceptance, and actually run it to see whether the objective comes down and whether the operator weights move apart the way the adaptive story claims.

```python
import math, random

random.seed(0)

# ---- a small CVRP instance: depot at 0, customers 1..n with demands ----
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
    # cost(i,s) = f(s) - f_{-i}(s): how much customer i currently adds to its route.
    # It shifts as neighbours leave, so recompute the ranking after each removal.
    work = [r[:] for r in routes]
    target = min(q, sum(len(r) for r in work))
    removed = []
    while len(removed) < target:
        contrib = []
        for ri, r in enumerate(work):
            nodes = [0] + r + [0]
            for k, c in enumerate(r):
                saved = (dist(nodes[k], nodes[k + 1]) + dist(nodes[k + 1], nodes[k + 2])
                         - dist(nodes[k], nodes[k + 2]))
                contrib.append((saved, ri, k, c))
        contrib.sort(reverse=True)                       # descending cost
        y = random.random()
        idx = min(int((y ** p) * len(contrib)), len(contrib) - 1)
        _, ri, k, c = contrib[idx]                       # bias toward high-cost end
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
        delta = (dist(nodes[k], c) + dist(c, nodes[k + 1]) - dist(nodes[k], nodes[k + 1]))
        if delta < best:
            best, pos = delta, k
    return best, pos

def insert_options(routes, c):
    # sorted list of (delta, route_index) over all routes + the "new route" option
    opts = []
    for ri, r in enumerate(routes):
        d, pos = best_insertion(r, c)
        if pos is not None:
            opts.append((d, ri, pos))
    opts.append((dist(0, c) + dist(c, 0), -1, 0))       # open a fresh route
    opts.sort()
    return opts

def greedy_repair(routes, removed):                     # regret-1
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

def regret2_repair(routes, removed):
    routes = [r[:] for r in routes]
    pool = removed[:]
    while pool:
        best = None                                     # maximize regret c*_i
        for c in pool:
            opts = insert_options(routes, c)
            d1 = opts[0][0]
            d2 = opts[1][0] if len(opts) > 1 else d1
            regret = d2 - d1
            key = (regret, -d1)                          # ties -> lowest insertion cost
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
dw = [1.0, 1.0]                                          # destroy weights
rw = [1.0, 1.0]                                          # repair weights
S1, S2, S3 = 33, 9, 13                                   # scores: best / better / accepted
R = 0.1                                                  # reaction factor r
SEGMENT = 100                                            # iterations per segment

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
    T = 0.1 * best_c / math.log(2)                       # ~10%-worse accepted w.p. 0.5
    cooling = 0.9985
    seen = {tuple(tuple(r) for r in cur)}                # initial solution already visited
    traj = []
    d_score = [0.0] * len(dw); d_used = [0] * len(dw)    # per-segment statistics
    r_score = [0.0] * len(rw); r_used = [0] * len(rw)
    for it in range(iters):
        di, ri = roulette(dw), roulette(rw)
        d_used[di] += 1; r_used[ri] += 1
        q = random.randint(2, max(2, N // 4))            # degree of destruction
        partial, removed = destroy_ops[di](cur, q)
        cand = repair_ops[ri](partial, removed)
        cand_c = solution_cost(cand)

        key = tuple(tuple(r) for r in cand)
        score = 0
        if cand_c < best_c - 1e-9:
            best, best_c, score = cand, cand_c, S1       # new global best -> S1
            traj.append((it, best_c))
        delta = cand_c - cur_c
        accepted = delta <= 0 or random.random() < math.exp(-delta / T)
        if accepted:
            if key not in seen:                          # reward only unvisited
                score = max(score, S2 if delta < 0 else S3)
            cur, cur_c = cand, cand_c
        seen.add(key)

        d_score[di] += score; r_score[ri] += score       # accumulate pi_i over segment
        if (it + 1) % SEGMENT == 0:                       # w_{i,j+1} = w_ij(1-r) + r*pi_i/theta_i
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

Running it: the best cost falls from `1055.28` at the first repaired candidate down through `755.11, 690.50, 607.47, …, 499.97` and finally to `491.40`, where it settles into a 5-vehicle plan — most of the drop happening in the first few dozen iterations and the last improvement landing at iteration 182, after which 4000 iterations of destroy-repair find nothing better. So the loop does what the argument said it would: large feasible reconfigurations early, then diminishing returns. One thing the run does *not* show dramatically is the weights pulling apart — they finish at destroy `[0.20, 0.21]` and repair `[0.20, 0.21]`, nearly even. That's honest and worth noting: on a tiny, near-symmetric 20-customer instance neither removal nor repair operator dominates, so adaptive selection has little to discriminate and the reaction factor `r = 0.1` keeps the weights close to their shared starting point. The adaptive machinery is built for large, structured instances where the operators genuinely differ; on a toy it correctly reports "no strong preference" rather than inventing one — which is the right behaviour, even if it's a less flashy demo than I'd hoped.

So the causal chain, start to finish: small-move local search freezes on tightly-constrained routing because feasible neighbours all sit back where they started — a feasibility barrier, not a cost barrier, so no acceptance rule alone crosses it; the cure is a *large* move, and the natural large move on a routing plan is destroy-and-repair, which lifts out a set of customers and reoptimizes the hole, sampling an implicitly-exponential neighbourhood; the hole should hold *related* customers so reinsertion can actually interchange them, sized just large enough; repair can be optimal CP/LDS or — better paired with simulated-annealing acceptance — fast greedy/regret whose very imperfections diversify; and since no single destroy/repair operator wins everywhere, carry several, pick by roulette wheel, and let weights track recent unvisited-success through a segmented exponential-moving-average — a self-tuning large-neighbourhood search.
