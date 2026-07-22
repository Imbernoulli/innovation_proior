# Reasoning: Tower Placement for Signal Coverage

## Understanding the objective

I read the instance format first and translated the words into an integer program. There are `n`
demand nodes and `m` candidate tower sites. Building a tower at site `j` costs one unit and delivers
a fixed, pre-computed power `P[i][j] >= 0` to every demand `i`. A demand is satisfied once the
*sum* of powers from the towers I build reaches its requirement `req[i]`. So I want

```
minimize    sum_j x_j
subject to  sum_j P[i][j] x_j >= req[i]   for all i,
            x_j in {0,1}.
```

This is a 0/1 *covering* program — specifically a set-multicover, because each constraint is a
weighted threshold rather than a simple "covered or not". It is NP-hard, so I am not chasing the
optimum; I am chasing the fewest towers I can defensibly find, and I am judged on a continuous score
that rewards using fewer towers than a reference baseline. The one hard rule I must never break:
if even a single demand ends up under its requirement, the whole output is infeasible and scores
zero. That feasibility floor dominates everything — a clever solution that misses one demand by a
hair is worth less than the dumbest feasible one.

Two facts about the objective shape the whole approach. First, "received signal" is **additive and
monotone**: adding a tower can only raise every `recv[i]`, never lower it. So feasibility is monotone
in the built set — supersets of a feasible set are feasible. Second, the obvious quantity to track is
the per-demand accumulator `recv[i] = sum over built j of P[i][j]`; with it, the feasibility test is
just "is `recv[i] >= req[i]` for all i", and the marginal value of a candidate tower is cheap to
evaluate. I decided early that every phase of the solver would maintain `recv[]` incrementally rather
than recompute coverage from scratch — that is the difference between a toy and something that can
run many local-search moves in the time budget.

## A feasible baseline first

Before any cleverness I needed a solution that is *always* valid, because the scorer floors me to
zero otherwise and I want a non-zero number from the very first run. The generator guarantees that
building **every** site satisfies every demand (each `req[i]` is set to a fraction of the all-sites
total). So "build all `m` sites" is a trivially feasible answer. That is my safety net: any time my
heuristic somehow fails to reach feasibility, I fall back to opening every remaining site, which
cannot leave a demand short. I wired this fallback in from the start so that I literally cannot emit
an infeasible output — feasibility is structurally guaranteed, and only the *count* is up to the
heuristic.

Of course "build everything" is terrible on the count. On the seed set it builds ~90-140 towers
where a good solution needs ~50-70. So the baseline establishes feasibility, and now the entire game
is shrinking the tower count while staying feasible.

## The natural strong heuristic, and why I want to beat it

The textbook heuristic for covering is **greedy**: repeatedly build the site that reduces the total
remaining deficit `sum_i max(0, req[i] - recv[i])` the most, until the deficit hits zero. This is the
classic `H_n`-approximation, and it is genuinely strong here. The marginal deficit-reduction of a
site, `gain(j) = sum_i min(req[i] - recv[i], P[i][j])` over still-unmet demands, is **submodular**:
it only goes down as I build more towers. That means I can use a *lazy greedy* (CELF) priority
queue — keep cached upper bounds on each site's gain, and only recompute the gain of the site at the
top of the queue; if its fresh gain still beats the next cached bound, accept it, else re-insert it
with the fresh (smaller) value. Submodularity guarantees the cached value is a valid upper bound, so
this returns exactly the greedy pick while recomputing far fewer gains. That is the right way to make
greedy fast.

But greedy is myopic: it commits to whichever site looks best *right now*, and on covering problems
that can lock in one tower too many. The scorer normalizes against exactly this greedy baseline
(greedy scores 1.0), so to score above 1.0 I need something that systematically improves on the
greedy ordering. The lever the problem points at is the **LP relaxation**.

## The innovation: relax, round, repair, reduce

Drop the integrality constraint to get the covering LP `min 1^T x` s.t. `P x >= req`,
`0 <= x_j <= 1`. Its fractional optimum `xf[j]` is informative: a site the LP opens to near `1` is
structurally important (some demand really needs it), while a site left near `0` is one the LP would
rather avoid paying for. Rounding *guided by the LP* tends to pick the structurally important sites
that pure deficit-greedy might pass over, and that is how I get below the greedy count.

I did not want to ship a full simplex/interior-point LP in a single file — overkill and fragile.
A covering LP of this size is solved well enough by a **multiplicative-weights / fractional
primal-dual** scheme (the Young / Plotkin-Shmoys-Tardos style for covering LPs). I maintain dual
weights `y[i]` on the demands that emphasize the ones still far from their requirement (I use the
squared relative deficit `(deficit_i / req_i)^2`, which sharply focuses pressure on the worst-served
demands). Each round I find the site that best serves the current weighted unmet mass,
`argmax_j sum_i y[i] * min(req[i]-recv[i], P[i][j])`, and open a fractional step of it (half-open,
clipped so `xf[j] <= 1`), updating `recv` by that fraction. After a few hundred rounds the weights
collapse on the demands that are genuinely hard to cover and `xf` settles into a good fractional
profile. I am not solving the LP to optimality — I only need `xf` good enough to *rank* sites, and
this gives me that cheaply and robustly.

With `xf` in hand the pipeline is:

1. **Greedy rounding by coverage-per-cost.** Sort sites by `key = (xf[j] + 1e-3) * (coverage[j]+1)`
   — LP importance times raw reach — and walk down the list opening a site only if it still helps an
   unmet demand, until everything is served. The `+1e-3` keeps a sensible coverage ordering even
   where the LP left a site at zero.
2. **Lazy-greedy (CELF) repair.** A rounding can stop just short on a stubborn demand. Whatever is
   left unmet, I fill with the lazy submodular greedy described above — the exact same machinery as
   the standalone greedy baseline, but only on the residual.
3. **Redundancy prune.** After construction some towers are no longer needed (later picks made an
   earlier one redundant). I try to drop opened sites, least-coverage first (most likely redundant),
   removing any whose removal keeps every demand satisfied. This is pure win on the count.
4. **Multi-start.** I run the rounding several times from *perturbed* LP-guided orders (a small
   multiplicative jitter on the key) and keep the smallest feasible set. Different tie-breaks explore
   different roundings; the prune cleans each up; the best survives.
5. **remove-2-add-1 local search.** Greedy and rounding both get stuck in local optima the prune
   can't escape, because no single tower is removable but a *pair* could be replaced by *one* other
   tower. So I search for `(a, b) opened, c closed` such that removing `a, b` and adding `c` keeps
   every demand met — a net reduction of one tower. I order opened sites by least coverage (likeliest
   to drop) and closed sites by most coverage (likeliest to cover the gap) so the first improving
   move is found fast. Each accepted swap is followed by another prune, and I iterate until no move
   improves or the time budget runs out.

Every test in every phase reads `recv[]` and the relevant site column, so a feasibility check is
`O(n)` and a swap test is `O(n)` — cheap enough to run thousands of times.

## Implementing it, then a real debug episode

I wrote the first version as a single pass: LP → one rounding → repair → one prune, no multi-start,
no swap search. It compiled, every seed was feasible (the all-on fallback held), and it beat the
trivial all-sites baseline by a mile. But when I scored it against the *greedy* baseline that the
scorer normalizes to, the result was deflating:

```
seed  solK  greedK  sol_sc
2     53    53      1.000000
7     53    52      0.981132   <- WORSE than greedy
15    72    71      0.986111   <- WORSE than greedy
...
mean score = 1.0122
```

On two seeds my "smarter" pipeline actually used *more* towers than plain greedy, and on several
others it merely tied. That is the classic failure mode of single-shot LP rounding: the rounded set
lands in essentially the same basin as greedy, and a one-pass prune can't escape it. The whole point
of the innovation — beating greedy — was not happening reliably.

I diagnosed it by dumping, on seed 7, the set my solver built versus the greedy set. They differed by
only a couple of sites, and in both there was a pair of low-coverage towers near a cluster boundary
that *together* covered a handful of edge demands that one well-placed central tower could also cover.
Neither of those two towers was individually removable (each was the sole supplier of signal to one
edge demand), so the prune left both in. That is exactly the situation a `remove-2-add-1` swap fixes
and a removal-only prune cannot. It confirmed the fix: I needed the pairwise swap local search, and I
needed multi-start so a single unlucky rounding wouldn't decide the whole answer.

I added the multi-start loop (perturbed orders, keep the best) and the `reduce2for1` swap search
alternating with re-pruning, all under the time budget. Re-running the seed set:

```
seed  solK  greedK  sol_sc
2     53    53      1.000000
7     52    52      1.000000   <- now ties greedy (was 0.98)
15    69    71      1.028986   <- now beats greedy (was 0.99)
...
mean score = 1.0273   (was 1.0122)
```

Now **no seed scores below 1.0** — the solver never loses to greedy — and the mean climbed to 1.027,
with the best seeds around 1.05 (5% fewer towers than greedy). Against the trivial all-sites baseline
the gap is enormous: solver mean score 1.027 vs the all-sites baseline's 0.54 (it uses roughly twice
as many towers as greedy, so it scores about half). I also checked the feasibility floor directly:
an empty output, a single-tower output, a duplicate index, an out-of-range index, and a garbage
output all score exactly 0, and a valid output never does. And the solver is deterministic — fixed
RNG seed, identical output across repeated runs on the same instance — which matters for a frozen
scorer. Wall-clock per instance is ~25-40 ms, far under the budget, because the swap search converges
quickly at these sizes; the `TIME_LIMIT` guard only matters as insurance on harder instances.

The remaining sub-1.0 risk I deliberately closed: the all-on safety net means even a pathological LP
or a starved time budget still yields a feasible answer, just a worse-counted one — never a zero.
That ordering of priorities (feasibility first, then count) is the right one for a problem with a
hard zero floor.

## Final solver

```cpp
// Minimise the number of chosen tower sites so that every demand node i
// receives accumulated signal >= req[i], where a site j delivers power
// P[i][j] to demand i.  This is the covering ILP
//        min sum_j x_j   s.t.   sum_j P[i][j] x_j >= req[i],  x_j in {0,1}.
//
// Pipeline (the innovation):
//   1. Solve the COVERING LP RELAXATION (0<=x_j<=1) approximately with a
//      multiplicative-weights / fractional primal-dual solver -> fractional
//      site values xf[j] in [0,1].
//   2. GREEDY ROUNDING guided by "coverage-per-cost": process sites in order of
//      a score combining the LP fractional weight and current marginal
//      deficit-reduction, opening sites until all demands are met.  We do this
//      from several randomised perturbations of the LP-guided order and keep
//      the smallest feasible set (multi-start LP rounding).
//   3. LAZY-GREEDY REPAIR: if a rounding leaves a demand short, repeatedly add
//      the site with the largest remaining deficit-reduction gain, using a lazy
//      upper-bound priority queue (CELF) so few marginal gains are recomputed.
//   4. LOCAL SEARCH: redundancy pruning, plus remove-2-add-1 / remove-1-add-0
//      reductions to escape the pure-greedy local optimum and shrink the count.
//
// Output: k  then the k chosen 0-based site indices.  An all-on safety net
// guarantees we ALWAYS emit a feasible solution within the time budget.

#include <bits/stdc++.h>
using namespace std;

static int N, M;
static vector<double> req;          // req[i]
static vector<vector<double>> Pji;  // column-major: Pji[j][i] = power site j -> demand i

static const double EPS = 1e-7;

// timing
static chrono::steady_clock::time_point T0;
static double TIME_LIMIT = 3.5;     // seconds, comfortably under any 5s/10s judge
static double elapsed() {
    return chrono::duration<double>(chrono::steady_clock::now() - T0).count();
}

// rng
static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static inline uint64_t xrand() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state;
}
static inline double urand() { return (xrand() >> 11) * (1.0 / 9007199254740992.0); }

// ---- read instance ---------------------------------------------------------
static void readInput() {
    if (scanf("%d %d", &N, &M) != 2) { N = 0; M = 0; return; }
    req.assign(N, 0.0);
    for (int i = 0; i < N; i++) { if (scanf("%lf", &req[i]) != 1) req[i] = 0; }
    Pji.assign(M, vector<double>(N, 0.0));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) {
            double v; if (scanf("%lf", &v) != 1) v = 0; Pji[j][i] = v;
        }
}

// total raw coverage of a site (sum over demands)
static vector<double> siteCov;
static void precompute() {
    siteCov.assign(M, 0.0);
    for (int j = 0; j < M; j++) {
        double s = 0; for (int i = 0; i < N; i++) s += Pji[j][i];
        siteCov[j] = s;
    }
}

// ---- covering-LP relaxation via multiplicative weights ---------------------
// Approximately solve  min 1^T x  s.t.  P x >= req, 0<=x<=1, returning xf.
// Dual weights y[i] emphasise still-unmet demands; each round opens a small
// fraction of the site that best serves the weighted unmet mass.
static vector<double> solveLP_MW() {
    vector<double> xf(M, 0.0);
    vector<double> recv(N, 0.0);
    const int ROUNDS = 240;
    for (int r = 0; r < ROUNDS; r++) {
        double wsum = 0.0;
        static vector<double> y; y.assign(N, 0.0);
        for (int i = 0; i < N; i++) {
            double deficit = req[i] - recv[i];
            if (deficit < 0) deficit = 0;
            double rel = (req[i] > EPS) ? deficit / req[i] : 0.0;
            y[i] = rel * rel + 1e-9;
            wsum += y[i];
        }
        if (wsum < 1e-9) break;
        int bestJ = -1; double bestScore = -1.0;
        for (int j = 0; j < M; j++) {
            if (xf[j] >= 1.0 - 1e-9) continue;
            const vector<double>& col = Pji[j];
            double sc = 0.0;
            for (int i = 0; i < N; i++) {
                if (y[i] <= 0) continue;
                double need = req[i] - recv[i];
                if (need <= 0) continue;
                sc += y[i] * min(need, col[i]);
            }
            if (sc > bestScore) { bestScore = sc; bestJ = j; }
        }
        if (bestJ < 0 || bestScore <= EPS) break;
        double step = 0.5, room = 1.0 - xf[bestJ];
        if (step > room) step = room;
        xf[bestJ] += step;
        const vector<double>& col = Pji[bestJ];
        for (int i = 0; i < N; i++) recv[i] += step * col[i];
    }
    return xf;
}

// ---- feasibility helper ----------------------------------------------------
static inline bool allMet(const vector<double>& recv) {
    for (int i = 0; i < N; i++) if (recv[i] < req[i] - EPS) return false;
    return true;
}

// marginal deficit-reduction gain of site j given current recv
static inline double marginalGain(int j, const vector<double>& recv) {
    const vector<double>& col = Pji[j];
    double g = 0.0;
    for (int i = 0; i < N; i++) {
        double need = req[i] - recv[i];
        if (need > 0) g += (need < col[i] ? need : col[i]);
    }
    return g;
}

// ---- one rounding+repair run from a (possibly perturbed) LP-guided order ----
// Returns an `open` mask and fills recv; always feasible (all-on fallback).
static vector<char> buildOnce(const vector<double>& xf, double noise) {
    vector<char> open(M, 0);
    vector<double> recv(N, 0.0);

    // priority key = (LP value + small const) * coverage, with optional noise.
    vector<pair<double,int>> order(M);
    for (int j = 0; j < M; j++) {
        double key = (xf[j] + 1e-3) * (siteCov[j] + 1.0);
        if (noise > 0) key *= (1.0 + noise * (urand() - 0.5));
        order[j] = { key, j };
    }
    sort(order.begin(), order.end(), greater<pair<double,int>>());

    for (auto& pr : order) {
        if (allMet(recv)) break;
        int j = pr.second;
        if (marginalGain(j, recv) <= EPS) continue;   // would not help any unmet demand
        open[j] = 1;
        const vector<double>& col = Pji[j];
        for (int i = 0; i < N; i++) recv[i] += col[i];
    }

    // lazy-greedy CELF repair for any residual shortfall
    if (!allMet(recv)) {
        priority_queue<tuple<double,int,int>> pq;
        vector<int> ver(M, 0);
        for (int j = 0; j < M; j++) {
            if (open[j]) continue;
            double g = marginalGain(j, recv);
            if (g > EPS) pq.push(make_tuple(g, j, 0));
        }
        while (!allMet(recv) && !pq.empty()) {
            auto top = pq.top(); pq.pop();
            int j = get<1>(top), v = get<2>(top);
            if (open[j] || v != ver[j]) continue;
            double fresh = marginalGain(j, recv);
            if (fresh <= EPS) continue;
            if (!pq.empty() && fresh < get<0>(pq.top()) - 1e-12) {
                ver[j]++; pq.push(make_tuple(fresh, j, ver[j])); continue;
            }
            open[j] = 1;
            const vector<double>& col = Pji[j];
            for (int i = 0; i < N; i++) recv[i] += col[i];
        }
    }

    // all-on safety net
    if (!allMet(recv)) {
        for (int j = 0; j < M; j++) if (!open[j]) {
            open[j] = 1;
            const vector<double>& col = Pji[j];
            for (int i = 0; i < N; i++) recv[i] += col[i];
        }
    }
    return open;
}

// recompute recv for a mask
static vector<double> recvOf(const vector<char>& open) {
    vector<double> recv(N, 0.0);
    for (int j = 0; j < M; j++) if (open[j]) {
        const vector<double>& col = Pji[j];
        for (int i = 0; i < N; i++) recv[i] += col[i];
    }
    return recv;
}

// redundancy pruning: drop opened sites (least coverage first) while feasible
static void prune(vector<char>& open, vector<double>& recv) {
    vector<pair<double,int>> ord;
    for (int j = 0; j < M; j++) if (open[j]) ord.push_back({ siteCov[j], j });
    sort(ord.begin(), ord.end());
    for (auto& pr : ord) {
        int j = pr.second;
        const vector<double>& col = Pji[j];
        bool ok = true;
        for (int i = 0; i < N; i++) if (recv[i] - col[i] < req[i] - EPS) { ok = false; break; }
        if (ok) { open[j] = 0; for (int i = 0; i < N; i++) recv[i] -= col[i]; }
    }
}

// remove-2-add-1 local search: try to replace two opened sites by one closed
// site and stay feasible (a net reduction of one tower).  Bounded effort.
static bool reduce2for1(vector<char>& open, vector<double>& recv) {
    vector<int> opened, closed;
    for (int j = 0; j < M; j++) (open[j] ? opened : closed).push_back(j);
    // order opened by least coverage (most droppable), closed by most coverage
    sort(opened.begin(), opened.end(), [](int a, int b){ return siteCov[a] < siteCov[b]; });
    sort(closed.begin(), closed.end(), [](int a, int b){ return siteCov[a] > siteCov[b]; });

    int OP = opened.size();
    for (int x = 0; x < OP && elapsed() < TIME_LIMIT; x++) {
        int a = opened[x];
        for (int y = x + 1; y < OP; y++) {
            int b = opened[y];
            // recv after removing a and b
            const vector<double>& ca = Pji[a];
            const vector<double>& cb = Pji[b];
            for (int c : closed) {
                if (c == a || c == b) continue;
                const vector<double>& cc = Pji[c];
                bool ok = true;
                for (int i = 0; i < N; i++) {
                    double r = recv[i] - ca[i] - cb[i] + cc[i];
                    if (r < req[i] - EPS) { ok = false; break; }
                }
                if (ok) {
                    open[a] = 0; open[b] = 0; open[c] = 1;
                    for (int i = 0; i < N; i++) recv[i] += cc[i] - ca[i] - cb[i];
                    return true;   // one improvement; caller re-runs
                }
            }
        }
    }
    return false;
}

int main() {
    T0 = chrono::steady_clock::now();
    readInput();
    if (N == 0) { printf("0\n"); return 0; }
    precompute();

    vector<double> xf = solveLP_MW();

    // multi-start LP rounding: deterministic order first, then perturbed.
    vector<char> best = buildOnce(xf, 0.0);
    {
        vector<double> rv = recvOf(best);
        prune(best, rv);
    }
    int bestCount = (int)count(best.begin(), best.end(), (char)1);

    int restart = 0;
    while (elapsed() < TIME_LIMIT * 0.55 && restart < 40) {
        restart++;
        double noise = 0.15 + 0.35 * urand();
        vector<char> cand = buildOnce(xf, noise);
        vector<double> rv = recvOf(cand);
        prune(cand, rv);
        int c = (int)count(cand.begin(), cand.end(), (char)1);
        if (c < bestCount) { best = cand; bestCount = c; }
    }

    // local search on the best: alternate 2-for-1 reductions and re-pruning.
    {
        vector<double> recv = recvOf(best);
        while (elapsed() < TIME_LIMIT) {
            bool improved = reduce2for1(best, recv);
            if (!improved) break;
            prune(best, recv);
        }
        // final prune guarantee
        prune(best, recv);
        bestCount = (int)count(best.begin(), best.end(), (char)1);
    }

    // emit
    vector<int> chosen;
    for (int j = 0; j < M; j++) if (best[j]) chosen.push_back(j);
    printf("%d", (int)chosen.size());
    for (int j : chosen) printf(" %d", j);
    printf("\n");
    return 0;
}
```
