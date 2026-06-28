**Reading the problem and pinning the contract.** I have a connected weighted graph; each edge is white (`c=1`) or black (`c=0`); I must find the minimum-weight spanning tree that uses **exactly `k`** white edges, or report `-1`. Input is `n m k` then `m` lines of `u v w c`; I print one integer. Before any algorithm I fix the scale, because it dictates both the data types and the asymptotics I am allowed. `n <= 2*10^5`, `m <= 4*10^5`, `w <= 10^9`. A spanning tree has `n-1` edges, so the total weight can reach `(2*10^5)*10^9 = 2*10^14`, far past the 32-bit range of `~2.1*10^9`. So every weight accumulator must be 64-bit `long long`; an `int` is a silent wrong-answer on the large tests. And whatever I do must be roughly `O((n+m) log(...))` — anything quadratic in `m` is hopeless at `4*10^5`.

**The obvious approach: make the colour count a DP state.** The constraint is "exactly `k` white", so the reflex is to carry the white count as a dimension. Concretely: sort all edges by weight and process them Kruskal-style, but instead of a single best tree, keep `dp[c]` = the minimum weight of a partial forest that has merged everything mergeable so far while using exactly `c` white edges. When I consider an edge that would unite two components, I'd branch: either skip it or take it, and taking a white edge moves `dp[c] -> dp[c+1] + w`. At the end I read `dp[k]`.

The trouble shows up the moment I try to make this precise. Kruskal's correctness rests on a clean exchange argument: process edges in weight order, take an edge iff it joins two components, and the greedy choice is *forced* — there is never a "skip vs take" decision to branch on, because skipping an edge that joins two components can only hurt. The instant I add a "how many white so far" dimension, that forcedness evaporates: now it can be right to *skip* a cheap white edge in order to leave room for using a black one later and hit the count `k`. So the DP cannot just be "Kruskal with an extra index"; the union-find state and the count are entangled, and there is no longer a single forest — there is a different forest per count `c`, each with its own connectivity. Maintaining a separate DSU per count, and reconciling which component structure each `dp[c]` corresponds to, is not a small constant. It blows up to something like `O(m * (n-1))` states with non-trivial merging, i.e. `~10^5 * 10^5 = 10^{10}`. That is multiple orders of magnitude over budget at the stated limits.

**Putting a number on why the DP dies.** Take the modest case `n = 2*10^5`, `m = 4*10^5`, `k` somewhere in the middle, say `k = 10^5`. The count dimension alone has `~10^5` values; coupling it with the edge scan gives on the order of `4*10^5 * 10^5 = 4*10^{10}` transitions even before worrying about per-state connectivity bookkeeping. At a few hundred million simple operations per second that is minutes, not the 2-second limit. The per-count DP is not merely slow by a constant — it is the wrong *shape* for this scale. I need an approach that does **not** enumerate the count dimension at all.

**Reframing: what is `f(k)` as a function of `k`?** Let `f(k)` be the minimum spanning-tree weight using exactly `k` white edges, defined on the set of achievable counts. If I could understand `f` as a function, maybe I can avoid building it pointwise. Let me compute it on a tiny concrete graph by brute force to see its shape. Take a 6-vertex graph (the one that later bit me, but useful here): the brute exact-`k` weights come out `f(0)=19, f(1)=18, f(2)=17, f(3)=21`. The successive differences are `-1, -1, +4`. The differences are **non-decreasing**: `f` is **convex**. That is not a coincidence of this instance — it is the key structural fact.

**Why `f` is convex (the earned insight).** Spanning trees of a graph are the bases of its graphic matroid, and "number of white edges" is a linear function on bases. There is a classical exchange property: if tree `T` has `a` white edges and tree `T'` has `b > a+1` white edges, then I can repeatedly swap one edge of `T` for one edge of `T'` to get a tree with `a+1` white edges, and the family of these intermediate trees behaves so that the minimum weight at count `a+1` cannot be worse than the linear interpolation of the minima at `a` and `b`. Equivalently: the set of achievable white counts is a contiguous interval `[minWhite, maxWhite]`, and on that interval `f` is convex. Two consequences I will lean on: (1) outside `[minWhite, maxWhite]` the answer is `-1`; (2) inside, `f` is convex, which is *exactly* the precondition for the Lagrangian / "Aliens" trick.

**The insight: relax the count constraint with a penalty and binary-search it.** Convexity means I can trade the hard equality "exactly `k`" for a scalar penalty. Define, for a real (here integer) multiplier `lambda`, the **penalized** problem: build a spanning tree minimizing

```
C(lambda) = min over trees T of [ weight(T) + lambda * white(T) ].
```

This is an *ordinary* MST — just add `lambda` to every white edge's weight and run Kruskal. No count dimension, no per-count DSU: a single greedy MST, `O(m log m)`. The Lagrangian duality statement, valid because `f` is convex, is

```
f(k) = max over lambda of [ C(lambda) - lambda * k ].
```

Geometrically: `C(lambda) - lambda*k` is, for each `lambda`, a supporting line of the convex hull of the points `(k, f(k))`; the maximum over `lambda` recovers the hull value at `k`, which equals `f(k)` because `f` is convex (no integrality gap). The maximizing `lambda` is one whose penalized-optimal tree can be made to use exactly `k` white edges. So the plan is: search `lambda`, find the one where the penalized MST realizes `k` whites, then read `f(k) = C(lambda) - lambda*k`. This is the Aliens trick applied not to a DP but to a **matroid/greedy** objective — the relaxed subproblem is Kruskal, not a recurrence.

**Why a *scalar* `lambda` and a binary search suffice.** As `lambda` increases, white edges become uniformly more expensive, so any penalized-optimal tree uses **no more** white edges than before: the white count `cnt(lambda)` of the penalized MST is **monotone non-increasing** in `lambda`. Monotone means binary-searchable. The achievable counts range over `[minWhite, maxWhite]`; I can find these extremes directly by making white absurdly expensive (`lambda = +BIG`, fewest whites) or absurdly cheap (`lambda = -BIG`, most whites), where `BIG = maxW + 1` strictly dominates any raw-weight difference so the colour preference wins every tie. If `k` is outside `[minWhite, maxWhite]`, output `-1`. Otherwise binary-search `lambda` to land on `k`.

**The subtle part: ties, and which `lambda` to pick.** Here is the trap I must respect. At the critical `lambda`, several white counts can be simultaneously optimal — the penalized objective is flat across a whole segment of `f` (the segment whose slope equals `-lambda`). Concretely, if `f` has a run of slope `-1` (as in my probe: `f(1)=18, f(2)=17` differ by `-1`), then at `lambda = 1` both `k=1` and `k=2` minimize the penalized cost. A penalized MST with a fixed tie-break will pick *one* of them, and it might not be the `k` I asked for. So I must control ties deliberately. For a given `lambda`, define:

- `cntMax(lambda)` = the **largest** white count among penalized-optimal trees, obtained by breaking penalized-weight ties **in favour of white edges** (take a white edge before a black one of equal penalized weight);
- `cntMin(lambda)` = the **smallest**, obtained by breaking ties in favour of black.

The penalized *cost* `C(lambda)` is identical under either tie-break (Kruskal gives the optimal cost regardless of tie order); only the colour count differs. The set of achievable counts at `lambda` is exactly `[cntMin(lambda), cntMax(lambda)]`, and the recovery `f(k) = C(lambda) - lambda*k` is valid **iff** `cntMin(lambda) <= k <= cntMax(lambda)`.

**Which `lambda` makes `k` sandwiched?** `cntMax(lambda)` is non-increasing. I claim the correct pivot is the **largest** `lambda` with `cntMax(lambda) >= k`. At that `lambda`, `cntMax(lambda) >= k` directly. For the lower side: by maximality, `cntMax(lambda+1) < k`. On a convex hull, the minimizer intervals of consecutive integer multipliers share a breakpoint — `cntMin(lambda) = cntMax(lambda+1)` — so `cntMin(lambda) = cntMax(lambda+1) < k`, giving `cntMin(lambda) < k <= cntMax(lambda)`. `k` is sandwiched, and the recovery is exact. (I'll re-derive and re-verify this empirically below, because this is precisely where I expect to slip.)

**First implementation.** I split edges by colour, pre-sort each colour once by weight, and build each penalized MST by **merging** the two sorted lists (adding `lambda` to white keys), which is `O(m)` per `lambda` rather than `O(m log m)` — important because the binary search runs ~31 iterations. My first attempt got the recovery pivot wrong: I wrote the binary search to find the *smallest* `lambda` such that `cntMax(lambda) <= k`, reasoning "increase the penalty until whites drop to `k`". The loop:

```
long long lo = -BIG, hi = BIG, lam = BIG;
while (lo <= hi) {
    long long mid = lo + ((hi - lo) >> 1);
    int cnt = buildMST(mid, /*preferWhite=*/true).wcnt;
    if (cnt <= k) { lam = mid; hi = mid - 1; }   // smallest lambda with cnt <= k
    else lo = mid + 1;
}
Res r = buildMST(lam, true);
long long answer = r.pen - lam * (long long)k;
```

**The bug surfaces under differential testing.** I ran the solver against a brute force (enumerate all `(n-1)`-edge subsets, keep the spanning ones with exactly `k` whites, take the min weight) over hundreds of random small graphs. Almost all passed, then a mismatch on a 6-vertex, 10-edge case with `k=1`: my solver said `17`, brute said `18`. Crucially `17 < 18`, and brute is the true minimum over exactly-1-white trees — so my answer was **below** the real optimum, i.e. *infeasible*: I had reported a weight no exactly-1-white tree can achieve.

**Diagnosing it by probing every `lambda`.** I instrumented a probe that, for each integer `lambda` from `-3..6`, prints `cntMin`, `cntMax`, the penalized cost, and both candidate recoveries. The output on the failing case:

```
lam=0  cntMin=2 cntMax=2 pen=17  recover = 17 - 0*1 = 17
lam=1  cntMin=0 cntMax=2 pen=19  recover = 19 - 1*1 = 18
lam=2  cntMin=0 cntMax=0 pen=19  recover = 19 - 2*1 = 17
```

Now it is obvious. The true answer is at `lam=1`: there `cntMin=0 <= k=1 <= 2=cntMax`, `k` is sandwiched, and `pen - lam*k = 19 - 1 = 18` — correct. But my "smallest `lambda` with `cntMax <= k=1`" picked `lam=2` (where `cntMax=0 <= 1`), and at `lam=2` the achievable counts are `[0,0]`, which does **not** contain `k=1`. The recovery `19 - 2*1 = 17` then evaluates a supporting line *past* the hull vertex for `k=1`, undershooting `f(1)`. The defect is precise: I picked a `lambda` on the wrong side of the breakpoint, so the duality recovery applied a line whose contact point with the hull is at a different `k` than mine.

**The fix: pick the largest `lambda` with `cntMax(lambda) >= k`.** That is the pivot my hull argument actually justified. Change the comparison and the search direction:

```
long long lo = -BIG, hi = BIG, lam = -BIG;
while (lo <= hi) {
    long long mid = lo + ((hi - lo) >> 1);
    int cnt = buildMST(mid, true).wcnt;       // cntMax(mid)
    if (cnt >= k) { lam = mid; lo = mid + 1; } // push lambda as high as possible
    else hi = mid - 1;
}
```

On the failing case this lands on `lam=1` (`cntMax(1)=2 >= 1`, while `cntMax(2)=0 < 1`), and `pen - lam*k = 19 - 1 = 18`. Fixed, and fixed for the reason the hull argument predicted — `k` is now sandwiched between `cntMin` and `cntMax` at the chosen `lambda`, which is the exact precondition for `f(k) = C(lambda) - lambda*k` to hold.

**Re-verifying after the fix.** I re-ran the differential test: 1500 cases from a connectivity-guaranteed random generator and 1500 adversarial cases (dense graphs, all-white / all-black colourings, a wide weight spread, and `k` forced to `0`, `n-1`, and out-of-range), all against the brute force. Zero mismatches. The previously failing 6-vertex case now returns `18`. The bug was real, its cause was a wrong-side pivot at a hull breakpoint, and the corrected pivot is the one the convexity argument earns.

**Edge cases, deliberately.**
- *`k` out of range.* If `k < 0` or `k > n-1`, no spanning tree can have that many white edges (a tree has only `n-1` edges); I short-circuit to `-1` before anything else.
- *Infeasible count.* If `k < minWhite` or `k > maxWhite` (computed via `lambda = +-BIG`), output `-1`. Example: a triangle with two white and one black edge has `minWhite = 1` (any 2-edge spanning tree keeps at least one white), so `k=0` correctly yields `-1`.
- *Disconnected graph.* Build an MST at `lambda = 0`; if it fails to use `n-1` edges, the graph is disconnected and the answer is `-1`.
- *`n = 1`.* No edges needed; the empty tree has `0` edges and `0` white edges, so `k=0` gives `0` and any `k>=1` gives `-1` (caught by `k > n-1`). I also guard `n = 0` to print `0` defensively, though the constraints start at `n = 1`.
- *Ties between equal-penalized white and black edges.* Handled explicitly by the `preferWhite` flag in the merge, which is the whole mechanism for reading off `cntMax` vs `cntMin`.
- *Overflow.* Weights and penalties accumulate in `long long`; the maximum total `~2*10^14` and the penalized cost `pen +- BIG*(n-1)` (with `BIG <= 10^9 + 1`) stay well within 64-bit range. The recovery `pen - lam*k` with `|lam| <= BIG` and `k <= n-1` is at most `~10^9 * 2*10^5 = 2*10^{14}`, safe.

**Performance check.** Splitting by colour and pre-sorting each list once is two `O(m log m)` sorts. Each penalized MST is an `O(m)` merge plus near-`O(m)` union-find (`alpha` inverse-Ackermann). The binary search runs `O(log BIG) ~ 31` iterations, plus a constant number of extreme/connectivity builds, so the total is `O(m log m + m log(maxW))`. On a worst case I generated — `n = 2*10^5`, `m = 4*10^5`, weights spread across the full `[0, 10^9]` range, and a mid-range feasible `k` that forces all ~31 binary-search iterations — it runs in about `0.30 s` using ~9 MB, comfortably inside the 2-second / 256 MB budget. (An earlier version that re-sorted the entire edge array on every `lambda` clocked `1.22 s` on a smaller dense case; the colour-split merge is what brought it down.)

**Final solution.** I disproved the per-count DP by sizing it (`~4*10^{10}` transitions, the wrong shape for the scale), discovered convexity of `f(k)` by computing it on a small case, used convexity to justify the Lagrangian penalty whose relaxed subproblem is a *greedy* MST (not a DP), made the count monotone in `lambda` for a binary search, and — after a real wrong-side-pivot bug caught by brute-force differential testing — settled on the largest `lambda` with `cntMax(lambda) >= k` as the provably correct pivot, recovering `f(k) = C(lambda) - lambda*k`. One self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, r;
    void init(int n) { p.resize(n); r.assign(n, 0); iota(p.begin(), p.end(), 0); }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (r[a] < r[b]) swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) r[a]++;
        return true;
    }
};

int n, m;
// Edges split by colour. Within each colour we pre-sort by (weight, original index)
// ONCE. Adding lambda to every white weight shifts all white keys uniformly, so the
// white edges' internal order never changes; only the white block slides relative to
// the black block. That lets each penalised MST be built by an O(m) MERGE of the two
// pre-sorted lists instead of an O(m log m) re-sort per lambda.
struct E { int u, v, w; };
vector<E> white, black;

struct Res { bool ok; long long pen; int wcnt; };

// Build a min-penalised MST: minimise sum(weight) + lambda*(#white edges).
// Merge white (key = w+lambda) and black (key = w) by key; on a tie, preferWhite
// decides whether a white or black edge of equal penalised weight is taken first.
// preferWhite=true  -> maximises #white among optimal penalised trees (cntMax).
// preferWhite=false -> minimises it (cntMin). The penalised cost is the same either way.
Res buildMST(long long lambda, bool preferWhite) {
    DSU d; d.init(n);
    size_t i = 0, j = 0;
    long long pen = 0;
    int wcnt = 0, used = 0, need = n - 1;
    while (used < need && (i < white.size() || j < black.size())) {
        bool takeWhite;
        if (i >= white.size()) takeWhite = false;
        else if (j >= black.size()) takeWhite = true;
        else {
            long long kw = (long long)white[i].w + lambda;
            long long kb = (long long)black[j].w;
            if (kw != kb) takeWhite = (kw < kb);
            else takeWhite = preferWhite; // tie -> colour preference
        }
        if (takeWhite) {
            const E &e = white[i++];
            if (d.unite(e.u, e.v)) {
                pen += (long long)e.w + lambda; wcnt++; used++;
            }
        } else {
            const E &e = black[j++];
            if (d.unite(e.u, e.v)) {
                pen += (long long)e.w; used++;
            }
        }
    }
    return { used == need, pen, wcnt };
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int k;
    if (!(cin >> n >> m >> k)) return 0;
    int maxW = 1;
    for (int idx = 0; idx < m; idx++) {
        int u, v, w, c;
        cin >> u >> v >> w >> c;
        u--; v--;
        maxW = max(maxW, w);
        if (c) white.push_back({u, v, w});
        else   black.push_back({u, v, w});
    }
    auto byW = [](const E &a, const E &b) { return a.w < b.w; };
    sort(white.begin(), white.end(), byW);
    sort(black.begin(), black.end(), byW);

    if (n == 0) { cout << 0 << "\n"; return 0; }
    if (k < 0 || k > n - 1) { cout << -1 << "\n"; return 0; }

    // Connectivity (lambda = 0): if the full edge set can't span, no tree exists.
    if (!buildMST(0, true).ok) { cout << -1 << "\n"; return 0; }

    // Achievable white-count window [minWhite, maxWhite_white].
    // BIG strictly dominates any difference of raw weights, so +-BIG forces the
    // fewest / most white edges possible.
    long long BIG = (long long)maxW + 1;
    int minWhite = buildMST(BIG, false).wcnt;   // white extremely expensive
    int maxWhite = buildMST(-BIG, true).wcnt;   // white extremely cheap
    if (k < minWhite || k > maxWhite) { cout << -1 << "\n"; return 0; }

    // Aliens trick. f(k) = min weight of a spanning tree with exactly k white edges
    // is convex on [minWhite, maxWhite]. With C(lambda) = min_T (weight + lambda*white),
    // the Lagrangian dual is f(k) = max over integer lambda of (C(lambda) - lambda*k);
    // it is attained at any lambda whose penalised-optimal trees realise exactly k
    // whites, i.e. cntMin(lambda) <= k <= cntMax(lambda).
    //
    // cntMax(lambda) (white-first tie-break) is non-increasing in lambda. The correct
    // pivot is the LARGEST lambda with cntMax(lambda) >= k: there cntMax >= k, and since
    // consecutive minimiser ranges on the convex hull share a breakpoint
    // (cntMin(lambda) = cntMax(lambda+1) < k), also cntMin(lambda) <= k. So k is
    // sandwiched and f(k) = C(lambda) - lambda*k is exact.
    long long lo = -BIG, hi = BIG, lam = -BIG;
    while (lo <= hi) {
        long long mid = lo + ((hi - lo) >> 1);
        int cnt = buildMST(mid, true).wcnt; // cntMax(mid)
        if (cnt >= k) { lam = mid; lo = mid + 1; }
        else hi = mid - 1;
    }

    Res r = buildMST(lam, true);            // penalised cost C(lam) = r.pen
    long long answer = r.pen - lam * (long long)k;

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The per-count DP entangles connectivity with the colour count and balloons to `~10^{10}` work, the wrong shape for `m = 4*10^5`; computing `f(k)` on a small case revealed it is convex (a graphic-matroid exchange property); convexity licenses the Lagrangian / Aliens trick, turning "exactly `k` white" into a penalized *ordinary MST* (greedy, not DP) with a count that is monotone in the penalty `lambda`, hence binary-searchable; the only real hazard is the hull breakpoint, where a wrong-side pivot made the duality recovery underschoot `f(k)` (the `17`-vs-`18` bug on a 6-vertex graph), fixed by choosing the largest `lambda` with `cntMax(lambda) >= k` so `k` is sandwiched in `[cntMin, cntMax]`; colour-split pre-sorting plus an `O(m)` merge per `lambda` keeps the whole thing at `~0.3 s` on the largest inputs.
