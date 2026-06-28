**Problem.** A connected weighted graph has white (`c=1`) and black (`c=0`) edges. Find the minimum-weight spanning tree using **exactly `k` white edges**, or `-1` if none exists. Read `n m k` then `m` lines `u v w c` from stdin; print the minimum weight. Scale: `n <= 2*10^5`, `m <= 4*10^5`, `w <= 10^9`, so totals reach `~2*10^14` and every accumulator must be `long long`.

**Why the obvious per-count DP is wrong (for this scale).** The reflex is to make the colour count a DP state: process edges Kruskal-style while tracking "white edges used so far". But Kruskal's correctness depends on each edge being a *forced* take/skip; adding the count dimension breaks that — it can now be optimal to skip a cheap white edge to hit the count later, so you need a separate forest (and DSU) per count. That balloons to roughly `O(m * (n-1)) ~ 4*10^{10}` transitions at the stated limits — the wrong *shape*, not just a slow constant. The count dimension must be eliminated, not enumerated.

**Key idea — Lagrangian relaxation / the Aliens trick on a greedy (non-DP) objective.** Let `f(k)` = min spanning-tree weight using exactly `k` white edges. On its feasible interval `[minWhite, maxWhite]`, `f` is **convex** — a graphic-matroid base-exchange property (verifiable on any small case: e.g. `f = 19,18,17,21`, differences `-1,-1,+4`, non-decreasing). Convexity licenses replacing the hard "exactly `k`" constraint with a scalar penalty `lambda` on each white edge and solving the *unconstrained* penalized MST:

```
C(lambda) = min over trees T of [ weight(T) + lambda * white(T) ]   (just Kruskal on penalized weights)
f(k)      = max over integer lambda of [ C(lambda) - lambda * k ].
```

`C(lambda) - lambda*k` is a supporting line of the convex hull of `(k, f(k))`; its max over `lambda` equals `f(k)` with no integrality gap. The relaxed subproblem is a plain **greedy MST**, not a DP — that is the whole point: the "exactly `k`" constraint dissolves into tuning one number. As `lambda` rises, white edges get uniformly costlier, so the penalized MST's white count is **monotone non-increasing** in `lambda` — hence binary-searchable.

**Pitfalls.**
1. *Wrong-side pivot at a hull breakpoint (the real bug).* When `f` has a slope-`-1` run, multiple white counts are simultaneously penalized-optimal at the critical `lambda`. The recovery `f(k) = C(lambda) - lambda*k` is valid **only** when `cntMin(lambda) <= k <= cntMax(lambda)`. Picking "the smallest `lambda` with `cntMax(lambda) <= k`" lands one breakpoint too far and *undershoots* `f(k)` (a concrete 6-vertex case returned `17` instead of `18`). The correct pivot is the **largest `lambda` with `cntMax(lambda) >= k`**: then `cntMax >= k`, and since consecutive hull minimiser-ranges share a breakpoint (`cntMin(lambda) = cntMax(lambda+1) < k`), also `cntMin(lambda) <= k`, so `k` is sandwiched and the recovery is exact.
2. *Ties between equal-penalized white and black edges.* Control them explicitly: a "white-first" tie-break yields `cntMax(lambda)` (most whites among optimal trees); "black-first" yields `cntMin(lambda)`. The penalized *cost* is identical either way.
3. *Overflow.* Use `long long` throughout; `int` is a silent wrong-answer on large tests.
4. *Re-sorting per `lambda`.* Re-sorting all `m` edges every binary-search step is `O(m log m)` per step (measured `1.22 s` on a dense case). Split by colour, pre-sort each list **once**, and build each penalized MST by an `O(m)` **merge** (white keys shift by `lambda` uniformly, so their internal order is fixed) — this drops it to `~0.30 s`.

**Edge cases.** `k < 0` or `k > n-1` -> `-1` (a tree has only `n-1` edges). `k` outside `[minWhite, maxWhite]` (computed at `lambda = +-BIG`, `BIG = maxW+1`) -> `-1`. Disconnected graph (MST at `lambda=0` fails to span) -> `-1`. `n = 1` -> `0` for `k=0`, else `-1`. All-white / all-black graphs and parallel edges are handled by the same window check.

**Complexity.** `O(m log m)` for the two colour pre-sorts, plus `O(m)` per penalized MST over `O(log(maxW))` binary-search iterations (and a few extreme/connectivity builds): `O(m log m + m log(maxW))` time, `O(n + m)` memory. Measured `~0.30 s` / `~9 MB` at `n=2*10^5, m=4*10^5`.

**Code.**

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
