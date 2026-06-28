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
