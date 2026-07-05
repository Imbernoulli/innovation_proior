#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Multi-commodity shortest-path interdiction generator ("Lag Storm: Arena Fiber Interdiction").
// testId is a difficulty/structure ladder:
//   testId 1  -> tiny (example scale), testId 10 -> large/adversarial.
// Instance design: a CHEAP backbone (spanning tree, small weights) so shortest paths funnel
// through few links, plus HEAVY-tailed shortcut links (expensive alternates). Cutting cheap
// backbone links forces long detours -> real interdiction leverage & strategy divergence.

struct DSU {
    vector<int> p;
    DSU(int n): p(n+1){ iota(p.begin(), p.end(), 0); }
    int f(int x){ while(p[x]!=x){ p[x]=p[p[x]]; x=p[x]; } return x; }
    bool uni(int a,int b){ a=f(a); b=f(b); if(a==b) return false; p[a]=b; return true; }
};

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // ---- size ladder ----
    int n = 6 + 4 * t * t;                 // t1: 10 ... t10: 406
    if (n > 420) n = 420;
    int extra = (int)(1.6 * n);            // extra shortcut edges on top of the (n-1) backbone
    int m = (n - 1) + extra;
    if (m > 1200) { m = 1200; extra = m - (n - 1); }
    int P = min(30, 2 + 2 * t);            // t1: 4 ... t10: 22
    int k = min(15, 1 + t);                // t1: 2 ... t10: 11

    // ---- build a cheap random spanning tree (backbone) ----
    vector<array<ll,3>> edges;             // {u, v, w}
    edges.reserve(m);
    {
        vector<int> perm(n);
        for (int i = 0; i < n; i++) perm[i] = i + 1;
        shuffle(perm.begin(), perm.end());
        DSU dsu(n);
        // attach node perm[i] to a random earlier node -> connected tree, backbone weights small
        for (int i = 1; i < n; i++) {
            int j = rnd.next(0, i - 1);
            int u = perm[i], v = perm[j];
            ll w = rnd.next(1, 3);
            edges.push_back({(ll)u, (ll)v, w});
            dsu.uni(u, v);
        }
    }

    // ---- add heavy-tailed shortcut edges (expensive alternates) ----
    for (int e = 0; e < extra; e++) {
        int u = rnd.next(1, n), v = rnd.next(1, n);
        while (v == u) v = rnd.next(1, n);
        ll w;
        int roll = rnd.next(0, 9);
        if (roll < 6)      w = rnd.next(8, 60);     // medium
        else if (roll < 9) w = rnd.next(60, 300);   // expensive
        else               w = rnd.next(300, 1000); // rare very expensive
        edges.push_back({(ll)u, (ll)v, w});
    }

    // shuffle edge order so backbone links are not trivially the first indices
    shuffle(edges.begin(), edges.end());

    m = (int)edges.size();

    // ---- monitored demand pairs (distinct endpoints, heavy priorities) ----
    vector<array<int,3>> demands; // {a, b, d}
    for (int i = 0; i < P; i++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        while (b == a) b = rnd.next(1, n);
        int d = rnd.next(1, 5);
        demands.push_back({a, b, d});
    }

    // ---- emit ----
    printf("%d %d %d %d\n", n, m, P, k);
    for (auto& e : edges) printf("%lld %lld %lld\n", e[0], e[1], e[2]);
    for (auto& q : demands) printf("%d %d %d\n", q[0], q[1], q[2]);
    return 0;
}
