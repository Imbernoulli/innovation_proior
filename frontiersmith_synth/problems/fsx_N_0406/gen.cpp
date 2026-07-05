#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ------------------------------------------------------------------
// Smart-City Lighting: Phase Harmonization over GF(q)
//
// Nodes = light controllers, each gets a phase label x_u in {0..q-1}.
// Edge e=(u,v): weight w_e, forbidden SUM set F_e subset of GF(q).
//   edge "clear"  iff (x_u+x_v) mod q  NOT in F_e.
// Hub h: bonus g_h, "harmonized" iff EVERY incident edge is clear (AND).
//
// The generator plants a hidden algebraic labeling xstar that clears
// all "spine" edges and harmonizes planted hubs, hides it amid
// adversarial conflict cliques and needle hubs, and fills the size
// envelope on the largest tests.
// ------------------------------------------------------------------

int q, n;
vector<int> xs;               // hidden planted labeling
struct Edge { int u, v; ll w; vector<int> forb; };
vector<Edge> E;

// forbidden set that EXCLUDES sstar (planted edge -> xstar clears it).
// allowSize = how many residues are allowed (>=1). allowSize=1 -> only
// xstar's residue clears it (hardest); larger allowSize -> easier.
vector<int> forbExclude(int sstar, int allowSize) {
    allowSize = max(1, min(allowSize, q));
    vector<int> allowed;
    allowed.push_back(sstar);
    vector<int> pool;
    for (int r = 0; r < q; r++) if (r != sstar) pool.push_back(r);
    shuffle(pool.begin(), pool.end());
    for (int i = 0; i < allowSize - 1 && i < (int)pool.size(); i++)
        allowed.push_back(pool[i]);
    vector<char> isAllowed(q, 0);
    for (int a : allowed) isAllowed[a] = 1;
    vector<int> forb;
    for (int r = 0; r < q; r++) if (!isAllowed[r]) forb.push_back(r);
    return forb;
}

// fully random forbidden set of given size (MAY include sstar -> conflict).
vector<int> forbRandom(int ksize) {
    ksize = max(1, min(ksize, q - 1));
    vector<int> pool;
    for (int r = 0; r < q; r++) pool.push_back(r);
    shuffle(pool.begin(), pool.end());
    vector<int> forb(pool.begin(), pool.begin() + ksize);
    return forb;
}

void addEdge(int u, int v, ll w, vector<int> forb) {
    if (u == v) return;
    E.push_back({u, v, w, move(forb)});
}

int sstarOf(int u, int v) { return (xs[u] + xs[v]) % q; }

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    if (t < 1) t = 1;
    if (t > 10) t = 10;

    int primes[10] = {5, 7, 7, 11, 11, 13, 13, 13, 17, 17};
    int nn[10]     = {8, 120, 500, 1200, 3000, 6000, 10000, 18000, 28000, 40000};
    q = primes[t - 1];
    n = nn[t - 1];
    int inv2 = (q + 1) / 2;   // multiplicative inverse of 2 mod q

    // hidden planted labeling
    xs.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) xs[i] = rnd.next(0, q - 1);

    // ---- 1. SPINE: heavy, hard planted edges (dominate total weight) ----
    // connectivity tree; only xstar (allowSize 1..2) clears them, so a
    // constant labeling recovers ~1/q of this weight -> small baseline.
    for (int i = 2; i <= n; i++) {
        int j = rnd.next(1, i - 1);
        ll w = rnd.next(120, 400);
        int as = (rnd.next(0, 4) == 0) ? 2 : 1;
        addEdge(i, j, w, forbExclude(sstarOf(i, j), as));
    }

    // ---- 2. EXTRA planted edges (medium weight, medium hardness) ----
    int extra = (int)(1.4 * n);
    for (int k = 0; k < extra; k++) {
        int u = rnd.next(1, n), v = rnd.next(1, n);
        if (u == v) continue;
        ll w = rnd.next(20, 90);
        int as = rnd.next(2, max(2, q / 2));
        addEdge(u, v, w, forbExclude(sstarOf(u, v), as));
    }

    // ---- 3. FILLER easy edges (light weight, small forbidden set) ----
    // cheap output; many are clearable by lots of labelings -> greedy noise.
    int filler = (int)(0.8 * n);
    for (int k = 0; k < filler; k++) {
        int u = rnd.next(1, n), v = rnd.next(1, n);
        if (u == v) continue;
        ll w = rnd.next(1, 12);
        addEdge(u, v, w, forbRandom(rnd.next(1, 3)));
    }

    // ---- 4. CONFLICT CLIQUES (adversarial; some edges even xstar misses) --
    int nClq = 4 + t * 3;
    for (int c = 0; c < nClq; c++) {
        int cs = rnd.next(3, 5);
        vector<int> nodes;
        for (int i = 0; i < cs; i++) nodes.push_back(rnd.next(1, n));
        for (int a = 0; a < cs; a++)
            for (int b = a + 1; b < cs; b++) {
                int u = nodes[a], v = nodes[b];
                if (u == v) continue;
                ll w = rnd.next(40, 120);
                // 60% conflict (random, may block xstar), 40% planted-hard
                if (rnd.next(0, 9) < 6) addEdge(u, v, w, forbRandom(rnd.next(q / 2, q - 1)));
                else                    addEdge(u, v, w, forbExclude(sstarOf(u, v), 1));
            }
    }

    // ---- 5. HUBS: planted cliques with all-or-nothing bonus ----
    // hub incident edges are hard-planted (allowSize 1) so a constant almost
    // never harmonizes a hub -> the bonus lives above the baseline.
    int H = 3 + t * 4;
    if (H > n / 2) H = n / 2;
    vector<pair<int, ll>> hubs;                 // (node, bonus)
    vector<char> usedHub(n + 1, 0);
    for (int h = 0; h < H; h++) {
        int hub = rnd.next(1, n);
        if (usedHub[hub]) continue;
        usedHub[hub] = 1;
        int deg = rnd.next(3, 6);
        for (int d = 0; d < deg; d++) {
            int v = rnd.next(1, n);
            if (v == hub) continue;
            ll w = rnd.next(60, 160);
            addEdge(hub, v, w, forbExclude(sstarOf(hub, v), 1));
        }
        ll g = rnd.next(60, 240);
        hubs.push_back({hub, g});
    }

    // ---- 6. NEEDLE hub: one large planted clique, big bonus (t>=6) ----
    if (t >= 6) {
        int center = rnd.next(1, n);
        if (!usedHub[center]) {
            usedHub[center] = 1;
            int deg = rnd.next(7, 10);
            for (int d = 0; d < deg; d++) {
                int v = rnd.next(1, n);
                if (v == center) continue;
                ll w = rnd.next(80, 200);
                addEdge(center, v, w, forbExclude(sstarOf(center, v), 1));
            }
            ll g = rnd.next(1500, 4000);
            hubs.push_back({center, g});
        }
    }

    // ---- emit ----
    shuffle(E.begin(), E.end());
    int m = (int)E.size();
    printf("%d %d %d %d\n", n, m, q, (int)hubs.size());
    for (auto& e : E) {
        printf("%d %d %lld %d", e.u, e.v, e.w, (int)e.forb.size());
        for (int f : e.forb) printf(" %d", f);
        printf("\n");
    }
    for (auto& hb : hubs) printf("%d %lld\n", hb.first, hb.second);
    return 0;
}
