#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Quantum Coupler Interdiction: Composite-Length Signal Paths  (generator)
//
// The lattice is built from "resonance bundles". A bundle of hop-length L routes
// pulses s -> (p parallel cheap entries) -> g -> [bridge couplers] -> t so that
// EVERY route in the bundle has exactly L hops and shares a chain of BRIDGE
// couplers (g->h and the tail): cutting any one bridge destroys hop-length L.
// Bridges cost more than the cheap parallel entries, so the myopic "cut the
// cheapest coupler on the shortest route" greedy is lured onto entries (which
// never eliminate a length). Bundles use DISTINCT lengths, so the set of
// realizable s->t hop-lengths equals the set of bundle lengths.
//
// A short COMPOSITE bait bundle is the globally shortest route -> traps the
// shortest-path greedy. Prime bundles split into cheap-bridge (worth cutting)
// and expensive-bridge (not worth cutting under penalty P) so the optimum knocks
// out only some prime lengths -> graded, non-saturating objective.
//
// Vertices carry a topological "depth"; every coupler goes strictly deeper, and
// after relabelling by depth every edge satisfies u < v.
//
// Output:  n m s t k P   then m lines  u v c .
// -----------------------------------------------------------------------------

struct Edge { int u, v, c; };
vector<ll> depth;                 // depth per internal vertex index
vector<Edge> edges;
int S_IDX, T_IDX;
ll DEEP = (ll)4e18;

int newVertex(ll d){ depth.push_back(d); return (int)depth.size() - 1; }
void addC(int u, int v, int c){ edges.push_back({u, v, c}); }

// Build a bundle of hop-length L, p parallel entries; entries cheap, bridges = bridgeCost.
void buildBundle(int L, int p, int entryCost, int bridgeCost){
    // entries at depth 1, g at depth 2, h at depth 3, tail y_i at depth 3+i
    int g = newVertex(2);
    for (int i = 0; i < p; i++){
        int e = newVertex(1);
        addC(S_IDX, e, entryCost);
        addC(e, g, entryCost);
    }
    int h = newVertex(3);
    addC(g, h, bridgeCost);                 // bridge coupler #1
    int steps = L - 3;                       // remaining bridge couplers to reach t
    int prev = h; ll dd = 3;
    for (int i = 1; i <= steps; i++){
        if (i < steps){
            int y = newVertex(dd + i);
            addC(prev, y, bridgeCost);
            prev = y;
        } else {
            addC(prev, T_IDX, bridgeCost);
        }
    }
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    // primes / composites in [4,40]
    vector<int> primes, comps;
    {
        vector<char> ip(41, 1); ip[0]=ip[1]=0;
        for (int i=2;i<=40;i++) if(ip[i]) for(int j=2*i;j<=40;j+=i) ip[j]=0;
        for (int i=5;i<=40;i++){ if(ip[i]) primes.push_back(i); else if(i>=6) comps.push_back(i); }
    }

    S_IDX = newVertex(0);              // source, depth 0
    T_IDX = newVertex(DEEP);           // sink, deepest

    // ---- ladder parameters ----
    int nPrime = 4 + (int)llround(f * 4.0);            // 4 .. 8 prime bundles
    nPrime = min(nPrime, (int)primes.size());
    int cheapPrime = (int)ceil(nPrime * 0.7);          // cheap-bridge prime bundles
    ll P = 500 + (ll)llround(f * 1500.0);              // penalty per prime length
    int k = min(25, cheapPrime + 3);                   // budget
    int baitEntries = k + 2;                            // > k so greedy can't clear bait
    int pE = 3 + (int)llround(f * 8.0);                // parallel entries per prime bundle
    int targetN = 8 + (int)llround(f * 1600.0);        // envelope target on #vertices

    // ---- composite bait bundle: the globally shortest route (length 4) ----
    buildBundle(4, baitEntries, 1, 40 + rnd.next(0, 40));

    // ---- prime bundles with distinct lengths ----
    // pick nPrime distinct primes spread across the range
    vector<int> chosen;
    for (int i = 0; i < nPrime; i++){
        int idx = (int)((ll)i * (primes.size() - 1) / max(1, nPrime - 1));
        chosen.push_back(primes[idx]);
    }
    sort(chosen.begin(), chosen.end());
    chosen.erase(unique(chosen.begin(), chosen.end()), chosen.end());
    for (int i = 0; i < (int)chosen.size(); i++){
        int L = chosen[i];
        bool cheap = (i < cheapPrime);
        int bridgeCost = cheap ? (5 + rnd.next(0, 35))          // worth cutting: < P
                               : (int)min((ll)1000, 2 * P + rnd.next(0, (int)min((ll)500, P)));
        int p = max(2, pE + rnd.next(-1, 2));
        buildBundle(L, p, 1, bridgeCost);
    }

    // ---- composite noise bundles to fill the envelope (never affect penalty) ----
    while ((int)depth.size() < targetN - 60){
        int L = comps.empty() ? 8 : comps[rnd.next(0, (int)comps.size() - 1)];
        int p = 2 + rnd.next(0, 3 + (int)llround(f * 6.0));
        int bc = 20 + rnd.next(0, 400);
        if ((int)depth.size() + p + L + 4 > 2050) break;
        if ((int)edges.size() + 2 * p + L + 4 > 39000) break;
        buildBundle(L, p, 1 + rnd.next(0, 3), bc);
    }

    // ---- relabel vertices by depth so every edge has u < v ----
    int N = (int)depth.size();
    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i;
    stable_sort(order.begin(), order.end(), [&](int a, int b){ return depth[a] < depth[b]; });
    vector<int> id(N);
    for (int newid = 0; newid < N; newid++) id[order[newid]] = newid;
    int s = id[S_IDX], t = id[T_IDX];
    // s must be 0 (unique min depth) and t must be N-1 (unique max depth)
    // (guaranteed: depth[s]=0 is the unique minimum, depth[t]=DEEP the unique max)

    // shuffle listing order (keeps u<v; only line order changes)
    for (int i = (int)edges.size() - 1; i > 0; i--) swap(edges[i], edges[rnd.next(0, i)]);

    int m = (int)edges.size();
    printf("%d %d %d %d %d %lld\n", N, m, s, t, k, (ll)P);
    for (auto &e : edges){
        int u = id[e.u], v = id[e.v];
        if (u > v) swap(u, v);          // safety: enforce u<v (depths differ so never equal on a real edge)
        printf("%d %d %d\n", u, v, e.c);
    }
    return 0;
}
