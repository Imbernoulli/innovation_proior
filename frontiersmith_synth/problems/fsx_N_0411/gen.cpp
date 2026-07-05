#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

/*
 * Generator for "Interstellar Relay: Robust Degree-Bounded Backbone".
 * Emits an undirected candidate-edge network on n nodes. Node 0 is the hub.
 *
 * Candidate edge blocks (order matters -- the FIRST n-1 edges are the backbone
 * path, which the checker uses as its calibrated baseline B):
 *   [1 .. n-1]   backbone path (i-1,i)         : HIGH cost  -> baseline is expensive AND fragile
 *   [ .. ]       planted balanced degree-D tree: LOW-MED cost -> a hidden low-stress structure
 *   [ .. ]       trap chain (random perm path) : VERY cheap -> min-cost greedy falls into a fragile chain
 *   [ .. ]       noise edges                    : medium cost -> search-space filler
 *
 * The instance is guaranteed connected and admits a feasible degree<=D spanning
 * tree (the planted tree and the backbone are both such trees).
 */
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    int N[]  = {0, 8, 40, 150, 400, 900, 1500, 2500, 3500, 4500, 5000};
    int Dp[] = {0, 4,  4,   5,   4,   6,    5,    4,    5,    6,    4};
    int idx = t; if (idx < 1) idx = 1; if (idx > 10) idx = 10;
    int n = N[idx];
    int D = Dp[idx];

    vector<array<long long,3>> edges; // {u, v, w}

    // ---- backbone path (indices 1..n-1) : high cost ----
    long long bsum = 0;
    for (int i = 1; i <= n - 1; i++) {
        long long w = rnd.next(500, 1000);
        edges.push_back({(long long)(i - 1), (long long)i, w});
        bsum += w;
    }

    // ---- planted balanced degree-D tree over a shuffled node order, root = hub 0 ----
    {
        vector<int> ord; ord.push_back(0);
        vector<int> rest; for (int i = 1; i < n; i++) rest.push_back(i);
        for (int i = (int)rest.size() - 1; i >= 1; i--) { int j = rnd.next(0, i); swap(rest[i], rest[j]); }
        for (int x : rest) ord.push_back(x);
        int ptr = 1;
        for (int i = 0; i < n && ptr < n; i++) {
            int cap = (i == 0) ? D : (D - 1);       // total degree stays <= D
            for (int c = 0; c < cap && ptr < n; c++) {
                long long w = rnd.next(40, 120);
                edges.push_back({(long long)ord[i], (long long)ord[ptr], w});
                ptr++;
            }
        }
    }

    // ---- trap chain : a random Hamiltonian path with very cheap edges ----
    {
        vector<int> p; for (int i = 0; i < n; i++) p.push_back(i);
        for (int i = n - 1; i >= 1; i--) { int j = rnd.next(0, i); swap(p[i], p[j]); }
        for (int i = 0; i + 1 < n; i++) {
            long long w = rnd.next(1, 20);
            edges.push_back({(long long)p[i], (long long)p[i + 1], w});
        }
    }

    // ---- noise edges ----
    {
        int noise = 2 * n;
        for (int i = 0; i < noise; i++) {
            int a = rnd.next(0, n - 1), b = rnd.next(0, n - 1);
            if (a == b) b = (b + 1) % n;
            long long w = rnd.next(100, 400);
            edges.push_back({(long long)a, (long long)b, w});
        }
    }

    int m = (int)edges.size();
    long long L = bsum / (long long)max(1, n - 1);
    if (L < 1) L = 1;

    printf("%d %d %d %lld\n", n, m, D, L);
    for (auto& e : edges) printf("%lld %lld %lld\n", e[0], e[1], e[2]);
    return 0;
}
