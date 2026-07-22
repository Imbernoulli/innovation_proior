// TIER: strong
// Global-then-local: approximate the Fiedler vector (second-smallest eigenvector
// of the graph Laplacian) via a fixed shifted power-iteration budget, sweep every
// balance-feasible threshold along that global coordinate to seed a partition,
// then run a bounded boundary local-search pass on the ACTUAL normalized-cut
// objective to polish it.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, lo, hi;
    scanf("%d %d %d %d", &n, &m, &lo, &hi);
    vector<int> eu(m), ev(m), ew(m);
    vector<vector<pair<int,int>>> adj(n + 1);
    vector<double> deg(n + 1, 0.0);
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        eu[i] = u; ev[i] = v; ew[i] = w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        deg[u] += w; deg[v] += w;
    }

    // ---- Stage 1: approximate Fiedler vector via shifted power iteration ----
    double dmax = 0.0;
    for (int v = 1; v <= n; v++) dmax = max(dmax, deg[v]);
    double c = 2.0 * dmax + 1.0;

    vector<double> x(n + 1);
    for (int v = 1; v <= n; v++) {
        unsigned long long h = (unsigned long long)v * 2654435761ULL + 12345ULL;
        h ^= (h >> 13); h *= 0x2545F4914F6CDD1DULL; h ^= (h >> 17);
        x[v] = (double)(h % 1000003) / 1000003.0 - 0.5;
    }
    auto orthonormalize = [&](vector<double>& z) {
        double mean = 0.0;
        for (int v = 1; v <= n; v++) mean += z[v];
        mean /= n;
        double norm = 0.0;
        for (int v = 1; v <= n; v++) { z[v] -= mean; norm += z[v] * z[v]; }
        norm = sqrt(max(norm, 1e-18));
        for (int v = 1; v <= n; v++) z[v] /= norm;
    };
    orthonormalize(x);

    const int NITER = 220;
    vector<double> y(n + 1);
    for (int it = 0; it < NITER; it++) {
        for (int v = 1; v <= n; v++) {
            double Lx = deg[v] * x[v]; // (D - W) x contribution, minus neighbor part below
            for (auto& e : adj[v]) Lx -= e.second * x[e.first];
            y[v] = c * x[v] - Lx; // (cI - L) x
        }
        orthonormalize(y);
        swap(x, y);
    }

    // ---- Stage 2: sweep along the global coordinate for the best threshold ----
    vector<int> ord(n);
    for (int i = 0; i < n; i++) ord[i] = i + 1;
    sort(ord.begin(), ord.end(), [&](int a, int b) { return x[a] < x[b]; });

    vector<char> inSide0(n + 1, 0);
    long long cut = 0;
    double volA = 0.0, volB = 0.0;
    for (int v = 1; v <= n; v++) volB += deg[v];

    double bestNcut = 1e18;
    int bestK = -1;
    for (int k = 1; k <= n; k++) {
        int v = ord[k - 1];
        for (auto& e : adj[v]) {
            int u = e.first, w = e.second;
            if (inSide0[u]) cut -= w; else cut += w;
        }
        inSide0[v] = 1;
        volA += deg[v]; volB -= deg[v];
        if (k >= lo && k <= hi && volA > 0 && volB > 0) {
            double val = (double)cut * (1.0 / volA + 1.0 / volB);
            if (val < bestNcut) { bestNcut = val; bestK = k; }
        }
    }

    vector<int> side(n + 1, 1);
    if (bestK == -1) {
        int half = n / 2;
        for (int i = 0; i < n; i++) side[ord[i]] = (i < half) ? 0 : 1;
    } else {
        for (int i = 0; i < bestK; i++) side[ord[i]] = 0;
        for (int i = bestK; i < n; i++) side[ord[i]] = 1;
    }

    // recompute exact cut/volA/volB for the chosen partition
    long long curCut = 0;
    double curVolA = 0.0, curVolB = 0.0;
    for (int v = 1; v <= n; v++) {
        if (side[v] == 0) curVolA += deg[v]; else curVolB += deg[v];
    }
    for (int i = 0; i < m; i++) if (side[eu[i]] != side[ev[i]]) curCut += ew[i];

    long long c0 = 0;
    for (int v = 1; v <= n; v++) if (side[v] == 0) c0++;

    // ---- Stage 3: bounded boundary local search on the ACTUAL objective ----
    const int PASSES = 40;
    for (int pass = 0; pass < PASSES; pass++) {
        bool changed = false;
        double curF = (double)curCut * (1.0 / curVolA + 1.0 / curVolB);
        for (int v = 1; v <= n; v++) {
            int s = side[v];
            long long deltaCut = 0;
            for (auto& e : adj[v]) {
                int u = e.first, w = e.second;
                if (side[u] == s) deltaCut += w; else deltaCut -= w;
            }
            long long newCut = curCut + deltaCut;
            double newVolA = curVolA, newVolB = curVolB;
            long long newC0 = c0;
            if (s == 0) { newVolA -= deg[v]; newVolB += deg[v]; newC0--; }
            else { newVolA += deg[v]; newVolB -= deg[v]; newC0++; }
            if (newC0 < lo || newC0 > hi) continue;
            if (newVolA <= 0 || newVolB <= 0) continue;
            double newF = (double)newCut * (1.0 / newVolA + 1.0 / newVolB);
            if (newF < curF - 1e-12) {
                side[v] = 1 - s;
                curCut = newCut; curVolA = newVolA; curVolB = newVolB; c0 = newC0;
                curF = newF;
                changed = true;
            }
        }
        if (!changed) break;
    }

    for (int v = 1; v <= n; v++) printf("%d ", side[v]);
    printf("\n");
    return 0;
}
