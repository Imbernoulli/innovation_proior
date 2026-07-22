// TIER: greedy
// Single-pass "grow from the two loudest hubs" heuristic: purely edge-local,
// no global structure. Vulnerable to decoy high-degree vertices.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, lo, hi;
    scanf("%d %d %d %d", &n, &m, &lo, &hi);
    vector<int> eu(m), ev(m), ew(m);
    vector<vector<pair<int,int>>> adj(n + 1);
    vector<long long> deg(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        eu[i] = u; ev[i] = v; ew[i] = w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        deg[u] += w; deg[v] += w;
    }

    // seed1 = highest weighted-degree vertex
    int seed1 = 1;
    for (int v = 2; v <= n; v++) if (deg[v] > deg[seed1]) seed1 = v;

    // seed2 = highest degree vertex NOT adjacent to seed1 (fallback: highest degree overall)
    vector<char> adjToSeed1(n + 1, 0);
    for (auto& e : adj[seed1]) adjToSeed1[e.first] = 1;
    int seed2 = -1;
    for (int v = 1; v <= n; v++) {
        if (v == seed1 || adjToSeed1[v]) continue;
        if (seed2 == -1 || deg[v] > deg[seed2]) seed2 = v;
    }
    if (seed2 == -1) {
        for (int v = 1; v <= n; v++) {
            if (v == seed1) continue;
            if (seed2 == -1 || deg[v] > deg[seed2]) seed2 = v;
        }
    }

    vector<int> side(n + 1, -1);
    vector<char> decided(n + 1, 0);
    side[seed1] = 0; decided[seed1] = 1;
    side[seed2] = 1; decided[seed2] = 1;
    long long nDecided0 = 1, nDecided1 = 1;

    // Region growing from the two seeds: repeatedly sweep the undecided
    // vertices (index order) and decide any vertex that already has at least
    // one decided neighbor, by majority edge weight toward each side. Vertices
    // with no decided neighbor yet are deferred to a later sweep. This is a
    // purely edge-local / one-hop rule -- no global objective is evaluated.
    vector<double> margin(n + 1, 0.0); // |w0 - w1| at decision time
    int remaining = n - 2;
    while (remaining > 0) {
        bool progressed = false;
        for (int v = 1; v <= n; v++) {
            if (decided[v]) continue;
            long long w0 = 0, w1 = 0;
            for (auto& e : adj[v]) {
                int u = e.first, w = e.second;
                if (!decided[u]) continue;
                if (side[u] == 0) w0 += w; else w1 += w;
            }
            if (w0 == 0 && w1 == 0) continue; // no signal yet, defer
            int s;
            if (w0 == w1) s = (nDecided0 <= nDecided1) ? 0 : 1; // tie: balance
            else s = (w0 > w1) ? 0 : 1;
            side[v] = s; decided[v] = 1;
            if (s == 0) nDecided0++; else nDecided1++;
            margin[v] = (double)llabs(w0 - w1);
            remaining--; progressed = true;
        }
        if (!progressed) {
            // components unreachable from the seeds (shouldn't happen given
            // connectivity, but stay safe): assign leftovers to smaller side.
            for (int v = 1; v <= n; v++) {
                if (decided[v]) continue;
                int s = (nDecided0 <= nDecided1) ? 0 : 1;
                side[v] = s; decided[v] = 1;
                if (s == 0) nDecided0++; else nDecided1++;
                margin[v] = 0.0;
                remaining--;
            }
        }
    }

    // balance fixup: repeatedly flip the least-confident vertex on the excess side
    auto count0 = [&]() {
        long long c = 0;
        for (int v = 1; v <= n; v++) if (side[v] == 0) c++;
        return c;
    };
    long long c0 = count0();
    while (c0 < lo || c0 > hi) {
        int wantSide = (c0 < lo) ? 0 : 1; // side we need MORE of -> flip a vertex on the other side into wantSide
        int fromSide = 1 - wantSide;
        int best = -1;
        for (int v = 1; v <= n; v++) {
            if (v == seed1 || v == seed2) continue; // keep seeds fixed
            if (side[v] != fromSide) continue;
            if (best == -1 || margin[v] < margin[best]) best = v;
        }
        if (best == -1) {
            // fall back to flipping a seed if truly stuck (rare, tiny n edge case)
            for (int v = 1; v <= n; v++) {
                if (side[v] != fromSide) continue;
                best = v; break;
            }
            if (best == -1) break;
        }
        side[best] = wantSide;
        c0 = count0();
    }

    for (int v = 1; v <= n; v++) printf("%d ", side[v]);
    printf("\n");
    return 0;
}
