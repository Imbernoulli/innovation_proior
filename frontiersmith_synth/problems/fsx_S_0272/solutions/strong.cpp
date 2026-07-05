// TIER: strong
// Randomised multi-restart local search combining balance-guarded flip moves with
// balance-preserving swap moves (so it keeps improving even when the headcount band is
// tight and single flips are forbidden). Deltas are maintained incrementally; the best
// split over all restarts is emitted.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, L;
vector<vector<pair<int,ll>>> adj;
vector<int> eu, ev; vector<ll> ew;
vector<int> side;                       // 0 = Habitat A, 1 = Habitat B
vector<ll> ds, dopp;                    // deg to same / opposite side
int aCount;                             // number in Habitat A (side 0)

void recompute() {
    fill(ds.begin(), ds.end(), 0);
    fill(dopp.begin(), dopp.end(), 0);
    aCount = 0;
    for (int u = 1; u <= n; u++) if (side[u] == 0) aCount++;
    for (int u = 1; u <= n; u++)
        for (auto& pr : adj[u]) {
            if (side[pr.first] == side[u]) ds[u] += pr.second;
            else dopp[u] += pr.second;
        }
}

// flip u, updating deltas of its neighbours incrementally
void applyFlip(int u) {
    int old = side[u];
    for (auto& pr : adj[u]) {
        int v = pr.first; ll w = pr.second;
        if (side[v] == old) { ds[v] -= w; dopp[v] += w; }
        else                { ds[v] += w; dopp[v] -= w; }
    }
    swap(ds[u], dopp[u]);
    side[u] ^= 1;
    aCount += (old == 0 ? -1 : +1);     // left A or joined A
}

bool bandOkAfterFlip(int u) {
    int na = aCount + (side[u] == 0 ? -1 : +1);
    return llabs((ll)na - (ll)(n - na)) <= L;
}

ll cutOf() {
    ll c = 0;
    for (int i = 0; i < m; i++) if (side[eu[i]] != side[ev[i]]) c += ew[i];
    return c;
}

int main() {
    if (scanf("%d %d %d", &n, &m, &L) != 3) return 0;
    adj.assign(n + 1, {});
    eu.resize(m); ev.resize(m); ew.resize(m);
    for (int i = 0; i < m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        eu[i] = u; ev[i] = v; ew[i] = w;
        adj[u].push_back({v, w}); adj[v].push_back({u, w});
    }
    side.assign(n + 1, 0); ds.assign(n + 1, 0); dopp.assign(n + 1, 0);

    mt19937 rng(0xA0C0FFEEu);
    vector<int> best; ll bestCut = -1;
    int t = n / 2;                       // balanced target size for random restarts

    int restarts = 18;
    for (int r = 0; r < restarts; r++) {
        // ---- initialise ----
        if (r == 0) {
            for (int i = 1; i <= n; i++) side[i] = (i <= t) ? 0 : 1;   // contiguous
        } else {
            vector<int> idx(n);
            for (int i = 0; i < n; i++) idx[i] = i + 1;
            shuffle(idx.begin(), idx.end(), rng);
            for (int i = 0; i < n; i++) side[idx[i]] = (i < t) ? 0 : 1; // random balanced
        }
        recompute();

        // ---- local search ----
        for (int round = 0; round < 60; round++) {
            bool improved = false;

            // flip pass (band-guarded, steepest per node)
            for (int u = 1; u <= n; u++) {
                ll gain = ds[u] - dopp[u];
                if (gain > 0 && bandOkAfterFlip(u)) { applyFlip(u); improved = true; }
            }

            // swap sampling (balance-preserving: one crew of A with one of B)
            vector<int> As, Bs;
            for (int u = 1; u <= n; u++) (side[u] == 0 ? As : Bs).push_back(u);
            if (!As.empty() && !Bs.empty()) {
                int samples = 2 * n;
                for (int s = 0; s < samples; s++) {
                    int u = As[rng() % As.size()];
                    int v = Bs[rng() % Bs.size()];
                    if (side[u] != 0 || side[v] != 1) continue;  // stale after earlier swaps
                    ll g1 = ds[u] - dopp[u];
                    applyFlip(u);                                // temporarily move u to B
                    ll g2 = ds[v] - dopp[v];
                    if (g1 + g2 > 0) { applyFlip(v); improved = true; }
                    else             { applyFlip(u); }           // revert
                }
            }

            if (!improved) break;
        }

        ll c = cutOf();
        if (c > bestCut) {
            bestCut = c;
            best.clear();
            for (int i = 1; i <= n; i++) if (side[i] == 0) best.push_back(i);
        }
    }

    printf("%d\n", (int)best.size());
    for (int x : best) printf("%d\n", x);
    return 0;
}
