// TIER: strong
// Seeded multi-restart local-move descent. From several initial labelings (greedy +
// randomized), repeatedly relabel any pool to the niche minimizing conflict over ALL
// its neighbors until no single move improves; keep the lowest-conflict assignment.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, C;
vector<int> head, nxt, to, wt; // CSR-ish adjacency via linked lists

ll conflict(const vector<int>& lab, const vector<int>& eu, const vector<int>& ev, const vector<int>& ew) {
    ll F = 0;
    for (int i = 0; i < m; i++) if (lab[eu[i]] == lab[ev[i]]) F += ew[i];
    return F;
}

int main() {
    if (scanf("%d %d %d", &n, &m, &C) != 3) return 0;
    vector<int> eu(m), ev(m), ew(m);
    // adjacency as vectors
    vector<vector<pair<int,int>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        scanf("%d %d %d", &eu[i], &ev[i], &ew[i]);
        adj[eu[i]].push_back({ev[i], ew[i]});
        adj[ev[i]].push_back({eu[i], ew[i]});
    }

    mt19937 rng(0xC0FFEEu ^ (unsigned)(n * 1000003 + m * 7 + C));

    // one full sweep of best-move descent; returns true if anything changed
    vector<ll> cost(C + 1);
    auto descend = [&](vector<int>& lab) {
        bool improvedAny = true;
        int sweeps = 0;
        int maxSweeps = 60;
        while (improvedAny && sweeps++ < maxSweeps) {
            improvedAny = false;
            for (int i = 1; i <= n; i++) {
                for (int c = 1; c <= C; c++) cost[c] = 0;
                for (auto& e : adj[i]) cost[lab[e.first]] += e.second;
                int cur = lab[i];
                int best = cur; ll bv = cost[cur];
                for (int c = 1; c <= C; c++)
                    if (cost[c] < bv) { bv = cost[c]; best = c; }
                if (best != cur) { lab[i] = best; improvedAny = true; }
            }
        }
    };

    vector<int> bestLab(n + 1, 1);
    ll bestF = conflict(bestLab, eu, ev, ew); // all-1 baseline

    // restart 0: greedy sequential start, then descent
    {
        vector<int> lab(n + 1, 0);
        for (int i = 1; i <= n; i++) {
            for (int c = 1; c <= C; c++) cost[c] = 0;
            for (auto& e : adj[i]) if (lab[e.first]) cost[lab[e.first]] += e.second;
            int best = 1; ll bv = cost[1];
            for (int c = 2; c <= C; c++) if (cost[c] < bv) { bv = cost[c]; best = c; }
            lab[i] = best;
        }
        descend(lab);
        ll f = conflict(lab, eu, ev, ew);
        if (f < bestF) { bestF = f; bestLab = lab; }
    }

    // budget restarts by size so we stay well within the time limit
    int restarts = 6;
    if ((ll)n * (ll)max(1, m) > 20000000LL) restarts = 3;
    for (int r = 0; r < restarts; r++) {
        vector<int> lab(n + 1);
        for (int i = 1; i <= n; i++) lab[i] = (int)(rng() % C) + 1;
        descend(lab);
        ll f = conflict(lab, eu, ev, ew);
        if (f < bestF) { bestF = f; bestLab = lab; }
    }

    string out; out.reserve(n * 2);
    char buf[16];
    for (int i = 1; i <= n; i++) { int len = sprintf(buf, "%d\n", bestLab[i]); out.append(buf, len); }
    fputs(out.c_str(), stdout);
    return 0;
}
