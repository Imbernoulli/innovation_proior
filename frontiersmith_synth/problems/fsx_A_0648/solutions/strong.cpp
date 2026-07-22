// TIER: strong
// The insight: register pressure, not dependence height, is the bottleneck --
// so treat RELEASE-and-RECOMPUTE as a scheduling primitive instead of a last
// resort. Release every predecessor EAGERLY right after each individual use
// (not only at its true last use), and whenever an idle lane's worth of a
// step's own type is free, opportunistically re-make (op=1) any already-made,
// currently-released step that some pending step still needs (this is only
// ever profitable / legal-for-free for zero-predecessor "root" steps, which
// is exactly what the braided instances are built from). Because a value's
// footprint on the floor shrinks to almost nothing between uses, every chain
// can run in parallel from the start and all K lanes stay busy nearly the
// whole schedule -- a few cheap recompute instructions buy back a register
// spill cliff's worth of stalls.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m, R, K;
    scanf("%d %d %d %d", &n, &m, &R, &K);
    vector<int> type(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d", &type[i]);
    vector<vector<int>> preds(n + 1), succ(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        preds[v].push_back(u);
        succ[u].push_back(v);
    }
    vector<int> indeg(n + 1);
    for (int v = 1; v <= n; v++) indeg[v] = (int)preds[v].size();

    vector<char> live(n + 1, 0), computedEver(n + 1, 0);
    vector<int> avail(n + 1, -1);
    ll liveCount = 0;

    list<int> ready;
    for (int v = 1; v <= n; v++) if (indeg[v] == 0) ready.push_back(v);

    vector<string> outLines;
    ll TCAP = (ll)10 * n + 200;
    long long t = 0;
    while (!live[n] && t <= TCAP) {
        t++;
        vector<char> laneUsed(4, 0);
        vector<pair<int,int>> commits;   // (op, node)
        set<int> discardTracker;

        // Pass 1: commit any topo-ready step whose predecessors are all held.
        for (auto it = ready.begin(); it != ready.end(); ) {
            int v = *it;
            int ty = type[v];
            if (laneUsed[ty]) { ++it; continue; }
            bool ok = true;
            for (int u : preds[v]) if (!(live[u] && avail[u] <= t)) { ok = false; break; }
            if (ok && liveCount + 1 <= R) {
                laneUsed[ty] = 1;
                commits.push_back({0, v});
                computedEver[v] = 1; live[v] = 1; avail[v] = t + 1; liveCount++;
                for (int u : preds[v]) discardTracker.insert(u);
                it = ready.erase(it);
            } else {
                ++it;
            }
        }
        // Pass 2: with any lane still idle, re-make a released value some
        // pending step needs, so it will be ready to use NEXT cycle.
        for (auto it = ready.begin(); it != ready.end(); ++it) {
            int v = *it;
            for (int u : preds[v]) {
                if (computedEver[u] && !live[u]) {
                    int ty = type[u];
                    if (laneUsed[ty]) continue;
                    bool ok2 = true;
                    for (int w : preds[u]) if (!(live[w] && avail[w] <= t)) { ok2 = false; break; }
                    if (ok2 && liveCount + 1 <= R) {
                        laneUsed[ty] = 1;
                        commits.push_back({1, u});
                        live[u] = 1; avail[u] = t + 1; liveCount++;
                    }
                }
            }
        }

        for (int u : discardTracker) if (live[u]) { live[u] = 0; liveCount--; }
        for (auto &pr : commits) {
            if (pr.first == 0) {
                int v = pr.second;
                for (int w : succ[v]) { indeg[w]--; if (indeg[w] == 0) ready.push_back(w); }
            }
        }

        vector<int> discList(discardTracker.begin(), discardTracker.end());
        string line = to_string((int)commits.size());
        for (auto &pr : commits) line += " " + to_string(pr.first) + " " + to_string(pr.second);
        line += " " + to_string((int)discList.size());
        for (int u : discList) line += " " + to_string(u);
        outLines.push_back(line);
    }

    printf("%d\n", (int)outLines.size());
    for (auto &s : outLines) printf("%s\n", s.c_str());
    return 0;
}
