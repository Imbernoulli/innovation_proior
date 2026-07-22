// TIER: greedy
// The obvious approach: pack all K lanes every cycle by topological readiness
// (standard list scheduling), and HOLD every value alive until its recorded
// true last use fires (never recompute -- once released a value is gone for
// good, so this greedy never releases early). It only self-throttles issuing
// a new step when doing so would blow the register file (computed exactly,
// so it never actually violates R -- it just leaves lanes idle instead).
// On braided instances this launches every chain in lockstep, so ALL chains'
// roots stay "still needed later" at once: register demand climbs toward
// ~2x the number of chains in flight, the file saturates, and lanes sit idle
// for long stretches waiting for some chain to reach its own reuse point.
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
    vector<int> indeg(n + 1), remaining(n + 1, 0);
    for (int v = 1; v <= n; v++) indeg[v] = (int)preds[v].size();
    for (int u = 1; u <= n; u++) remaining[u] = (int)succ[u].size();

    vector<char> live(n + 1, 0), computedEver(n + 1, 0);
    ll liveCount = 0;

    list<int> ready;
    for (int v = 1; v <= n; v++) if (indeg[v] == 0) ready.push_back(v);

    vector<string> outLines;
    ll TCAP = (ll)10 * n + 200;
    long long t = 0;
    while (!live[n] && t <= TCAP) {
        t++;
        vector<char> laneUsed(4, 0);
        vector<pair<int,int>> commits;
        vector<int> discardList;

        for (auto it = ready.begin(); it != ready.end(); ) {
            int v = *it;
            int ty = type[v];
            if (laneUsed[ty]) { ++it; continue; }
            int willDiscard = 0;
            for (int u : preds[v]) if (remaining[u] == 1) willDiscard++;
            ll newLive = liveCount + 1 - willDiscard;
            if (newLive <= R) {
                laneUsed[ty] = 1;
                commits.push_back({0, v});
                computedEver[v] = 1; live[v] = 1; liveCount++;
                for (int u : preds[v]) {
                    remaining[u]--;
                    if (remaining[u] == 0) discardList.push_back(u);
                }
                it = ready.erase(it);
            } else {
                ++it;
            }
        }
        for (int u : discardList) { live[u] = 0; liveCount--; }
        for (auto &pr : commits) {
            int v = pr.second;
            for (int w : succ[v]) { indeg[w]--; if (indeg[w] == 0) ready.push_back(w); }
        }

        string line = to_string((int)commits.size());
        for (auto &pr : commits) line += " " + to_string(pr.first) + " " + to_string(pr.second);
        line += " " + to_string((int)discardList.size());
        for (int u : discardList) line += " " + to_string(u);
        outLines.push_back(line);
    }

    printf("%d\n", (int)outLines.size());
    for (auto &s : outLines) printf("%s\n", s.c_str());
    return 0;
}
