#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, R;
vector<int> cost;
vector<vector<int>> adj;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    R = inf.readInt();
    cost.assign(N + 1, 0);
    ll B = 0;
    for (int v = 1; v <= N; v++) { cost[v] = inf.readInt(); B += cost[v]; }
    adj.assign(N + 1, {});
    for (int i = 0; i < M; i++) {
        int u = inf.readInt(), v = inf.readInt();
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate the participant's set of towers ----
    int k = ouf.readInt(0, N, "k");
    vector<char> chosen(N + 1, 0);
    vector<int> towers;
    towers.reserve(k);
    ll F = 0;
    for (int i = 0; i < k; i++) {
        int v = ouf.readInt(1, N, "towerCell");
        if (chosen[v]) quitf(_wa, "tower cell %d listed more than once", v);
        chosen[v] = 1;
        towers.push_back(v);
        F += cost[v];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- coverage check via bounded BFS (depth R) from each tower ----
    // Global covered[] is only ever set to 1. A per-tower stamp array avoids re-visiting
    // a cell twice inside the same tower's BFS without O(N) resets between towers.
    vector<char> covered(N + 1, 0);
    vector<int> stamp(N + 1, -1);
    vector<int> dist(N + 1, 0);
    vector<int> frontier, nextf;
    frontier.reserve(1024);
    nextf.reserve(1024);

    for (int t : towers) {
        frontier.clear();
        stamp[t] = t; dist[t] = 0; covered[t] = 1;
        frontier.push_back(t);
        for (int d = 0; d < R; d++) {
            nextf.clear();
            for (int u : frontier) {
                for (int w : adj[u]) {
                    if (stamp[w] != t) {
                        stamp[w] = t;
                        dist[w] = d + 1;
                        covered[w] = 1;
                        nextf.push_back(w);
                    }
                }
            }
            frontier.swap(nextf);
        }
    }

    for (int u = 1; u <= N; u++)
        if (!covered[u])
            quitf(_wa, "cell %d is not covered by any watchtower within radius %d", u, R);

    // ---- score: minimization, ratio = min(1, 0.1 * B / F) ----
    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
