// TIER: strong
// Critical-path priority list scheduling.  Compute latency-weighted bottom levels
// bl[i] = L[i] + max over successors of bl[succ] (longest remaining latency path).  Step
// cycle by cycle; among ready ops (all preds scheduled, latency satisfied) pack the most
// critical first, subject to issue width, per-kind unit caps and the read-port budget.
// This shortens the makespan-gating path and diverges from in-order packing on trap/needle
// tests.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, W, T, P;
    if (scanf("%d %d %d %d %d", &n, &m, &W, &T, &P) != 5) return 0;
    vector<int> cap(T + 1), rp(T + 1);
    for (int t = 1; t <= T; t++) scanf("%d", &cap[t]);
    for (int t = 1; t <= T; t++) scanf("%d", &rp[t]);
    vector<int> type(n + 1), L(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d %d", &type[i], &L[i]);
    vector<vector<int>> succ(n + 1);
    vector<int> indeg(n + 1, 0);
    for (int j = 0; j < m; j++) {
        int u, v; scanf("%d %d", &u, &v);
        succ[u].push_back(v);
        indeg[v]++;
    }

    // bottom levels (nodes are in topological order 1..n, successors have larger id)
    vector<long long> bl(n + 1, 0);
    for (int i = n; i >= 1; i--) {
        long long best = 0;
        for (int v : succ[i]) best = max(best, bl[v]);
        bl[i] = (long long)L[i] + best;
    }

    vector<long long> s(n + 1, -1), earliest(n + 1, 0);
    vector<int> indegRun = indeg;
    vector<char> done(n + 1, 0);

    // eligible = ops whose predecessors are all scheduled (but latency may push them later)
    vector<int> elig;
    for (int i = 1; i <= n; i++) if (indegRun[i] == 0) elig.push_back(i);

    long long c = 0;
    int scheduled = 0;
    vector<int> cand;
    while (scheduled < n) {
        cand.clear();
        for (int x : elig) if (earliest[x] <= c) cand.push_back(x);
        // most critical first; ties: earlier ready, then lower id
        sort(cand.begin(), cand.end(), [&](int a, int b) {
            if (bl[a] != bl[b]) return bl[a] > bl[b];
            if (earliest[a] != earliest[b]) return earliest[a] < earliest[b];
            return a < b;
        });

        int used = 0;
        vector<int> tcnt(T + 1, 0);
        long long ports = 0;
        vector<char> placedNow(n + 1, 0);
        vector<int> newReady;

        for (int x : cand) {
            int t = type[x];
            if (used >= W) break;
            if (tcnt[t] >= cap[t]) continue;
            if (ports + rp[t] > P) continue;
            // place x at cycle c
            s[x] = c;
            done[x] = 1;
            placedNow[x] = 1;
            used++; tcnt[t]++; ports += rp[t];
            scheduled++;
            for (int v : succ[x]) {
                earliest[v] = max(earliest[v], c + (long long)L[x]);
                if (--indegRun[v] == 0) newReady.push_back(v);
            }
        }

        // rebuild eligible = surviving unscheduled + newly ready
        vector<int> nextElig;
        nextElig.reserve(elig.size());
        for (int x : elig) if (!placedNow[x]) nextElig.push_back(x);
        for (int v : newReady) nextElig.push_back(v);
        elig.swap(nextElig);

        c++;
    }

    for (int i = 1; i <= n; i++) printf("%lld\n", s[i]);
    return 0;
}
