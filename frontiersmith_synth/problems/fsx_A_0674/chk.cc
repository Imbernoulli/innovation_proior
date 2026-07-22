#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Checker for "Blizzard Fleet Bootstrap".
// Each crew's own move list carries EXPLICIT start times (participant-chosen,
// integer) so crews can synchronize (a crew may delay a move to let a
// teammate clear a shared street first). We validate, per crew, that moves
// are time-ordered and spatially connected, that every street is P-moded
// (cleared) exactly once across the whole output, and that every F-mode use
// of a street starts no earlier than that street's unique clearing finishes.

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int K = inf.readInt();
    vector<int> U(M + 1), V(M + 1);
    vector<ll> LEN(M + 1);
    ll sumLen = 0;
    for (int i = 1; i <= M; i++) {
        U[i] = inf.readInt(1, N, "u");
        V[i] = inf.readInt(1, N, "v");
        LEN[i] = inf.readLong(1LL, 1000000LL, "len");
        sumLen += LEN[i];
    }
    ll SLOW = inf.readLong(1LL, 1000000LL, "slow");
    ll FAST = inf.readLong(1LL, 1000000LL, "fast");
    int depot = inf.readInt(1, N, "depot");

    // moves[p] entries: edge_id, mode(0=P,1=F), start, finish
    vector<vector<array<ll, 4>>> moves(K + 1);
    vector<ll> finish(K + 1, 0);
    vector<int> pClearCount(M + 1, 0);
    vector<ll> clearTime(M + 1, -1);

    const ll TIME_CAP = (ll)2e15;

    for (int p = 1; p <= K; p++) {
        int m = ouf.readInt(0, 5000, "m_p");
        int pos = depot;
        ll lastFinish = 0;
        for (int j = 0; j < m; j++) {
            int e = ouf.readInt(1, M, "edge_id");
            string modeStr = ouf.readToken();
            if (modeStr != "P" && modeStr != "F")
                quitf(_wa, "crew %d move %d: bad mode token '%s'", p, j, modeStr.c_str());
            char mode = modeStr[0];
            ll st = ouf.readLong(0LL, TIME_CAP, "start_time");
            if (st < lastFinish)
                quitf(_wa, "crew %d move %d: start_time %lld precedes previous finish %lld",
                      p, j, st, lastFinish);
            int u = U[e], v = V[e];
            int npos;
            if (u == pos) npos = v;
            else if (v == pos) npos = u;
            else quitf(_wa, "crew %d move %d: street %d not incident to current position %d",
                       p, j, e, pos);
            ll dur = (mode == 'P') ? LEN[e] * SLOW : LEN[e] * FAST;
            ll fin = st + dur;
            moves[p].push_back({(ll)e, (mode == 'P' ? 0LL : 1LL), st, fin});
            if (mode == 'P') pClearCount[e]++;
            pos = npos;
            lastFinish = fin;
        }
        finish[p] = lastFinish;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in output");

    for (int e = 1; e <= M; e++) {
        if (pClearCount[e] != 1)
            quitf(_wa, "street %d cleared %d times (must be exactly once)", e, pClearCount[e]);
    }
    for (int p = 1; p <= K; p++)
        for (auto &mv : moves[p])
            if (mv[1] == 0) clearTime[mv[0]] = mv[3];

    for (int p = 1; p <= K; p++) {
        for (auto &mv : moves[p]) {
            if (mv[1] == 1) {
                ll e = mv[0];
                if (clearTime[e] < 0 || mv[2] < clearTime[e])
                    quitf(_wa, "crew %d: street %lld used fast at %lld before it was cleared (at %lld)",
                          p, e, mv[2], clearTime[e]);
            }
        }
    }

    ll MAKESPAN = 0, TOTAL = 0;
    for (int p = 1; p <= K; p++) { MAKESPAN = max(MAKESPAN, finish[p]); TOTAL += finish[p]; }
    const ll C = 4; // weight MAKESPAN heavily: TOTAL alone barely moves with a
                     // better partition (total fleet work is roughly
                     // conserved), so the objective must lean on MAKESPAN to
                     // actually reward balancing it.
    ll F = C * MAKESPAN * (ll)K + TOTAL;
    if (F <= 0) F = 1;

    // Internal baseline: one crew, alone, clears the depot's single trunk
    // street, then walks out-and-back through the trunk to every other
    // street's neighborhood one at a time (never staying out to chain two
    // neighborhoods together). Naive but always feasible for this input
    // shape: the depot has one incident street (the trunk); everything else
    // hangs off its far endpoint through a set of further incident streets
    // (each such street plus everything reachable beyond it, without
    // recrossing it, is one "neighborhood").
    vector<vector<pair<int, int>>> adj(N + 1);
    for (int i = 1; i <= M; i++) {
        adj[U[i]].push_back({V[i], i});
        adj[V[i]].push_back({U[i], i});
    }
    ll B;
    if (!adj[depot].empty()) {
        int trunkEid = adj[depot][0].second;
        int hub = adj[depot][0].first;
        ll trunkLen = LEN[trunkEid];
        vector<pair<ll, ll>> nbhd; // (branchLen, localSum) per neighborhood, in adjacency order
        for (auto &pr : adj[hub]) {
            int v = pr.first, eid = pr.second;
            if (eid == trunkEid) continue;
            ll localSum = 0;
            vector<char> vis(N + 1, 0), visE(M + 1, 0);
            queue<int> q;
            vis[v] = 1;
            q.push(v);
            while (!q.empty()) {
                int x = q.front(); q.pop();
                for (auto &pr2 : adj[x]) {
                    int y = pr2.first, e2 = pr2.second;
                    if (e2 == eid) continue;
                    if (!visE[e2]) { visE[e2] = 1; localSum += LEN[e2]; }
                    if (!vis[y]) { vis[y] = 1; q.push(y); }
                }
            }
            nbhd.push_back({LEN[eid], localSum});
        }
        int R = (int)nbhd.size();
        ll localTotal = 0, spokeTotal = 0;
        for (auto &nb : nbhd) localTotal += (SLOW + FAST) * nb.second;
        for (int r = 0; r < R; r++) {
            bool isLast = (r == R - 1);
            spokeTotal += isLast ? nbhd[r].first * SLOW : nbhd[r].first * (SLOW + FAST);
        }
        ll trunkCost = trunkLen * SLOW + (ll)max(0, R - 1) * 2 * trunkLen * FAST;
        B = (localTotal + spokeTotal + trunkCost) * (C * (ll)K + 1);
    } else {
        B = (SLOW + FAST) * sumLen * (C * (ll)K + 1);
    }
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
