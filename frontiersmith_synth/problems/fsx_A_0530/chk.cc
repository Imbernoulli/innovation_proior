#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Firebreaks Ahead of the Blaze".
//
// Input:  N M B S / w_1..w_N / S sources / M edges (u v).
// Output: K ; then K lines "c a_1..a_c" (0<=c<=B), the stands protected at step t.
//
// Simulation. burnt = sources at step 0. For step t = 1,2,...:
//   (protect) apply the participant's list P_t: each id must be currently unburnt and not
//             already protected, else the output is INFEASIBLE (score 0).
//   (spread)  every currently-burning stand ignites each neighbour that is unburnt and
//             unprotected. Spreads from the last frontier (standard BFS layer).
// Runs until the frontier is empty AND all K listed steps have been applied.
//
// F (MAX) = sum of w_v over stands not burning at quiescence.
// Baseline B0 = do-nothing saved value = sum of w_v over stands unreachable from the sources
//   (computed by a protection-free BFS). Generator guarantees B0 > 0.
// Score: sc = min(1000, 100 * F / max(1,B0)); ratio = sc/1000 (do-nothing -> 0.1, cap 1.0).
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int B = inf.readInt();
    int S = inf.readInt();
    vector<ll> w(N + 1);
    for (int i = 1; i <= N; i++) w[i] = inf.readLong();
    vector<int> src(S);
    for (int i = 0; i < S; i++) src[i] = inf.readInt();
    vector<vector<int>> adj(N + 1);
    for (int e = 0; e < M; e++){
        int u = inf.readInt(1, N, "u");
        int v = inf.readInt(1, N, "v");
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // ---- baseline B0: value unreachable from sources (protection-free BFS) ----
    {
        vector<char> reach(N + 1, 0);
        queue<int> q;
        for (int s : src){ if (!reach[s]){ reach[s] = 1; q.push(s); } }
        while (!q.empty()){
            int u = q.front(); q.pop();
            for (int v : adj[u]) if (!reach[v]){ reach[v] = 1; q.push(v); }
        }
        ll B0 = 0;
        for (int i = 1; i <= N; i++) if (!reach[i]) B0 += w[i];
        if (B0 <= 0) B0 = 1;

        // ---- read participant schedule ----
        int K = ouf.readInt(0, N, "K");
        vector<vector<int>> prot(K + 1);
        for (int t = 1; t <= K; t++){
            int c = ouf.readInt(0, B, "count");
            for (int j = 0; j < c; j++) prot[t].push_back(ouf.readInt(1, N, "vid"));
        }
        if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the schedule");

        // ---- simulate ----
        vector<char> burnt(N + 1, 0), prt(N + 1, 0);
        vector<int> frontier;
        for (int s : src){ if (!burnt[s]){ burnt[s] = 1; frontier.push_back(s); } }

        int t = 1;
        while (true){
            bool doProt   = (t <= K);
            bool doSpread = !frontier.empty();
            if (!doProt && !doSpread) break;
            if (doProt){
                for (int id : prot[t]){
                    if (burnt[id]) quitf(_wa, "step %d: stand %d already burned", t, id);
                    if (prt[id])   quitf(_wa, "step %d: stand %d protected twice", t, id);
                    prt[id] = 1;
                }
            }
            if (doSpread){
                vector<int> nxt;
                for (int u : frontier)
                    for (int v : adj[u])
                        if (!burnt[v] && !prt[v]){ burnt[v] = 1; nxt.push_back(v); }
                frontier.swap(nxt);
            }
            t++;
        }

        ll F = 0;
        for (int i = 1; i <= N; i++) if (!burnt[i]) F += w[i];

        double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B0));
        quitp(sc / 1000.0, "OK F=%lld B0=%lld Ratio: %.6f", F, B0, sc / 1000.0);
    }
    return 0;
}
