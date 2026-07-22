#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Win a two-sided idea war on a network".
//
// Input:  N M K S ; N values ; M edges (u v) ; S rival(B)-seeded node ids.
// Output: K distinct node ids in [1,N], none among the rival-seeded ids -- the
//   solver's own (A) seeds.
//
// Dynamics: synchronous majority-threshold contagion to a fixed point. Every
//   node starts B (if rival-seeded), A (if solver-seeded) or uncoloured.
//   Each round, EVERY currently uncoloured node looks at its neighbours' colours
//   as of the START of the round: if strictly more are A than B it becomes A; if
//   strictly more are B than A it becomes B; otherwise (tie, incl. 0-0) it stays
//   uncoloured. All updates in a round are applied simultaneously. Repeat until
//   no node changes (guaranteed within N rounds since each round that changes
//   anything commits >=1 node permanently).
//
// Objective (MAX): F = sum of values of nodes coloured A at the fixed point.
//
// Baseline B (checker-computed, matches the "trivial" reference exactly): walk
//   arenas (connected components) in ascending lowest-available-id order and
//   take one HIGHEST-id available node per arena (structurally always a
//   market node, never a control -- generation order lists controls first)
//   until K are chosen, run the SAME dynamics, B = its resulting F. One seed
//   per arena never triggers a cascade (see below) -- this baseline is the
//   "grab whatever comes first, no network awareness" default a solver would
//   try before noticing the pair-of-controls effect.
// Score (max): sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static int N, M, K, S;
static vector<ll> val;
static vector<vector<int>> adj;
static vector<char> isRival;
static vector<int> dsuPar;
static int dsuFind(int x){ return dsuPar[x] == x ? x : dsuPar[x] = dsuFind(dsuPar[x]); }
static void dsuUnion(int a, int b){ a = dsuFind(a); b = dsuFind(b); if (a != b) dsuPar[a] = b; }

// colors: 0=uncoloured, 1=A, 2=B. seedA = set of ids to seed A (already validated
// disjoint from rival ids and, by caller, within [1,N] and distinct).
static ll simulate(const vector<int>& seedA){
    vector<char> color(N + 1, 0);
    for (int i = 1; i <= N; i++) if (isRival[i]) color[i] = 2;
    for (int id : seedA) color[id] = 1;

    vector<char> cur = color;
    int rounds = 0, cap = N + 3;
    while (rounds++ < cap){
        vector<char> nxt = cur;
        bool changed = false;
        for (int v = 1; v <= N; v++){
            if (cur[v] != 0) continue;
            int cA = 0, cB = 0;
            for (int u : adj[v]){
                if (cur[u] == 1) cA++;
                else if (cur[u] == 2) cB++;
            }
            if (cA > cB) { nxt[v] = 1; changed = true; }
            else if (cB > cA) { nxt[v] = 2; changed = true; }
        }
        cur.swap(nxt);
        if (!changed) break;
    }
    ll F = 0;
    for (int v = 1; v <= N; v++) if (cur[v] == 1) F += val[v];
    return F;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    K = inf.readInt();
    S = inf.readInt();
    val.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) val[i] = inf.readLong();

    adj.assign(N + 1, {});
    dsuPar.resize(N + 1);
    for (int i = 1; i <= N; i++) dsuPar[i] = i;
    for (int i = 0; i < M; i++){
        int u = inf.readInt(1, N, "u");
        int v = inf.readInt(1, N, "v");
        if (u == v) quitf(_fail, "generator produced a self loop"); // input sanity
        adj[u].push_back(v);
        adj[v].push_back(u);
        dsuUnion(u, v);
    }
    isRival.assign(N + 1, 0);
    vector<int> rivalIds(S);
    for (int i = 0; i < S; i++){
        int b = inf.readInt(1, N, "rival id");
        rivalIds[i] = b;
        isRival[b] = 1;
    }

    // ---- internal baseline: one HIGHEST-id available node per arena, arenas
    //      visited in ascending lowest-available-id order, until K are chosen ----
    map<int, int> arenaLowest, arenaHighest;   // dsu root -> lowest / highest available id
    for (int v = 1; v <= N; v++){
        if (isRival[v]) continue;
        int r = dsuFind(v);
        auto itl = arenaLowest.find(r);
        if (itl == arenaLowest.end() || v < itl->second) arenaLowest[r] = v;
        auto ith = arenaHighest.find(r);
        if (ith == arenaHighest.end() || v > ith->second) arenaHighest[r] = v;
    }
    vector<pair<int,int>> avail;   // (lowest id for ordering, highest id to seed)
    for (auto &kv : arenaLowest) avail.push_back({kv.second, arenaHighest[kv.first]});
    sort(avail.begin(), avail.end());
    if ((int)avail.size() < K) quitf(_fail, "generator did not leave K arenas to seed from");
    vector<int> baseSeed;
    for (int i = 0; i < K; i++) baseSeed.push_back(avail[i].second);
    ll B = simulate(baseSeed);
    if (B <= 0) B = 1;

    // ---- read & validate the participant's K seed ids ----
    vector<int> seedA(K);
    vector<char> chosen(N + 1, 0);
    for (int i = 0; i < K; i++){
        int id = ouf.readInt(1, N, "seed id");
        if (chosen[id]) quitf(_wa, "node %d chosen more than once", id);
        if (isRival[id]) quitf(_wa, "node %d is already rival-seeded, cannot be chosen", id);
        chosen[id] = 1;
        seedA[i] = id;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after %d seed ids", K);

    ll F = simulate(seedA);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
