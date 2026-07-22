#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Ridgeline Valley: Planting the New Rootstock".
//
// Input:  n m R tau ; then n lines "p_i c_i s_i" ; then m lines "u v w"
//         (undirected trust tie, weight w).
// Output: k ; then k distinct household ids (1-indexed) -- the zealot
//         ("seed grower") set, fixed to NEW (opinion 1) from season 0 on.
//
// Feasibility: 0<=k<=n; ids in [1,n], all distinct; exactly k+1 integers,
//   nothing trailing. Simulate exactly R seasons of SYNCHRONOUS weighted
//   majority dynamics (non-zealots only; zealots stay at 1 throughout):
//     for household i, w1 = sum over trust ties (i,j) of weight*opinion_prev(j),
//     w0 = degW(i) - w1 (degW = total weight of i's ties). new opinion:
//       w1>w0 -> 1 ; w1<w0 -> 0 ; w1==w0 -> unchanged (keeps prev opinion).
//   All updates read the SAME previous-season snapshot (synchronous).
//   Let A = total acreage (sum p_i) over households with FINAL opinion 1.
//   Feasible iff 100*A >= tau * totalAcreage. Infeasible (or malformed)
//   output scores 0.
//
// Objective (MIN): cost = sum of c_i over the zealot set. Minimize, subject
//   to feasibility.
//
// Baseline B (checker-computed, always-feasible do-everything reference):
//   sponsor EVERY household as a zealot. Then every household is opinion 1
//   from season 0 onward regardless of R, so A = totalAcreage >= tau% for
//   any tau<=100 -- always feasible. B = sum of ALL c_i.
//   Score (min): sc = min(1000, 100*B/max(1,cost)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int R = inf.readInt();
    int tau = inf.readInt();

    vector<ll> p(n + 1), c(n + 1);
    vector<int> s(n + 1);
    ll B = 0, totalAcreage = 0;
    for (int i = 1; i <= n; i++){
        p[i] = inf.readLong();
        c[i] = inf.readLong();
        s[i] = inf.readInt();
        B += c[i];
        totalAcreage += p[i];
    }
    if (B <= 0) B = 1;
    if (totalAcreage <= 0) totalAcreage = 1;

    vector<vector<pair<int,int>>> adj(n + 1);   // (neighbor, weight)
    vector<ll> degW(n + 1, 0);
    for (int e = 0; e < m; e++){
        int u = inf.readInt();
        int v = inf.readInt();
        int w = inf.readInt();
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        degW[u] += w;
        degW[v] += w;
    }

    // ---- read & validate participant output ----
    int k = ouf.readInt(0, n, "k");
    vector<int> zealots(k);
    vector<char> isZ(n + 1, 0), seen(n + 1, 0);
    for (int i = 0; i < k; i++){
        int z = ouf.readInt(1, n, "zealot id");
        if (seen[z]) quitf(_wa, "zealot id %d listed more than once", z);
        seen[z] = 1;
        isZ[z] = 1;
        zealots[i] = z;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the zealot list");

    // ---- simulate R seasons of synchronous weighted majority dynamics ----
    vector<int> cur(n + 1), nxt(n + 1);
    for (int i = 1; i <= n; i++) cur[i] = isZ[i] ? 1 : s[i];
    for (int t = 0; t < R; t++){
        for (int i = 1; i <= n; i++){
            if (isZ[i]){ nxt[i] = 1; continue; }
            ll w1 = 0;
            for (auto &pr : adj[i]) w1 += (ll)pr.second * cur[pr.first];
            ll w0 = degW[i] - w1;
            if (w1 > w0) nxt[i] = 1;
            else if (w1 < w0) nxt[i] = 0;
            else nxt[i] = cur[i];
        }
        swap(cur, nxt);
    }

    ll newAcreage = 0;
    for (int i = 1; i <= n; i++) if (cur[i] == 1) newAcreage += p[i];

    if (newAcreage * 100 < (ll)tau * totalAcreage)
        quitf(_wa, "coverage target missed: %.3f%% NEW acreage, need %d%% (k=%d zealots)",
              100.0 * (double)newAcreage / (double)totalAcreage, tau, k);

    ll cost = 0;
    for (int z : zealots) cost += c[z];

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, cost));
    quitp(sc / 1000.0, "OK cost=%lld B=%lld Ratio: %.6f", cost, B, sc / 1000.0);
    return 0;
}
