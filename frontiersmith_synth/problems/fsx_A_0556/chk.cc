#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Frozen Tables, Severed Cables".
//
// Input:  n m ; then m lines  u v   (undirected, connected, simple).
// Output: for each destination d = 1..n, and for each source u = 1..n with u != d
//         in increasing u, two integers  p b  = primary and backup next-hop of u
//         toward d. Both p and b must be neighbors of u (b may equal p).
//
// Forwarding under a single cut link e (per destination d):
//   at router w != d take link (w,p); if that link is e, take (w,b) instead; if
//   THAT link is also e (only when b==p) the packet is dropped. Reaching d = ok.
//   The forwarding map is deterministic, so a source delivers iff it does not fall
//   into a cycle / drop -- checked as "reach d" in the induced functional graph.
//
// Objective (MIN): U(e) = # stranded ordered pairs under cut e ; F = max_e U(e).
// Baseline B (checker-built): the SAME quantity for the reference routing where
//   primary(w,d) = the neighbor of w nearest to d (smallest index on ties) and
//   backup = primary (no backup). This is exactly shortest-path forwarding.
// Score (min): sc = min(1000, 100 * B / max(1,F)); ratio = sc/1000  in [0,1].
//   Reference -> ~0.1 ; you gain by pushing your worst-case F far below B.
// -----------------------------------------------------------------------------

int n, m;
vector<vector<int>> adj;                 // neighbor lists
vector<vector<char>> isadj;              // adjacency matrix (n small)
vector<int> ex, ey;                      // edge endpoints

// distances / reference primary, indexed [d][u]
vector<vector<int>> dist_;
vector<vector<int>> refP;

// working stamps for the functional-graph reachability
vector<int> memoStamp, memoVal, pathStamp;
int stampG = 0;

static inline bool edgeEq(int w, int z, int x, int y){
    return (w == x && z == y) || (w == y && z == x);
}

// number of sources u != d that DELIVER to d under cut {x,y}, given tables P,Bk
// (each a vector<int> of size n+1 giving next-hops toward d).
ll deliverCount(const vector<int>& P, const vector<int>& Bk, int d, int x, int y){
    ll cnt = 0;
    static vector<int> path;
    for (int s = 1; s <= n; s++){
        if (s == d) continue;
        // resolve fate of s (memoized within this (d,e) via stampG)
        if (memoStamp[s] == stampG){ if (memoVal[s]) cnt++; continue; }
        path.clear();
        int w = s, res = -1;
        while (true){
            if (w == d){ res = 1; break; }
            if (memoStamp[w] == stampG){ res = memoVal[w]; break; }
            if (pathStamp[w] == stampG){ res = 0; break; }   // loop
            pathStamp[w] = stampG;
            path.push_back(w);
            int p = P[w];
            int z;
            if (edgeEq(w, p, x, y)){
                int b = Bk[w];
                if (edgeEq(w, b, x, y)){ z = -1; }           // backup also cut
                else z = b;
            } else z = p;
            if (z == -1){ res = 0; break; }
            w = z;
        }
        for (int nd : path){ memoStamp[nd] = stampG; memoVal[nd] = res; }
        if (res == 1) cnt++;
    }
    return cnt;
}

// F = max over all cut edges of the total stranded ordered pairs, for the given
// per-destination tables P[d], Bk[d].
ll worstStranded(const vector<vector<int>>& P, const vector<vector<int>>& Bk){
    ll best = 0;
    for (int e = 0; e < m; e++){
        int x = ex[e], y = ey[e];
        ll strandedTot = 0;
        for (int d = 1; d <= n; d++){
            ++stampG;
            ll deliv = deliverCount(P[d], Bk[d], d, x, y);
            strandedTot += (ll)(n - 1) - deliv;
        }
        if (strandedTot > best) best = strandedTot;
    }
    return best;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    adj.assign(n + 1, {});
    isadj.assign(n + 1, vector<char>(n + 1, 0));
    ex.resize(m); ey.resize(m);
    for (int i = 0; i < m; i++){
        int u = inf.readInt(1, n);
        int v = inf.readInt(1, n);
        ex[i] = u; ey[i] = v;
        adj[u].push_back(v);
        adj[v].push_back(u);
        isadj[u][v] = isadj[v][u] = 1;
    }

    // ---- reference shortest-path routing (baseline B) ----
    dist_.assign(n + 1, vector<int>(n + 1, -1));
    refP.assign(n + 1, vector<int>(n + 1, 0));
    for (int d = 1; d <= n; d++){
        vector<int>& dd = dist_[d];
        queue<int> q; dd[d] = 0; q.push(d);
        while (!q.empty()){
            int u = q.front(); q.pop();
            for (int v : adj[u]) if (dd[v] < 0){ dd[v] = dd[u] + 1; q.push(v); }
        }
        // primary = neighbor minimizing (dist, id)
        for (int u = 1; u <= n; u++){
            if (u == d) continue;
            int best = -1;
            for (int v : adj[u]){
                if (best == -1 || dd[v] < dd[best] || (dd[v] == dd[best] && v < best))
                    best = v;
            }
            refP[d][u] = best;   // for a connected graph every u!=d has such a nbr
        }
    }

    memoStamp.assign(n + 1, 0);
    memoVal.assign(n + 1, 0);
    pathStamp.assign(n + 1, 0);

    // ---- read participant tables : part_P[d][u], part_B[d][u] ----
    vector<vector<int>> pP(n + 1, vector<int>(n + 1, 0));
    vector<vector<int>> pB(n + 1, vector<int>(n + 1, 0));
    for (int d = 1; d <= n; d++){
        for (int u = 1; u <= n; u++){
            if (u == d) continue;
            int p = ouf.readInt(1, n, "primary");
            int b = ouf.readInt(1, n, "backup");
            if (!isadj[u][p])
                quitf(_wa, "primary %d for (u=%d,d=%d) is not a neighbor of %d", p, u, d, u);
            if (!isadj[u][b])
                quitf(_wa, "backup %d for (u=%d,d=%d) is not a neighbor of %d", b, u, d, u);
            pP[d][u] = p; pB[d][u] = b;
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the routing tables");

    // ---- baseline B (reference SP, backup = primary) ----
    ll B = worstStranded(refP, refP);
    if (B <= 0) B = 1;

    // ---- participant F ----
    ll F = worstStranded(pP, pB);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
