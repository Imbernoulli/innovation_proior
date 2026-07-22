#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// braess-toll-surgeon checker.
// Reads the network + a participant toll vector, runs the deterministic
// best-response settling with those tolls -> total travel time F, and with
// zero tolls -> baseline B, then emits Ratio = min(1000,100*B/max(1,F))/1000.

struct E { int u, v; long long a; int k; long long b; };

static const long long CAP = (long long)4e15;

static long long latency(const E& e, long long x){
    long long xp = 1;
    for (int i = 0; i < e.k; i++){ xp *= x; if (xp > CAP){ xp = CAP; break; } }
    long long r = e.a * xp + e.b;
    if (r > CAP) r = CAP;
    return r;
}

// Deterministic best-response settling. Returns total travel time F.
// flowOut (if non-null) receives the final per-edge flow.
static long long settle(int N, const vector<E>& edg,
                        const vector<vector<int>>& adj,
                        const vector<int>& cs, const vector<int>& ct,
                        int R, const vector<long long>& toll,
                        vector<int>* flowOut){
    int M = (int)edg.size();
    int D = (int)cs.size();
    vector<int> flow(M, 0);
    vector<vector<int>> path(D);

    auto edgeCost = [&](int e) -> long long {
        long long c = latency(edg[e], (long long)flow[e] + 1) + toll[e];
        if (c > CAP) c = CAP;
        return c;
    };

    // Dijkstra from s; returns edge list of the unique cheapest route to t.
    auto shortest = [&](int s, int t) -> vector<int> {
        vector<long long> dist(N, LLONG_MAX);
        vector<int> pe(N, -1);
        priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<pair<long long,int>>> pq;
        dist[s] = 0; pq.push({0, s});
        while (!pq.empty()){
            auto top = pq.top(); pq.pop();
            long long d = top.first; int u = top.second;
            if (d > dist[u]) continue;
            for (int e : adj[u]){
                int v = edg[e].v;
                long long nd = d + edgeCost(e);
                if (nd < dist[v]){ dist[v] = nd; pe[v] = e; pq.push({nd, v}); }
            }
        }
        vector<int> p;
        int cur = t;
        while (cur != s){
            int e = pe[cur];
            if (e < 0){ p.clear(); return p; } // unreachable (should not happen)
            p.push_back(e);
            cur = edg[e].u;
        }
        reverse(p.begin(), p.end());
        return p;
    };
    auto assign = [&](int i){
        vector<int> p = shortest(cs[i], ct[i]);
        path[i] = p;
        for (int e : p) flow[e]++;
    };
    auto rem = [&](int i){
        for (int e : path[i]) flow[e]--;
        path[i].clear();
    };

    for (int i = 0; i < D; i++) assign(i);
    for (int r = 0; r < R; r++)
        for (int i = 0; i < D; i++){ rem(i); assign(i); }

    long long F = 0;
    for (int e = 0; e < M; e++)
        F += (long long)flow[e] * latency(edg[e], (long long)flow[e]);
    if (flowOut) *flowOut = flow;
    return F;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int D = inf.readInt();
    long long T = inf.readLong();
    int R = inf.readInt();

    vector<E> edg(M);
    vector<vector<int>> adj(N);
    for (int e = 0; e < M; e++){
        int u = inf.readInt() - 1;
        int v = inf.readInt() - 1;
        long long a = inf.readLong();
        int k = inf.readInt();
        long long b = inf.readLong();
        edg[e] = {u, v, a, k, b};
        adj[u].push_back(e);
    }
    vector<int> cs(D), ct(D);
    for (int i = 0; i < D; i++){ cs[i] = inf.readInt() - 1; ct[i] = inf.readInt() - 1; }

    // ---- read participant tolls (strict validation) ----
    vector<long long> toll(M);
    long long tsum = 0;
    for (int e = 0; e < M; e++){
        toll[e] = ouf.readLong(0, T, format("toll[%d]", e + 1));
        tsum += toll[e];
        if (tsum > T) quitf(_wa, "total toll mass %lld exceeds budget %lld", tsum, T);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d tolls", M);

    // ---- objective F (with tolls) and baseline B (zero tolls) ----
    long long F = settle(N, edg, adj, cs, ct, R, toll, nullptr);
    vector<long long> zero(M, 0);
    long long B = settle(N, edg, adj, cs, ct, R, zero, nullptr);
    if (B <= 0) B = 1;

    // Score rewards the travel-time improvement ratio B/F super-linearly so that
    // the reference ceiling (a Braess price-of-anarchy < 2) leaves headroom:
    //   ratio = min(1, 0.1 * (B/F)^1.4)
    // do-nothing (F=B) scores 0.1; the cap (ratio=1) needs B/F ~= 5.2, which is
    // unreachable here, so the ceiling stays open.
    double improve = (double)B / (double)max(1LL, F);
    double ratio = 0.1 * pow(improve, 1.4);
    if (ratio > 1.0) ratio = 1.0;
    if (ratio < 0.0) ratio = 0.0;
    quitp(ratio, "OK F=%lld B=%lld Ratio: %.6f", F, B, ratio);
    return 0;
}
