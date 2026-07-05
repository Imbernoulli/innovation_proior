// TIER: greedy
// One-pass shortest-route heuristic: for each demand in turn, find a minimum-hop origin->
// destination path over segments with spare capacity, and push as many passengers as the
// tightest residual capacity on that path allows -- all down a SINGLE route.  Feasible and
// beats the do-nothing baseline, but it concentrates load onto shared routes and so pays the
// full convex a_e*x^2 crowding: exactly the trap the problem warns about.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, D; long long P;
    if (scanf("%d %d %d %lld", &N, &M, &D, &P) != 4) return 0;

    vector<int> eu(M), ev(M), ecap(M), ea(M), eres(M);
    vector<vector<int>> adj(N + 1);  // store edge ids
    for (int e = 0; e < M; e++) {
        scanf("%d %d %d %d", &eu[e], &ev[e], &ecap[e], &ea[e]);
        eres[e] = ecap[e];
        adj[eu[e]].push_back(e);
        adj[ev[e]].push_back(e);
    }
    vector<int> sd(D), td(D), vol(D);
    for (int d = 0; d < D; d++) scanf("%d %d %d", &sd[d], &td[d], &vol[d]);

    // collect output lines
    string out;
    int K = 0;
    vector<int> par(N + 1), parEdge(N + 1);

    for (int d = 0; d < D; d++) {
        int s = sd[d], t = td[d];
        // BFS over edges with residual > 0
        fill(par.begin(), par.end(), -1);
        fill(parEdge.begin(), parEdge.end(), -1);
        par[s] = s;
        queue<int> q; q.push(s);
        while (!q.empty()) {
            int x = q.front(); q.pop();
            if (x == t) break;
            for (int e : adj[x]) {
                if (eres[e] <= 0) continue;
                int y = (eu[e] == x) ? ev[e] : eu[e];
                if (par[y] == -1) { par[y] = x; parEdge[y] = e; q.push(y); }
            }
        }
        if (par[t] == -1) continue;  // unreachable with spare capacity -> leave unserved

        // reconstruct path and its bottleneck residual
        vector<int> path;
        long long bottleneck = LLONG_MAX;
        int cur = t;
        while (cur != s) {
            int e = parEdge[cur];
            bottleneck = min<long long>(bottleneck, eres[e]);
            path.push_back(cur);
            cur = par[cur];
        }
        path.push_back(s);
        reverse(path.begin(), path.end());

        long long f = min<long long>(vol[d], bottleneck);
        if (f <= 0) continue;
        // apply residual along path edges
        for (int j = 1; j < (int)path.size(); j++) {
            int x = path[j];
            int e = parEdge[x];
            eres[e] -= (int)f;
        }

        // emit line: d L w... f
        out += to_string(d + 1);
        out += ' ';
        out += to_string((int)path.size());
        for (int v : path) { out += ' '; out += to_string(v); }
        out += ' ';
        out += to_string(f);
        out += '\n';
        K++;
    }

    printf("%d\n", K);
    fputs(out.c_str(), stdout);
    return 0;
}
