// TIER: greedy
// The obvious first approach: collapse the seven seasons into ONE scalar per
// edge (its total weight summed over all seasons) and solve a classic weighted
// graph-coloring-style problem -- process towers in decreasing total weighted
// degree, and greedily give each tower the channel that minimizes the INCREMENTAL
// TOTAL weighted conflict with already-placed neighbors. This is blind to which
// season a unit of weight lands in, so it treats a "2*WH-total" hot edge (whose
// damage always lands on the same two seasons) as interchangeable with an
// equally-costly mild edge (whose damage lands on just one of five other
// seasons) -- exactly the average-weighting trap the sum-of-worst-two objective
// punishes.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, C, M;
    scanf("%d %d %d", &N, &C, &M);
    vector<int> eu(M), ev(M);
    vector<array<int,8>> ew(M);
    vector<ll> totalW(M);
    vector<vector<int>> adj(N + 1);
    for (int i = 0; i < M; i++) {
        int u, v; scanf("%d %d", &u, &v);
        eu[i] = u; ev[i] = v;
        array<int,8> w{};
        ll tw = 0;
        for (int s = 1; s <= 7; s++) { scanf("%d", &w[s]); tw += w[s]; }
        ew[i] = w; totalW[i] = tw;
        adj[u].push_back(i);
        adj[v].push_back(i);
    }

    vector<ll> deg(N + 1, 0);
    for (int i = 0; i < M; i++) { deg[eu[i]] += totalW[i]; deg[ev[i]] += totalW[i]; }

    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (deg[a] != deg[b]) return deg[a] > deg[b];
        return a < b;
    });

    vector<int> ch(N + 1, 0);
    vector<char> placed(N + 1, 0);
    for (int t : order) {
        vector<ll> cost(C + 1, 0);
        for (int eid : adj[t]) {
            int other = (eu[eid] == t) ? ev[eid] : eu[eid];
            if (!placed[other]) continue;
            cost[ch[other]] += totalW[eid];
        }
        int best = 1;
        for (int c = 2; c <= C; c++) if (cost[c] < cost[best]) best = c;
        ch[t] = best;
        placed[t] = 1;
    }

    for (int i = 1; i <= N; i++) printf("%d%c", ch[i], i == N ? '\n' : ' ');
    return 0;
}
