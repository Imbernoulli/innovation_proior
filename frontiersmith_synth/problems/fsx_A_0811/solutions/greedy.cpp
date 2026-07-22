// TIER: greedy
// The "obvious" first attempt: sort pipes by their OWN demand (biggest pipe
// first) and spend the valve budget fencing the junction that pipe hangs off
// of. This protects a few large single numbers but never looks at total
// segment demand or the double-burst list at all.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, M, V, C;
    scanf("%d %d %d %d", &N, &M, &V, &C);
    vector<char> isCand(M + 1, 0);
    for (int i = 0; i < C; i++) { int c; scanf("%d", &c); isCand[c] = 1; }

    vector<int> eu(M + 1), ev(M + 1);
    vector<ll> ed(M + 1);
    for (int k = 1; k <= M; k++) scanf("%d %d %lld", &eu[k], &ev[k], &ed[k]);

    int L; scanf("%d", &L);
    for (int t = 0; t < L; t++) { int a, b; scanf("%d %d", &a, &b); }

    // incident candidate edges per node
    vector<vector<int>> incidentCand(N + 1);
    for (int e = 1; e <= M; e++) if (isCand[e]) {
        incidentCand[eu[e]].push_back(e);
        incidentCand[ev[e]].push_back(e);
    }

    vector<int> order(M);
    iota(order.begin(), order.end(), 1);
    sort(order.begin(), order.end(), [&](int a, int b) { return ed[a] > ed[b]; });

    vector<char> chosen(M + 1, 0), visitedNode(N + 1, 0);
    int budget = V;
    for (int e : order) {
        if (budget <= 0) break;
        for (int node : {eu[e], ev[e]}) {
            if (visitedNode[node]) continue;
            visitedNode[node] = 1;
            for (int c : incidentCand[node]) {
                if (budget <= 0) break;
                if (!chosen[c]) { chosen[c] = 1; budget--; }
            }
        }
    }

    vector<int> result;
    for (int e = 1; e <= M; e++) if (chosen[e]) result.push_back(e);
    printf("%d\n", (int)result.size());
    for (size_t i = 0; i < result.size(); i++) printf("%d%c", result[i], i + 1 == result.size() ? '\n' : ' ');
    if (result.empty()) printf("\n");
    return 0;
}
