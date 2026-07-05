// TIER: greedy
// Longest-processing-time load balancer that IGNORES rig geometry: assign each act
// (in decreasing runtime) to the currently least-loaded stage. Balances makespan but
// scrambles clusters -> large setup (this is the trap the statement warns about).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int N, M, d;
    if (scanf("%d %d %d", &N, &M, &d) != 3) return 0;
    vector<ll> p(N + 1);
    for (int j = 1; j <= N; j++) { scanf("%lld", &p[j]); for (int k = 0; k < d; k++) { int x; scanf("%d", &x); } }
    vector<int> idx(N);
    for (int j = 0; j < N; j++) idx[j] = j + 1;
    sort(idx.begin(), idx.end(), [&](int a, int b) { return p[a] > p[b]; });
    vector<ll> load(M, 0);
    vector<vector<int>> st(M);
    for (int a : idx) {
        int best = 0;
        for (int i = 1; i < M; i++) if (load[i] < load[best]) best = i;
        st[best].push_back(a);
        load[best] += p[a];
    }
    for (int i = 0; i < M; i++) {
        printf("%d", (int)st[i].size());
        for (int a : st[i]) printf(" %d", a);
        printf("\n");
    }
    return 0;
}
