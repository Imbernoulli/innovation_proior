// TIER: greedy
// Nearest-neighbour path over the K ground stations only (relay masts ignored).
// Start at station 1, repeatedly hop to the closest unvisited station -> a short degree-2
// chain. Feasible since every cap>=2. Beats the input-order chain baseline.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, K;
    if (scanf("%d %d", &N, &K) != 2) return 0;
    vector<long long> X(N), Y(N);
    vector<int> cap(N);
    for (int i = 0; i < N; i++)
        scanf("%lld %lld %d", &X[i], &Y[i], &cap[i]);

    auto d2 = [&](int a, int b) {
        long long dx = X[a] - X[b], dy = Y[a] - Y[b];
        return dx * dx + dy * dy;
    };

    vector<char> vis(K, 0);
    vector<int> order;
    int cur = 0;
    vis[0] = 1;
    order.push_back(0);
    for (int step = 1; step < K; step++) {
        int best = -1;
        long long bd = LLONG_MAX;
        for (int j = 0; j < K; j++) {
            if (vis[j]) continue;
            long long dd = d2(cur, j);
            if (dd < bd) { bd = dd; best = j; }
        }
        vis[best] = 1;
        order.push_back(best);
        cur = best;
    }

    printf("%d\n", K - 1);
    for (int i = 0; i + 1 < K; i++)
        printf("%d %d\n", order[i] + 1, order[i + 1] + 1);
    return 0;
}
