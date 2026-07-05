// TIER: greedy
// Nearest-neighbour path (Manhattan): start at inverter 0 and repeatedly hop to the
// closest unvisited inverter, forming a single degree-2 chain.  Shorter trench than the
// arbitrary input-order chain, but it ignores termination fees entirely and cannot branch.
// Always feasible (a path needs only degree 2, and every cap_i >= 2).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    vector<long long> x(N), y(N);
    for (int i = 0; i < N; i++) {
        int c, w;
        scanf("%lld %lld %d %d", &x[i], &y[i], &c, &w);
    }
    if (N == 1) { printf("0\n"); return 0; }

    vector<char> vis(N, 0);
    vector<int> order;
    order.reserve(N);
    int cur = 0;
    vis[0] = 1;
    order.push_back(0);
    for (int step = 1; step < N; step++) {
        long long best = LLONG_MAX;
        int bi = -1;
        for (int j = 0; j < N; j++) {
            if (vis[j]) continue;
            long long d = llabs(x[cur] - x[j]) + llabs(y[cur] - y[j]);
            if (d < best) { best = d; bi = j; }
        }
        vis[bi] = 1;
        order.push_back(bi);
        cur = bi;
    }

    printf("%d\n", N - 1);
    for (int i = 0; i + 1 < N; i++)
        printf("%d %d\n", order[i] + 1, order[i + 1] + 1);
    return 0;
}
