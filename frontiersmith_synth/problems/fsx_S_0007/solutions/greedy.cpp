// TIER: greedy
// Nearest-neighbour path: start at station 0, repeatedly hop to the closest unvisited
// station, producing a short single chain (max degree 2 <= cap). Much shorter than the
// shuffled input-order chain baseline, but still a path (no branching), so it loses to a
// real degree-constrained tree.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    vector<long long> X(N), Y(N);
    vector<int> cap(N);
    for (int i = 0; i < N; i++) scanf("%lld %lld %d", &X[i], &Y[i], &cap[i]);

    auto d2 = [&](int a, int b) {
        long long dx = X[a] - X[b], dy = Y[a] - Y[b];
        return dx * dx + dy * dy;
    };

    vector<char> vis(N, 0);
    vector<int> order;
    order.reserve(N);
    int cur = 0;
    vis[0] = 1;
    order.push_back(0);
    for (int step = 1; step < N; step++) {
        int best = -1;
        long long bd = LLONG_MAX;
        for (int j = 0; j < N; j++) {
            if (vis[j]) continue;
            long long dd = d2(cur, j);
            if (dd < bd) { bd = dd; best = j; }
        }
        vis[best] = 1;
        order.push_back(best);
        cur = best;
    }

    printf("%d\n", N - 1);
    for (int i = 0; i + 1 < N; i++)
        printf("%d %d\n", order[i] + 1, order[i + 1] + 1);
    return 0;
}
