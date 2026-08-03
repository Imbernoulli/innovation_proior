// TIER: greedy
// Nearest-neighbour pole path: start at the first light pole and repeatedly hop to the
// closest unvisited pole, forming a short degree-2 chain. Ignores junction cabinets.
// Beats the input-order chain baseline on most tests.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    vector<long long> X(N), Y(N);
    vector<int> poles;
    for (int i = 0; i < N; i++) {
        long long x, y, t, cap;
        scanf("%lld %lld %lld %lld", &x, &y, &t, &cap);
        X[i] = x; Y[i] = y;
        if (t) poles.push_back(i);
    }
    int M = (int)poles.size();
    if (M <= 1) { printf("0\n"); return 0; }

    auto d2 = [&](int a, int b) {
        long long dx = X[a] - X[b], dy = Y[a] - Y[b];
        return dx * dx + dy * dy;
    };

    vector<char> vis(M, 0);
    vector<int> order;
    int cur = 0;
    vis[0] = 1;
    order.push_back(0);
    for (int step = 1; step < M; step++) {
        int best = -1; long long bd = LLONG_MAX;
        for (int j = 0; j < M; j++) {
            if (vis[j]) continue;
            long long dd = d2(poles[cur], poles[j]);
            if (dd < bd) { bd = dd; best = j; }
        }
        vis[best] = 1;
        order.push_back(best);
        cur = best;
    }

    printf("%d\n", M - 1);
    for (int j = 0; j + 1 < M; j++)
        printf("%d %d\n", poles[order[j]] + 1, poles[order[j + 1]] + 1);
    return 0;
}
