// TIER: greedy
// Nearest-neighbour Hamiltonian path: always hop to the closest unvisited station.
// Degree <= 2 everywhere (c_i >= 2), geometry-aware, beats the input-order chain.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n;
vector<ll> X, Y;

static inline ll d2(int a, int b) {
    ll dx = X[a] - X[b], dy = Y[a] - Y[b];
    return dx * dx + dy * dy;
}

int main() {
    if (scanf("%d", &n) != 1) return 0;
    X.assign(n + 1, 0); Y.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        ll c; scanf("%lld %lld %lld", &X[i], &Y[i], &c);
    }
    if (n == 1) { printf("0\n"); return 0; }

    vector<char> used(n + 1, 0);
    vector<int> order;
    order.reserve(n);
    int cur = 1;
    used[1] = 1; order.push_back(1);
    for (int step = 1; step < n; step++) {
        int best = -1; ll bd = -1;
        for (int j = 1; j <= n; j++) {
            if (used[j]) continue;
            ll dd = d2(cur, j);
            if (best == -1 || dd < bd) { bd = dd; best = j; }
        }
        used[best] = 1; order.push_back(best); cur = best;
    }

    printf("%d\n", n - 1);
    for (int i = 1; i < n; i++) printf("%d %d\n", order[i - 1], order[i]);
    return 0;
}
