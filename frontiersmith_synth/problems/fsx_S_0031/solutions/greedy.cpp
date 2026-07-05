// TIER: greedy
// Degree-constrained nearest-neighbour Hamiltonian path.
// Every internal node has degree 2, endpoints degree 1 -> always <= b_i (>=2). Always feasible.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    vector<ll> X(n), Y(n);
    vector<int> B(n);
    for (int i = 0; i < n; i++) { scanf("%lld %lld %d", &X[i], &Y[i], &B[i]); }

    auto d2 = [&](int a, int b) -> ll {
        ll dx = X[a]-X[b], dy = Y[a]-Y[b];
        return dx*dx + dy*dy;
    };

    vector<char> used(n, 0);
    vector<int> path;
    path.reserve(n);
    int cur = 0;
    used[0] = 1;
    path.push_back(0);
    for (int step = 1; step < n; step++) {
        int best = -1; ll bestd = LLONG_MAX;
        for (int j = 0; j < n; j++) {
            if (used[j]) continue;
            ll dd = d2(cur, j);
            if (dd < bestd) { bestd = dd; best = j; }
        }
        used[best] = 1;
        path.push_back(best);
        cur = best;
    }

    printf("%d\n", n - 1);
    for (int i = 1; i < n; i++)
        printf("%d %d\n", path[i-1] + 1, path[i] + 1);
    return 0;
}
