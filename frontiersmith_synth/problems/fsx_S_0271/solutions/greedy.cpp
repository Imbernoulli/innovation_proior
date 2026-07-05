// TIER: greedy
// Nearest-neighbour survivable tour: start at module 0 and repeatedly hop to the closest
// unvisited module, forming a short Hamiltonian cycle (degree 2 everywhere, no bridge).
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N; if (scanf("%d", &N) != 1) return 0;
    vector<long long> X(N), Y(N);
    for (int i = 0; i < N; i++) { long long c; scanf("%lld %lld %lld", &X[i], &Y[i], &c); }
    auto d2 = [&](int a, int b) {
        long long dx = X[a] - X[b], dy = Y[a] - Y[b];
        return dx * dx + dy * dy;
    };
    vector<int> tour; tour.reserve(N);
    vector<char> vis(N, 0);
    int cur = 0; vis[0] = 1; tour.push_back(0);
    for (int k = 1; k < N; k++) {
        int best = -1; long long bd = LLONG_MAX;
        for (int j = 0; j < N; j++) if (!vis[j]) {
            long long dd = d2(cur, j);
            if (dd < bd) { bd = dd; best = j; }
        }
        vis[best] = 1; tour.push_back(best); cur = best;
    }
    printf("%d\n", N);
    for (int i = 0; i < N; i++)
        printf("%d %d\n", tour[i] + 1, tour[(i + 1) % N] + 1);
    return 0;
}
