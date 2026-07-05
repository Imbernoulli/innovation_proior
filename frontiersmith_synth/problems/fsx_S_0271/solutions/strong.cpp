// TIER: strong
// Nearest-neighbour tour + 2-opt local search. Start from a greedy survivable cycle and
// repeatedly apply length-reducing segment reversals (measured with the same rounded
// Euclidean length the checker uses). The result is still a single Hamiltonian cycle:
// 2-edge-connected, degree 2 everywhere, so it stays feasible while getting much shorter.
#include <bits/stdc++.h>
using namespace std;
static vector<long long> X, Y;
static inline long long len(int a, int b) {
    double dx = (double)(X[a] - X[b]);
    double dy = (double)(Y[a] - Y[b]);
    return (long long)llround(sqrt(dx * dx + dy * dy));
}
int main() {
    int N; if (scanf("%d", &N) != 1) return 0;
    X.resize(N); Y.resize(N);
    for (int i = 0; i < N; i++) { long long c; scanf("%lld %lld %lld", &X[i], &Y[i], &c); }

    // nearest-neighbour seed
    vector<int> t; t.reserve(N);
    vector<char> vis(N, 0);
    auto d2 = [&](int a, int b) { long long dx = X[a] - X[b], dy = Y[a] - Y[b]; return dx*dx + dy*dy; };
    int cur = 0; vis[0] = 1; t.push_back(0);
    for (int k = 1; k < N; k++) {
        int best = -1; long long bd = LLONG_MAX;
        for (int j = 0; j < N; j++) if (!vis[j]) { long long dd = d2(cur, j); if (dd < bd) { bd = dd; best = j; } }
        vis[best] = 1; t.push_back(best); cur = best;
    }

    // 2-opt: reverse t[i+1..j] if it shortens the cycle. Bounded passes to stay in time.
    int maxPasses = (N <= 120) ? 400 : (N <= 300 ? 120 : 40);
    bool improved = true;
    for (int pass = 0; pass < maxPasses && improved; pass++) {
        improved = false;
        for (int i = 0; i < N - 1; i++) {
            int a = t[i], b = t[i + 1];
            long long dab = len(a, b);
            for (int j = i + 2; j < N; j++) {
                int c = t[j], dnode = t[(j + 1) % N];
                if (i == 0 && (j + 1) % N == 0) continue; // adjacent wrap, skip
                long long before = dab + len(c, dnode);
                long long after = len(a, c) + len(b, dnode);
                if (after < before) {
                    reverse(t.begin() + i + 1, t.begin() + j + 1);
                    b = t[i + 1];
                    dab = len(a, b);
                    improved = true;
                }
            }
        }
    }

    printf("%d\n", N);
    for (int i = 0; i < N; i++)
        printf("%d %d\n", t[i] + 1, t[(i + 1) % N] + 1);
    return 0;
}
