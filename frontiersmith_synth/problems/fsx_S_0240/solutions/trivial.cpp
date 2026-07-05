// TIER: trivial
// Earliest-deadline-first pack, identical to the checker's baseline B.
// Sort tasks by (deadline, release, index); place each at the earliest in-window
// step with spare throughput. Output assignment in original task order -> F == B -> 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int T, N, M;
    if (scanf("%d %d %d", &T, &N, &M) != 3) return 0;
    long long G; scanf("%lld", &G);
    vector<long long> f(T + 1), s(T + 1); vector<int> cap(T + 1);
    for (int t = 1; t <= T; t++) scanf("%lld %lld %d", &f[t], &s[t], &cap[t]);
    vector<int> r(N), dl(N);
    for (int i = 0; i < N; i++) scanf("%d %d", &r[i], &dl[i]);

    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (dl[a] != dl[b]) return dl[a] < dl[b];
        if (r[a] != r[b]) return r[a] < r[b];
        return a < b;
    });
    vector<int> y(T + 1, 0), assign(N, -1);
    for (int idx : order) {
        for (int t = r[idx]; t <= dl[idx]; t++) {
            if (y[t] < M) { assign[idx] = t; y[t]++; break; }
        }
        if (assign[idx] < 0) assign[idx] = r[idx]; // shouldn't happen (guaranteed feasible)
    }
    for (int i = 0; i < N; i++) printf("%d\n", assign[i]);
    return 0;
}
