// TIER: greedy
// Value-first best-fit: process racks by descending value; place each on the feasible unit
// that minimizes its cooling load (leaving the most room).  Beats first-fit by prioritizing
// valuable racks and by choosing the cheapest unit rather than the lowest-index one.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<long long> v(N);
    for (int i = 0; i < N; i++) scanf("%lld", &v[i]);
    vector<long long> C(M);
    for (int j = 0; j < M; j++) scanf("%lld", &C[j]);
    vector<vector<long long>> d(N, vector<long long>(M));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) scanf("%lld", &d[i][j]);

    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int x, int y){
        if (v[x] != v[y]) return v[x] > v[y];
        return x < y;
    });

    vector<long long> rem(C);
    vector<int> a(N, 0);
    for (int idx = 0; idx < N; idx++) {
        int i = order[idx];
        int best = -1; long long bestLoad = LLONG_MAX;
        for (int j = 0; j < M; j++) {
            if (rem[j] >= d[i][j] && d[i][j] < bestLoad) { bestLoad = d[i][j]; best = j; }
        }
        if (best >= 0) { rem[best] -= d[i][best]; a[i] = best + 1; }
    }
    for (int i = 0; i < N; i++) printf("%d%c", a[i], i + 1 < N ? ' ' : '\n');
    return 0;
}
