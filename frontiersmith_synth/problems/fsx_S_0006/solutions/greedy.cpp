// TIER: greedy
// Value-density greedy: rank (school,pool) pairs by value/volume, one pass.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, P;
    scanf("%d %d", &N, &P);
    vector<long long> C(P);
    for (int j = 0; j < P; j++) scanf("%lld", &C[j]);
    vector<vector<long long>> v(N, vector<long long>(P)), w(N, vector<long long>(P));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++) scanf("%lld %lld", &v[i][j], &w[i][j]);

    struct Cand { double dens; int i, j; };
    vector<Cand> cand;
    cand.reserve((size_t)N * P);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < P; j++)
            cand.push_back({(double)v[i][j] / (double)w[i][j], i, j});
    sort(cand.begin(), cand.end(), [](const Cand& x, const Cand& y) {
        return x.dens > y.dens;
    });

    vector<long long> rem = C;
    vector<int> a(N, 0);
    for (auto& c : cand) {
        if (a[c.i] != 0) continue;
        if (w[c.i][c.j] <= rem[c.j]) { rem[c.j] -= w[c.i][c.j]; a[c.i] = c.j + 1; }
    }

    for (int i = 0; i < N; i++) printf("%d%c", a[i], i + 1 == N ? '\n' : ' ');
    return 0;
}
