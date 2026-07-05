// TIER: greedy
// Density greedy: sort all (block,pump) pairs by yield-per-litre, assign greedily.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int P, B;
    if (scanf("%d %d", &P, &B) != 2) return 0;
    vector<long long> rem(P + 1);
    for (int p = 1; p <= P; p++) scanf("%lld", &rem[p]);
    vector<vector<int>> w(P + 1, vector<int>(B + 1)), v(P + 1, vector<int>(B + 1));
    for (int j = 1; j <= B; j++)
        for (int p = 1; p <= P; p++) scanf("%d %d", &w[p][j], &v[p][j]);

    struct Pair { double dens; int p, j; };
    vector<Pair> pairs;
    pairs.reserve((size_t)P * B);
    for (int j = 1; j <= B; j++)
        for (int p = 1; p <= P; p++)
            pairs.push_back({ (double)v[p][j] / (double)w[p][j], p, j });
    sort(pairs.begin(), pairs.end(),
         [](const Pair& x, const Pair& y) { return x.dens > y.dens; });

    vector<int> a(B + 1, 0);
    for (auto& pr : pairs) {
        if (a[pr.j] != 0) continue;
        if (rem[pr.p] >= w[pr.p][pr.j]) { rem[pr.p] -= w[pr.p][pr.j]; a[pr.j] = pr.p; }
    }

    for (int j = 1; j <= B; j++) printf("%d%c", a[j], j == B ? '\n' : ' ');
    return 0;
}
