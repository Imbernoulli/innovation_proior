// TIER: greedy
// The obvious "recipe": sort lines by weight descending, and for each line
// still worth defending, pair the two CHEAPEST currently-free cells inside
// THAT line if the budget allows -- a purely local, per-line decision that
// never asks whether a couple also helps any other line.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m;
    ll K;
    scanf("%d %d %lld", &n, &m, &K);
    vector<int> zone(n);
    for (int i = 0; i < n; i++) scanf("%d", &zone[i]);
    vector<vector<int>> cells(m);
    vector<ll> w(m);
    for (int i = 0; i < m; i++) {
        int s;
        scanf("%d %lld", &s, &w[i]);
        cells[i].resize(s);
        for (int j = 0; j < s; j++) scanf("%d", &cells[i][j]);
    }
    // (Aggressor order follows in the input but this tier does not use it.)

    vector<int> idx(m);
    for (int i = 0; i < m; i++) idx[i] = i;
    stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return w[a] > w[b]; });

    vector<char> used(n, 0);
    vector<pair<int, int>> pairs;
    ll budgetLeft = K;

    for (int li : idx) {
        auto &c = cells[li];
        vector<int> free;
        for (int cell : c) if (!used[cell]) free.push_back(cell);
        if ((int)free.size() < 2) continue;
        int bestA = -1, bestB = -1;
        ll bestCost = LLONG_MAX;
        for (size_t a = 0; a < free.size(); a++)
            for (size_t b = a + 1; b < free.size(); b++) {
                ll cost = 1 + llabs(zone[free[a]] - zone[free[b]]);
                if (cost < bestCost) { bestCost = cost; bestA = free[a]; bestB = free[b]; }
            }
        if (bestA != -1 && bestCost <= budgetLeft) {
            used[bestA] = used[bestB] = 1;
            pairs.push_back({bestA, bestB});
            budgetLeft -= bestCost;
        }
    }

    printf("%d\n", (int)pairs.size());
    for (auto &pr : pairs) printf("%d %d\n", pr.first, pr.second);
    return 0;
}
