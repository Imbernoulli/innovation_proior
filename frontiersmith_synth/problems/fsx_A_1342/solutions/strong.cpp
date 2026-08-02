// TIER: strong
// The insight: don't ask "what's the cheapest couple for THIS line", ask
// "which couple certifies the most total weight, across every line it sits
// inside, right now". A couple that lies inside several winning lines at once
// is a single certificate for all of them (an explicit involution shared by
// multiple lines) -- strictly better than spending cheaply on a couple that
// only ever helps one line, even when that couple is locally the "efficient"
// choice. Repeatedly commit the highest total-coverage affordable couple.
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

    map<pair<int, int>, vector<int>> covers;   // couple -> lines it lies fully inside
    for (int i = 0; i < m; i++) {
        auto &c = cells[i];
        for (size_t a = 0; a < c.size(); a++)
            for (size_t b = a + 1; b < c.size(); b++) {
                int x = min(c[a], c[b]), y = max(c[a], c[b]);
                covers[{x, y}].push_back(i);
            }
    }

    vector<char> used(n, 0), lineCovered(m, 0);
    ll budgetLeft = K;
    vector<pair<int, int>> pairs;

    while (true) {
        ll bestGain = 0, bestCost = LLONG_MAX;
        int bestA = -1, bestB = -1;
        const vector<int> *bestLines = nullptr;
        for (auto &kv : covers) {
            int a = kv.first.first, b = kv.first.second;
            if (used[a] || used[b]) continue;
            ll cost = 1 + llabs(zone[a] - zone[b]);
            if (cost > budgetLeft) continue;
            ll gain = 0;
            for (int li : kv.second) if (!lineCovered[li]) gain += w[li];
            if (gain <= 0) continue;
            if (gain > bestGain || (gain == bestGain && cost < bestCost)) {
                bestGain = gain; bestCost = cost; bestA = a; bestB = b; bestLines = &kv.second;
            }
        }
        if (bestA == -1) break;
        used[bestA] = used[bestB] = 1;
        budgetLeft -= bestCost;
        pairs.push_back({bestA, bestB});
        for (int li : *bestLines) lineCovered[li] = 1;
    }

    printf("%d\n", (int)pairs.size());
    for (auto &pr : pairs) printf("%d %d\n", pr.first, pr.second);
    return 0;
}
