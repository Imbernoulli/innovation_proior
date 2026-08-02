// TIER: greedy
// The obvious first approach: rank every bid by price-per-item (density) and
// accept greedily, highest density first, whenever it still fits. This is a
// perfectly reasonable-looking heuristic for a bundle knapsack -- and it is
// exactly what falls into the hub/path traps: a spoiler bid on a single scarce
// item (or a lone high-priced middle edge) always has a strictly higher
// density than the multi-item bundle (or the two disjoint edges) that it
// blocks, so this greedy takes the spoiler in every gadget, every time.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int M, N;
    scanf("%d %d", &M, &N);
    vector<ll> cap(M + 1);
    for (int j = 1; j <= M; j++) scanf("%lld", &cap[j]);

    vector<int> k(N + 1);
    vector<ll> p(N + 1);
    vector<vector<int>> items(N + 1);
    for (int i = 1; i <= N; i++) {
        scanf("%d %lld", &k[i], &p[i]);
        items[i].resize(k[i]);
        for (int t = 0; t < k[i]; t++) scanf("%d", &items[i][t]);
    }

    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b) {
        // density = price / bundle size, compared via cross-multiplication
        // (exact, no floating point). Tie-break: higher price, then lower id.
        __int128 lhs = (__int128)p[a] * k[b];
        __int128 rhs = (__int128)p[b] * k[a];
        if (lhs != rhs) return lhs > rhs;
        if (p[a] != p[b]) return p[a] > p[b];
        return a < b;
    });

    vector<ll> rem = cap;
    vector<int> accepted;
    for (int i : order) {
        bool ok = true;
        for (int it : items[i]) if (rem[it] < 1) { ok = false; break; }
        if (ok) {
            for (int it : items[i]) rem[it]--;
            accepted.push_back(i);
        }
    }

    printf("%d\n", (int)accepted.size());
    for (size_t i = 0; i < accepted.size(); i++)
        printf("%d%c", accepted[i], i + 1 == accepted.size() ? '\n' : ' ');
    if (accepted.empty()) printf("\n");
    return 0;
}
