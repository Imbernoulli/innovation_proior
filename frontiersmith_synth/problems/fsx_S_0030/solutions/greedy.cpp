// TIER: greedy
// Value-first greedy: consider all (herd, route) pairs in decreasing conservation value,
// assign a herd to a route the first time both the herd is free and the route has room.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int H, R;
    scanf("%d %d", &H, &R);
    vector<ll> rem(R + 1);
    for (int j = 1; j <= R; j++) scanf("%lld", &rem[j]);
    vector<vector<int>> V(H + 1, vector<int>(R + 1)), D(H + 1, vector<int>(R + 1));
    for (int i = 1; i <= H; i++)
        for (int j = 1; j <= R; j++) scanf("%d %d", &V[i][j], &D[i][j]);

    // pairs sorted by value descending
    vector<array<int,3>> pairs; // v, i, j
    pairs.reserve((size_t)H * R);
    for (int i = 1; i <= H; i++)
        for (int j = 1; j <= R; j++)
            pairs.push_back({V[i][j], i, j});
    sort(pairs.begin(), pairs.end(), [](const array<int,3>& a, const array<int,3>& b){
        return a[0] > b[0];
    });

    vector<int> ans(H + 1, 0);
    for (auto& p : pairs) {
        int i = p[1], j = p[2];
        if (ans[i] != 0) continue;
        if ((ll)D[i][j] <= rem[j]) { ans[i] = j; rem[j] -= D[i][j]; }
    }
    for (int i = 1; i <= H; i++) printf("%d\n", ans[i]);
    return 0;
}
