// TIER: greedy
// The obvious recipe: "an avalanche is coming from each release zone -- build a
// wall directly in its path, right in front of the village, as tall as the
// shared budget allows." Every distinct release column gets ONE wall barrier
// at the last row, height = floor(L / (#release columns)) each (capped at
// HMAX). This picks up real credit on the planted "stoppable" releases (low
// momentum) but is structurally overtopped on every "trap" release: a wall
// can only ever reach height HMAX, and those events were calibrated so the
// required height (>HMAX) is unreachable at ANY budget -- the wall never even
// dents them, no matter how the budget is split.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int H, W, K; ll L; int SCALE, DISS;
    if (scanf("%d %d %d %lld %d %d", &H, &W, &K, &L, &SCALE, &DISS) != 6) return 0;
    vector<int> w(W);
    for (int c = 0; c < W; c++) scanf("%d", &w[c]);
    vector<ll> c(K), m(K), p(K);
    for (int i = 0; i < K; i++) scanf("%lld %lld %lld", &c[i], &m[i], &p[i]);
    (void)m; (void)p; (void)SCALE; (void)DISS;

    // distinct release columns (K release events are already on distinct columns)
    int cnt = K;
    ll heightEach = min((ll)100, L / max(1, cnt)); // HMAX = 100
    if (heightEach < 1) heightEach = (L >= 1 && cnt > 0) ? 1 : 0;

    vector<tuple<int,int,int,int>> placed;
    ll spent = 0;
    for (int i = 0; i < K && heightEach >= 1; i++) {
        int r = H - 1;
        if (spent + heightEach > L) break;
        placed.push_back({r, (int)c[i], 0, (int)heightEach}); // orientation 0 = WALL
        spent += heightEach;
    }

    printf("%d\n", (int)placed.size());
    for (auto& t : placed) {
        printf("%d %d %d %d\n", get<0>(t), get<1>(t), get<2>(t), get<3>(t));
    }
    return 0;
}
