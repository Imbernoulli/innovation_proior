// TIER: greedy
// The obvious first approach: nearest-neighbor by UNWEIGHTED Hamming distance
// (minimize the number of flipped lines, ignoring that lines have wildly
// different costs). Starts at experiment 0, deterministic tie-break by lowest
// index. On planted trap cases this re-crosses expensive boundaries over and
// over because a master-line flip plus one cheap flip "looks" as cheap as any
// other 2-flip move.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 1;
    vector<long long> C(M);
    for (int j = 0; j < M; j++) scanf("%lld", &C[j]);
    vector<unsigned long long> cfg(N, 0);
    char buf[128];
    for (int i = 0; i < N; i++) {
        scanf("%127s", buf);
        for (int j = 0; j < M; j++)
            if (buf[j] == '1') cfg[i] |= (1ULL << j);
    }
    vector<int> order;
    order.reserve(N);
    vector<char> used(N, 0);
    int cur = 0;
    for (int step = 0; step < N; step++) {
        order.push_back(cur);
        used[cur] = 1;
        if (step == N - 1) break;
        int best = -1, bd = INT_MAX;
        for (int j = 0; j < N; j++) {
            if (used[j]) continue;
            int d = __builtin_popcountll(cfg[cur] ^ cfg[j]);
            if (d < bd) { bd = d; best = j; }
        }
        cur = best;
    }
    for (int i = 0; i < N; i++)
        printf("%d%c", order[i], i + 1 == N ? '\n' : ' ');
    return 0;
}
