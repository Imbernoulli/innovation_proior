// TIER: greedy
// Nearest-neighbour path over the artifact chambers only: start at chamber 1 and repeatedly
// hop to the closest unwired chamber. Produces a degree-2 path (always within caps>=2),
// noticeably shorter than the arbitrary-order chain baseline. Ignores relay posts.
#include <bits/stdc++.h>
using namespace std;
int N, T;
vector<long long> X, Y;
static inline long long len2(int a, int b) {
    long long dx = X[a] - X[b], dy = Y[a] - Y[b];
    return dx * dx + dy * dy;
}
int main() {
    if (scanf("%d %d", &N, &T) != 2) return 0;
    X.resize(N); Y.resize(N);
    vector<int> cap(N);
    for (int i = 0; i < N; i++) { scanf("%lld %lld %d", &X[i], &Y[i], &cap[i]); }

    vector<char> vis(T, 0);
    int cur = 0; vis[0] = 1;
    vector<pair<int,int>> edges;
    for (int step = 1; step < T; step++) {
        int best = -1; long long bd = LLONG_MAX;
        for (int j = 0; j < T; j++) {
            if (vis[j]) continue;
            long long d = len2(cur, j);
            if (d < bd) { bd = d; best = j; }
        }
        vis[best] = 1;
        edges.push_back({cur + 1, best + 1});
        cur = best;
    }
    printf("%d\n", (int)edges.size());
    for (auto& e : edges) printf("%d %d\n", e.first, e.second);
    return 0;
}
