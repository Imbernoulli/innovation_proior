// TIER: greedy
// Nearest-neighbour lift chain: start at station 1 and repeatedly hop to the cheapest
// unvisited station under the elevation-weighted cost. Produces a degree-2 path, always
// feasible (caps >= 2), and typically much shorter than the arbitrary input-order chain.
#include <bits/stdc++.h>
using namespace std;

static vector<long long> X, Y, H;
static inline long long len(int a, int b) {
    double dx = (double)(X[a] - X[b]);
    double dy = (double)(Y[a] - Y[b]);
    double dh = (double)(H[a] - H[b]);
    return (long long)llround(sqrt(dx * dx + dy * dy) + 3.0 * fabs(dh));
}

int main() {
    int N;
    if (scanf("%d", &N) != 1) return 0;
    X.resize(N); Y.resize(N); H.resize(N);
    vector<int> cap(N);
    for (int i = 0; i < N; i++) {
        scanf("%lld %lld %lld %d", &X[i], &Y[i], &H[i], &cap[i]);
    }
    if (N <= 1) { printf("0\n"); return 0; }

    vector<char> used(N, 0);
    vector<pair<int,int>> edges;
    int cur = 0;
    used[0] = 1;
    for (int step = 1; step < N; step++) {
        int best = -1;
        long long bd = LLONG_MAX;
        for (int j = 0; j < N; j++) {
            if (used[j]) continue;
            long long d = len(cur, j);
            if (d < bd) { bd = d; best = j; }
        }
        used[best] = 1;
        edges.push_back({cur + 1, best + 1});
        cur = best;
    }

    printf("%d\n", (int)edges.size());
    for (auto& e : edges) printf("%d %d\n", e.first, e.second);
    return 0;
}
