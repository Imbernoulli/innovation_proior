// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// The obvious single-pass recipe: walk the candidate edges in the order they
// appear in the input and greedily claim as much coupling as each one still
// allows (its own cap, the remaining budget, and both endpoints' remaining
// coupling-degree headroom). No edge is ever revisited. This "fill what you
// see, in the order you see it" allocator has no notion of which edges are
// structurally load-bearing -- it just spends locally-available capacity
// wherever it happens to land first.

static const double DEGCAP = 1.0;

int main() {
    int N, M; scanf("%d %d", &N, &M);
    double R, C; int T, W;
    scanf("%lf %lf %d %d", &R, &C, &T, &W);
    for (int i = 0; i < N; i++) { double x; scanf("%lf", &x); }
    vector<int> u(M), v(M); vector<double> cap(M);
    for (int e = 0; e < M; e++) scanf("%d %d %lf", &u[e], &v[e], &cap[e]);

    vector<double> deg(N + 1, 0.0), c(M, 0.0);
    double rem = C;
    for (int e = 0; e < M; e++) {
        double avail = min({cap[e], rem, DEGCAP - deg[u[e]], DEGCAP - deg[v[e]]});
        if (avail > 1e-12) {
            c[e] = avail;
            deg[u[e]] += avail; deg[v[e]] += avail;
            rem -= avail;
        }
    }
    for (int e = 0; e < M; e++) printf("%.9f%c", c[e], e + 1 == M ? '\n' : ' ');
    return 0;
}
