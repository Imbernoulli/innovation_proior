#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Maximum-Weight Independent Set on a GENERAL conflict graph, skinned as caldera
// monitoring-site selection.
//
// Structure ladder: testId 1 tiny (example scale) -> larger/denser by testId 10.
// Density is kept moderate-to-high so the maximum independent set stays small
// relative to n (a real search, not "take almost everything"), and site values
// mix uniform / mild-skew / clustered regimes so that trading a few high-value
// stations against many cheap-to-fit ones is genuinely non-trivial. The
// single-best-site baseline (checker's B = max value) is always weak: several
// non-conflicting sites easily beat one.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int ns[11] = {0, 12, 28, 48, 72, 100, 140, 180, 230, 300, 380};
    double ps[11] = {0, 0.35, 0.35, 0.35, 0.40, 0.40, 0.45, 0.45, 0.50, 0.50, 0.55};
    int n = ns[t];
    double p = ps[t];

    // value model varies for structural variety
    int model = t % 3; // 0 mild-skew, 1 uniform, 2 clustered (two tiers)
    vector<int> w(n + 1);
    for (int i = 1; i <= n; i++) {
        int val;
        if (model == 1) {
            val = rnd.next(1, 60);
        } else if (model == 2) {
            val = (rnd.next(0, 1) ? rnd.next(1, 15) : rnd.next(30, 60));
        } else { // mild skew: occasional premium site, but no single dominating whale
            val = rnd.next(1, 40);
            if (rnd.next(0.0, 1.0) < 0.15) val += rnd.next(40, 80);
        }
        w[i] = val;
    }

    // conflict edges via G(n,p); n <= 380 so the pair scan is cheap.
    vector<pair<int,int>> edges;
    for (int i = 1; i <= n; i++)
        for (int j = i + 1; j <= n; j++)
            if (rnd.next(0.0, 1.0) < p) edges.push_back({i, j});
    if (edges.empty() && n >= 2) edges.push_back({1, 2}); // guarantee >=1 conflict
    int m = (int)edges.size();

    printf("%d %d\n", n, m);

    string buf;
    buf.reserve((size_t)n * 4 + 16);
    char line[64];
    for (int i = 1; i <= n; i++) {
        int L = snprintf(line, sizeof line, "%s%d", (i > 1 ? " " : ""), w[i]);
        buf.append(line, L);
    }
    buf.push_back('\n');
    fputs(buf.c_str(), stdout);

    string eb;
    eb.reserve((size_t)m * 12);
    for (auto& pr : edges) {
        int L = snprintf(line, sizeof line, "%d %d\n", pr.first, pr.second);
        eb.append(line, L);
    }
    fputs(eb.c_str(), stdout);
    return 0;
}
