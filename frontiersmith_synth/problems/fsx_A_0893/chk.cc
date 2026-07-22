#include "testlib.h"
#include <vector>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int lo = inf.readInt();
    int hi = inf.readInt();

    vector<int> eu(m), ev(m), ew(m);
    vector<long long> deg(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u = inf.readInt();
        int v = inf.readInt();
        int w = inf.readInt();
        eu[i] = u; ev[i] = v; ew[i] = w;
        deg[u] += w; deg[v] += w;
    }

    // participant output: exactly n tokens, each 0 or 1
    vector<int> lab(n + 1, -1);
    for (int i = 1; i <= n; i++) {
        lab[i] = ouf.readInt(0, 1, "label");
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after n labels");

    long long count0 = 0;
    for (int i = 1; i <= n; i++) if (lab[i] == 0) count0++;
    if (count0 < lo || count0 > hi) {
        quitf(_wa, "balance violated: c0=%lld not in [%d,%d]", count0, lo, hi);
    }

    auto ncut = [&](const vector<int>& lb) -> double {
        long long cut = 0, volA = 0, volB = 0;
        for (int i = 1; i <= n; i++) {
            if (lb[i] == 0) volA += deg[i]; else volB += deg[i];
        }
        for (int i = 0; i < m; i++) {
            if (lb[eu[i]] != lb[ev[i]]) cut += ew[i];
        }
        // Every vertex in a generated instance has degree >= 1 (the generator
        // seeds each community with a spanning tree before adding density),
        // and the balance window [lo,hi] keeps both sides non-empty, so
        // volA/volB are always > 0 for any feasible labeling in practice.
        // Still, never let a degenerate zero-volume side yield an
        // artificially LOW F (that would be a free-score loophole): penalize
        // it heavily instead of silently substituting a small value.
        const double DEGENERATE_PENALTY = 1e12;
        double a = volA > 0 ? (double)cut / (double)volA : DEGENERATE_PENALTY;
        double b = volB > 0 ? (double)cut / (double)volB : DEGENERATE_PENALTY;
        return a + b;
    };

    double F = ncut(lab);

    // internal baseline: structure-blind index bisection (fixed, ignores edges entirely)
    vector<int> base(n + 1, 0);
    int half = n / 2;
    for (int i = 1; i <= n; i++) base[i] = (i <= half) ? 0 : 1;
    double B = ncut(base);
    if (B <= 0.0) B = 1e-9;

    // Scale factor tuned so the ceiling stays open above the reference "strong"
    // solution (which typically reaches normalized cut ~5-13x below B): a naive
    // 10x-to-cap convention would saturate at 1.0 almost everywhere on this
    // objective, so use 6x-to-cap instead (baseline still lands at ratio 0.06).
    double Fg = F > 1e-9 ? F : 1e-9;
    double sc = 60.0 * B / Fg;
    if (sc > 1000.0) sc = 1000.0;
    double ratio = sc / 1000.0;

    quitp(ratio, "F=%.6f B=%.6f Ratio: %.6f", F, B, ratio);
    return 0;
}
