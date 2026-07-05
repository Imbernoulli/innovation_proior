#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder ----
    // testId 1 tiny (example-scale sanity), growing to a large, tight window by testId 10.
    int m = 2 + min(testId, 8);              // 3, 4, ..., 10
    int n = 6 + 6 * testId;                  // 12, 18, ..., 66
    if (n > 70) n = 70;

    int vmax = 60 + (testId % 4) * 20;       // 60..120: value spread (station heterogeneity)
    int wmax = 10;                           // bandwidth per beacon-station
    // tightness controls how much of the total demand can be served -> binding capacities
    double tightness = 0.40 + 0.04 * (testId % 3);   // 0.40..0.48

    // value / bandwidth matrices (1-indexed)
    vector<vector<int>> v(m + 1, vector<int>(n + 1));
    vector<vector<int>> w(m + 1, vector<int>(n + 1));

    // per-station "specialty" bias: some stations intrinsically value beacons more,
    // creating structure that rewards good matching over first-fit.
    vector<double> stationBias(m + 1);
    for (int i = 1; i <= m; i++) stationBias[i] = rnd.next(0.5, 1.5);
    // per-beacon base attractiveness
    vector<double> beaconBase(n + 1);
    for (int j = 1; j <= n; j++) beaconBase[j] = rnd.next(0.3, 1.0);

    long long totalW = 0;
    for (int j = 1; j <= n; j++) {
        for (int i = 1; i <= m; i++) {
            double frac = beaconBase[j] * stationBias[i] * rnd.next(0.4, 1.6);
            int val = (int)llround(frac * vmax);
            if (val < 1) val = 1;
            if (val > vmax) val = vmax;
            v[i][j] = val;
            w[i][j] = rnd.next(1, wmax);
            totalW += w[i][j];
        }
    }

    // capacities: sum of budgets ~ tightness * (average per-beacon bandwidth) * n
    double avgW = (double)totalW / (double)(m * n);
    double capMean = tightness * avgW * n / (double)m;
    if (capMean < wmax) capMean = wmax;      // ensure any single beacon can fit somewhere
    vector<int> c(m + 1);
    for (int i = 1; i <= m; i++) {
        int cap = (int)llround(capMean * rnd.next(0.7, 1.3));
        if (cap < wmax) cap = wmax;          // >= max beacon bandwidth so B>0
        if (cap < 10) cap = 10;
        if (cap > 400) cap = 400;
        c[i] = cap;
    }

    // shuffle beacon order so input order carries no signal (first-fit stays weak)
    vector<int> perm(n);
    for (int j = 0; j < n; j++) perm[j] = j + 1;
    shuffle(perm.begin(), perm.end());

    printf("%d %d\n", m, n);
    for (int i = 1; i <= m; i++) printf("%d%c", c[i], i == m ? '\n' : ' ');
    for (int jj = 0; jj < n; jj++) {
        int j = perm[jj];
        for (int i = 1; i <= m; i++) {
            printf("%d %d%c", v[i][j], w[i][j], i == m ? '\n' : ' ');
        }
    }
    return 0;
}
