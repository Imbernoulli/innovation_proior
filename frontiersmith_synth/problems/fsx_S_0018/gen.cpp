#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    const int WMAX = 10;    // max kilowatts per act-platform
    const int VMAX = 100;   // max thrill value

    // ---- structure ladder ----
    // testId 1 is tiny (example scale); grows to a large dense circuit by testId 10.
    int m = 2 + 3 * (testId - 1);   // 2, 5, 8, ..., 29 platforms
    int n = 6 * m + 2 * testId;     // heavy competition for scarce power

    // per-test structural variety: how tight generators are, and whether thrill
    // is correlated with power draw (harder instances).
    int slack = testId % 4;                    // 0..3 extra budget spread
    bool correlated = (testId % 2 == 0);       // even tests: high draw -> high thrill

    // ---- generator budgets (tight: ~2 acts per platform fit) ----
    vector<int> C(m + 1);
    for (int j = 1; j <= m; j++) {
        int base = WMAX + rnd.next(0, 6 + slack);   // 10 .. ~19 kW
        C[j] = max(base, WMAX);                      // guarantee any single act fits
        if (C[j] > 60) C[j] = 60;
    }

    // ---- thrill (value) and power (weight) matrices ----
    // stored row-major: entry (i,j)
    vector<vector<int>> v(n + 1, vector<int>(m + 1));
    vector<vector<int>> w(n + 1, vector<int>(m + 1));
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            int wij = rnd.next(1, WMAX);
            wij = min(wij, C[j]);   // enforce w[i][j] <= C[j]
            int vij;
            if (correlated) {
                // thrill scales with draw plus noise -> classic hard GAP
                vij = wij * 8 + rnd.next(0, 20);
                vij = min(VMAX, max(1, vij));
            } else {
                vij = rnd.next(1, VMAX);
            }
            w[i][j] = wij;
            v[i][j] = vij;
        }
    }

    // ---- emit ----
    printf("%d %d\n", m, n);
    for (int j = 1; j <= m; j++) printf("%d%c", C[j], j == m ? '\n' : ' ');
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) printf("%d%c", v[i][j], j == m ? '\n' : ' ');
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) printf("%d%c", w[i][j], j == m ? '\n' : ' ');
    return 0;
}
