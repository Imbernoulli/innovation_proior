#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: telescope array observing schedule ----
    // testId 1 tiny (example scale); grows to a large, constrained instance by testId 10.
    int T = 3 + testId / 2;                 // 3,4,4,5,5,6,6,7,7,8
    int N = 12 + 18 * testId;               // 30 .. 192

    // budgets: some tests tight, some loose -> capacity pressure varies
    int budgetLo = 60 + (testId % 3) * 30;  // 60,90,120,...
    int budgetHi = budgetLo + 120;

    // cost range grows a little with testId
    int costLo = 5;
    int costHi = 40 + (testId % 4) * 20;    // 40,60,80,100,...

    // value model: odd tests correlate value with cost (value/minute greedy matters),
    // even tests use independent values (value greedy vs ratio greedy diverge).
    bool correlated = (testId % 2 == 1);

    vector<int> C(T);
    for (int j = 0; j < T; j++) C[j] = rnd.next(budgetLo, budgetHi);

    // cost[i][j], val[i][j]
    vector<vector<int>> cost(N, vector<int>(T)), val(N, vector<int>(T));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < T; j++) {
            int c = rnd.next(costLo, costHi);
            int v;
            if (correlated) {
                // value roughly proportional to cost plus per-telescope noise
                double factor = rnd.next(6, 30) / 10.0;   // 0.6 .. 3.0 value per minute
                v = (int)llround(c * factor) + rnd.next(1, 30);
            } else {
                v = rnd.next(1, 1000);
            }
            if (v < 1) v = 1;
            if (v > 1000) v = 1000;
            cost[i][j] = c;
            val[i][j]  = v;
        }
    }

    // Guarantee the first-fit reference is positive: force target 1 to fit telescope 1.
    cost[0][0] = rnd.next(1, min(costHi, C[0]));

    printf("%d %d\n", N, T);
    for (int j = 0; j < T; j++) printf("%d%c", C[j], j + 1 < T ? ' ' : '\n');
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < T; j++) {
            printf("%d %d%c", cost[i][j], val[i][j], j + 1 < T ? ' ' : '\n');
        }
    }
    return 0;
}
