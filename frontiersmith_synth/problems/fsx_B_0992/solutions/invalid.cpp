// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: on day 1, claim to melt CAP+1000 tons of virgin
// metal -- far more than the furnace's daily capacity (and, separately, more
// than any plausible remaining lifetime budget). The checker's bounded read
// of virgin_1 against min(CAP, remaining V) must reject this immediately.
int main() {
    int T;
    cin >> T;
    double CAP, V, RF, PPM_CAP, P0, CV, BETA;
    int LAG;
    cin >> CAP >> V >> RF >> LAG >> PPM_CAP >> P0 >> CV >> BETA;

    cout << setprecision(9) << fixed;
    for (int t = 1; t <= T; t++) {
        int K;
        cin >> K;
        for (int j = 0; j < K; j++) {
            double a, p, pr;
            cin >> a >> p >> pr;
        }
        if (t == 1) {
            cout << (CAP + 1000.0);
        } else {
            cout << 0.0;
        }
        for (int j = 0; j < K; j++) cout << ' ' << 0.0;
        cout << ' ' << 0.0 << '\n';
    }
    return 0;
}
