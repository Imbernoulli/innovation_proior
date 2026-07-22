// TIER: invalid
// Deliberately infeasible: prints a negative liquidity for question 0 (violates
// beta_q >= 0), then otherwise-plausible values that would also blow the total
// budget if reached. Must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long N, B;
    double lambda;
    if (!(cin >> N >> B >> lambda)) return 0;
    for (long long q = 0; q < N; q++) {
        long long Mq, outcome;
        cin >> Mq >> outcome;
        for (long long i = 0; i < Mq; i++) {
            long long belief, budget;
            cin >> belief >> budget;
        }
    }
    cout.setf(std::ios::fixed);
    cout << setprecision(6);
    cout << -1.0 << "\n"; // infeasible: negative liquidity
    for (long long q = 1; q < N; q++) cout << (double)B << "\n"; // also blows the sum
    return 0;
}
