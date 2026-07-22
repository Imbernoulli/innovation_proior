// TIER: trivial
// Uniform split: beta_q = B/N for every question. Reproduces the checker's
// own internal baseline construction exactly (-> ratio ~0.1).
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
    double beta = (double)B / (double)N;
    cout.setf(std::ios::fixed);
    cout << setprecision(6);
    for (long long q = 0; q < N; q++) cout << beta << "\n";
    return 0;
}
