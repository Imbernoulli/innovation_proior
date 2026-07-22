// TIER: greedy
// "More traffic -> more liquidity": allocate beta_q in proportion to the
// number of traders M_q who show up at each stall. Never inspects a single
// belief or budget value -- so it cannot tell a self-agreeing crowd (which
// needs almost no liquidity) from a small, disagreeing, well-funded faction
// (which needs a carefully sized beta to resolve correctly).
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long N, B;
    double lambda;
    if (!(cin >> N >> B >> lambda)) return 0;
    vector<long long> M(N);
    for (long long q = 0; q < N; q++) {
        long long Mq, outcome;
        cin >> Mq >> outcome;
        M[q] = Mq;
        for (long long i = 0; i < Mq; i++) {
            long long belief, budget;
            cin >> belief >> budget;
        }
    }
    long long sumM = 0;
    for (auto m : M) sumM += m;
    if (sumM <= 0) sumM = 1;

    cout.setf(std::ios::fixed);
    cout << setprecision(6);
    for (long long q = 0; q < N; q++) {
        double beta = (double)B * (double)M[q] / (double)sumM;
        if (beta > 0.0 && beta < 0.01) beta = 0.0; // round tiny slivers down to "closed"
        cout << beta << "\n";
    }
    return 0;
}
