// TIER: greedy
// The textbook Bloom-filter recipe: allocate bits proportional to each
// checkpoint's watch-list size n_i (so every tier gets the SAME m_i/n_i
// ratio -- the classic "equalize the analytic false-positive rate" rule),
// then pick k_i by the closed-form optimum k = round((m_i/n_i)*ln2). Never
// looks at the trace, never touches SEED1/SEED2, and completely ignores that
// a false alarm at checkpoint 8 costs 128x one at checkpoint 1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int T; ll B, S1, S2;
    cin >> T >> B >> S1 >> S2;
    vector<ll> n(T + 1, 0);
    ll sumN = 0;
    for (int i = 1; i <= T; i++) {
        int ni; cin >> ni;
        n[i] = ni; sumN += ni;
        for (int j = 0; j < ni; j++) { ll x; cin >> x; }
    }

    const ll M_MIN = 64; const int K_MAX = 10;
    ll leftover = B - (ll)T * M_MIN;
    if (leftover < 0) leftover = 0;

    vector<ll> m(T + 1);
    ll used = 0;
    for (int i = 1; i < T; i++) {
        ll share = sumN > 0 ? (leftover * n[i]) / sumN : 0;
        m[i] = M_MIN + share;
        used += share;
    }
    m[T] = M_MIN + (leftover - used);

    for (int i = 1; i <= T; i++) {
        double ratio = (double)m[i] / (double)max(1LL, n[i]);
        int k = (int)llround(ratio * log(2.0));
        k = max(1, min(K_MAX, k));
        cout << m[i] << " " << k << "\n";
    }
    return 0;
}
