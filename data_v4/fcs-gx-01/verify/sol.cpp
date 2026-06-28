#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> cost 0

    vector<long long> t(n), w(n);
    for (int i = 0; i < n; i++) cin >> t[i] >> w[i];

    // Single machine, minimize sum of w[i] * C[i], where C[i] is the completion
    // time (prefix sum of processing times) of job i in the chosen order.
    //
    // Exchange argument: for two adjacent jobs i (first) then j (second), the part
    // of the cost that depends on their relative order is, letting P be the time
    // accumulated before the pair,
    //     i before j:  w[i]*(P+t[i]) + w[j]*(P+t[i]+t[j])
    //     j before i:  w[j]*(P+t[j]) + w[i]*(P+t[j]+t[i])
    // The P-terms and the w*t self-terms cancel; the difference reduces to
    //     (i before j) - (j before i) = w[j]*t[i] - w[i]*t[j].
    // So putting i first is no worse exactly when w[j]*t[i] <= w[i]*t[j], i.e.
    //     t[i]*w[j] <= t[j]*w[i]   (Smith's ratio rule: ascending t/w).
    // Sorting by this total order is therefore optimal. Use cross-multiplication
    // (not the float ratio t/w) to stay exact; products fit in long long since
    // t,w <= 1e4 -> product <= 1e8, and the total cost <= ~2e18 < 9.2e18.
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int i, int j) {
        // i before j iff t[i]*w[j] < t[j]*w[i]; ties keep a stable, total order.
        long long lhs = t[i] * w[j];
        long long rhs = t[j] * w[i];
        if (lhs != rhs) return lhs < rhs;
        return i < j;                      // deterministic tie-break (cost is tie-invariant)
    });

    long long cost = 0;     // total weighted completion time, <= ~2e18, fits in signed 64-bit
    long long clock = 0;    // running completion time (prefix sum of processing times)
    for (int k = 0; k < n; k++) {
        int i = idx[k];
        clock += t[i];                     // completion time of job i
        cost += w[i] * clock;
    }

    cout << cost << "\n";
    return 0;
}
