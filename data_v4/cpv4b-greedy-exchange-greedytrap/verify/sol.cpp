#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> t(n), w(n);
    for (int i = 0; i < n; i++) cin >> t[i] >> w[i];

    // Smith's rule: process jobs in non-decreasing order of t/w.
    // Compare i before j  <=>  t[i]*w[j] < t[j]*w[i]   (cross-multiply, all positive).
    vector<int> ord(n);
    iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [&](int i, int j) {
        // primary key: t/w ascending  =>  t[i]*w[j] < t[j]*w[i]
        // products fit in long long: t,w <= 1e4 each => product <= 1e8
        long long lhs = t[i] * w[j];
        long long rhs = t[j] * w[i];
        if (lhs != rhs) return lhs < rhs;
        return i < j; // deterministic tie-break (does not affect cost)
    });

    long long cur = 0;      // running completion time
    long long total = 0;    // sum of w[i] * C[i]
    for (int idx : ord) {
        cur += t[idx];
        total += w[idx] * cur;
    }

    cout << total << "\n";
    return 0;
}
