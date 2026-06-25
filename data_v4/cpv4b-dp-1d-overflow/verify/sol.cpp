#include <bits/stdc++.h>
using namespace std;

int main() {
    const long long MOD = 1000000007LL;
    long long n, m;
    if (!(cin >> n >> m)) return 0;

    // lo = number of valid length-i sequences whose last round is a non-peak
    //      score (one of the m values 0..m-1);
    // hi = number whose last round is the peak score m (the single value m).
    // A peak may not immediately follow a peak.
    //
    // For i = 1: lo = m (scores 0..m-1), hi = 1 (the peak).
    // Transition for one more round:
    //   new_lo = (lo + hi) * m   (any previous ending may be followed by a
    //                             non-peak, and there are m non-peak scores)
    //   new_hi = lo              (a peak may only follow a non-peak; 1 score)
    //
    // n can be 0 (empty sequence: exactly one, the empty one).
    long long lo = (m % MOD), hi = 1 % MOD;
    if (n == 0) { cout << 1 % MOD << "\n"; return 0; }

    for (long long i = 2; i <= n; i++) {
        long long nlo = ((lo + hi) % MOD) * (m % MOD) % MOD;
        long long nhi = lo;
        lo = nlo;
        hi = nhi;
    }

    cout << (lo + hi) % MOD << "\n";
    return 0;
}
