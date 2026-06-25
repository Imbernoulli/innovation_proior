#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 -> empty schedule, cost 0
    vector<long long> p(n), w(n);
    for (int i = 0; i < n; i++) cin >> p[i] >> w[i];

    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);

    // Smith's rule: schedule by non-decreasing p/w.  Order i before j iff
    //   p_i / w_i <= p_j / w_j   <=>   p_i * w_j <= p_j * w_i   (w > 0).
    // Compare cross-products (each <= 10^18, fits in signed 64-bit) instead of
    // dividing as doubles, which loses the ordering on near-tied ratios.
    sort(idx.begin(), idx.end(), [&](int i, int j) {
        long long lhs = p[i] * w[j];      // <= 10^9 * 10^9 = 10^18 < 9.2e18
        long long rhs = p[j] * w[i];
        if (lhs != rhs) return lhs < rhs;
        return i < j;                     // deterministic tie-break (cost-neutral)
    });

    // Total weighted completion time.  Completion times reach sum(p) ~ 10^14;
    // each term w_i * C_i ~ 10^9 * 10^14 = 10^23, summed -> far past 64-bit.
    // Accumulate the running clock in 64-bit (<= 10^14, safe) but the objective
    // in __int128.
    long long clock_t = 0;                // running completion time
    __int128 total = 0;
    for (int k = 0; k < n; k++) {
        int i = idx[k];
        clock_t += p[i];                  // scene i finishes at this time
        total += (__int128)w[i] * clock_t;
    }

    // Print the __int128 result.
    if (total == 0) { cout << 0 << "\n"; return 0; }
    bool neg = total < 0;                 // never happens here, but be safe
    if (neg) total = -total;
    string s;
    while (total > 0) { s += char('0' + (int)(total % 10)); total /= 10; }
    if (neg) s += '-';
    reverse(s.begin(), s.end());
    cout << s << "\n";
    return 0;
}
