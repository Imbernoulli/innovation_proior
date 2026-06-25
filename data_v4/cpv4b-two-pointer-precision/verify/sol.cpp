#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long p, q;
    if (!(cin >> n >> p >> q)) return 0;
    vector<long long> g(n);
    for (auto &x : g) cin >> x;

    // A pair {i, j} (i != j), say with g_lo <= g_hi, is "balanced" when
    //   g_hi / g_lo <= p / q   <=>   g_hi * q <= g_lo * p   (q, p, values all > 0).
    // Sort ascending; for each right endpoint j the admissible left endpoints i (with i < j and
    // g[j]*q <= g[i]*p) form a suffix [lo, j-1] of the sorted prefix, and lo only moves rightward
    // as j advances -> a single two-pointer sweep. With values and p, q up to 4*10^9 the
    // cross-products reach ~1.6*10^19, which OVERFLOWS signed 64-bit; we form them in __int128 so no
    // division or floating point is ever used.
    sort(g.begin(), g.end());

    long long count = 0;     // up to n*(n-1)/2 ~ 2*10^10 at n = 2*10^5, must be 64-bit
    int lo = 0;
    for (int j = 0; j < n; j++) {
        // advance lo to the smallest i with g[j]*q <= g[i]*p, i.e. the pair (i, j) is balanced.
        while (lo < j && (__int128)g[j] * q > (__int128)g[lo] * p) lo++;
        // now i in [lo, j-1] are exactly the partners that pair with j as the larger element.
        count += (long long)(j - lo);
    }

    cout << count << "\n";
    return 0;
}
