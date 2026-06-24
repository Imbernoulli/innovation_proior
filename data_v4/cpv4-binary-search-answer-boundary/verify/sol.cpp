#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long m, k;
    if (scanf("%d %lld %lld", &n, &m, &k) != 3) return 0;
    vector<long long> b(n);
    for (int i = 0; i < n; i++) scanf("%lld", &b[i]);

    // We want the earliest day t on which at least m bouquets can be assembled.
    // A bed i is bloomed on day t iff b[i] <= t (it blooms on day b[i] and stays).
    // A bouquet needs k consecutive bloomed beds; a maximal bloomed run of length
    // L yields floor(L/k) bouquets. Need the total >= m.

    // Impossibility: even with every bed bloomed we have n beds, giving at most
    // floor(n/k) bouquets. If m*k > n, it can never be done.
    if ((__int128)m * (__int128)k > (__int128)n) {
        printf("-1\n");
        return 0;
    }

    // The earliest feasible day is necessarily one of the bloom days, so the
    // search range is [min b, max b], inclusive on both ends.
    long long lo = LLONG_MAX, hi = LLONG_MIN;
    for (int i = 0; i < n; i++) { lo = min(lo, b[i]); hi = max(hi, b[i]); }

    auto feasible = [&](long long t) -> bool {
        long long bouquets = 0, run = 0;
        for (int i = 0; i < n; i++) {
            if (b[i] <= t) {                 // bloomed by day t (inclusive)
                run++;
                if (run == k) { bouquets++; run = 0; }
                if (bouquets >= m) return true;
            } else {
                run = 0;
            }
        }
        return bouquets >= m;
    };

    // Binary search for the minimal feasible t in [lo, hi].
    // feasible(hi) is true because on day hi every bed is bloomed => one run of
    // length n => floor(n/k) >= m (from the m*k <= n check above).
    long long L = lo, R = hi;
    while (L < R) {
        long long mid = L + (R - L) / 2;     // floors toward L; no overflow
        if (feasible(mid)) R = mid;          // mid works: answer in [L, mid]
        else L = mid + 1;                    // mid fails: answer in [mid+1, R]
    }
    printf("%lld\n", L);
    return 0;
}
