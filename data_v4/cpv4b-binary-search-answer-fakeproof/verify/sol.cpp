#include <bits/stdc++.h>
using namespace std;

// floor(sqrt(x)) for x >= 0, exact (guards against long double rounding).
long long isqrt_ll(long long x) {
    if (x <= 0) return 0;
    long long r = (long long)sqrtl((long double)x);
    while (r > 0 && r * r > x) r--;
    while ((r + 1) * (r + 1) <= x) r++;
    return r;
}

// D(m) = #{ (a,b) : a >= 1, b >= 1, a*b <= m }
//      = sum_{t=1..m} d(t)   where d(t) = number of divisors of t.
// Hyperbola / Dirichlet identity with s = floor(sqrt(m)):
//   D(m) = 2 * sum_{i=1..s} floor(m/i) - s*s.
long long D(long long m) {
    if (m <= 0) return 0;
    long long s = isqrt_ll(m);
    long long acc = 0;
    for (long long i = 1; i <= s; i++) acc += m / i;
    return 2 * acc - s * s;
}

int main() {
    int q;
    if (scanf("%d", &q) != 1) return 0;
    while (q--) {
        long long K;
        if (scanf("%lld", &K) != 1) break;
        // Find the smallest m >= 1 with D(m) >= K.
        // D is non-decreasing; D(1) = 1 so K >= 1 guarantees an answer.
        // For K <= 1e12 the answer m < 7e10, where D(7e10) > 1.7e12 >= K.
        long long lo = 1, hi = 70000000000LL; // 7e10
        while (lo < hi) {
            long long mid = lo + (hi - lo) / 2;
            if (D(mid) >= K) hi = mid;
            else lo = mid + 1;
        }
        printf("%lld\n", lo);
    }
    return 0;
}
