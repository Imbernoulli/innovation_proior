#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;

int m;
ll k;
vector<ll> p;

// count of integers in [1, x] divisible by at least one p_i,
// via inclusion-exclusion over subsets, using lcm with an overflow cap.
ll countLit(ll x) {
    ll total = 0;
    for (int mask = 1; mask < (1 << m); mask++) {
        // build lcm of the chosen p_i; if it exceeds x, its contribution is 0
        // so we can stop early and treat the term as 0.
        ll lcmv = 1;
        bool overflow = false;
        for (int i = 0; i < m; i++) {
            if (mask & (1 << i)) {
                ll g = __gcd(lcmv, p[i]);
                ll step = p[i] / g;            // lcmv * step is the new lcm
                // overflow / over-x guard: if lcmv * step > x, this subset
                // contributes floor(x / lcm) == 0, so mark and bail.
                if (lcmv > x / step) { overflow = true; break; }
                lcmv *= step;
            }
        }
        if (overflow) continue;                // term is 0
        int bits = __builtin_popcount(mask);
        ll term = x / lcmv;
        if (bits & 1) total += term;           // odd-sized subset: add
        else total -= term;                    // even-sized subset: subtract
    }
    return total;
}

int main() {
    if (!(cin >> m >> k)) return 0;
    p.resize(m);
    for (auto &v : p) cin >> v;

    // Binary search for the smallest x with countLit(x) >= k.
    ll lo = 1, hi = (ll)2e18;
    while (lo < hi) {
        ll mid = lo + (hi - lo) / 2;
        if (countLit(mid) >= k) hi = mid;
        else lo = mid + 1;
    }
    cout << lo << "\n";
    return 0;
}
