#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;

// lcm with saturation: if the true lcm exceeds CAP it can never divide any x in range,
// so we cap it (any x/cap then contributes 0). CAP must exceed the maximum hi.
static const ll CAP = (ll)4e18;

ll gcd_ll(ll a, ll b) { while (b) { ll t = a % b; a = b; b = t; } return a; }

// lcm(a, b) saturated at CAP (returns a value > any x we ever query when it overflows)
ll lcm_sat(ll a, ll b) {
    ll g = gcd_ll(a, b);
    ll q = a / g;                       // a/g * b ; check overflow against CAP
    if (q > CAP / b) return CAP + 1;    // would exceed CAP -> never divides any x in [1..CAP]
    ll v = q * b;
    return v;
}

int main() {
    int n;
    if (!(cin >> n)) return 0;          // n = number of machines (1..3)
    vector<ll> p(n);
    for (auto &x : p) cin >> x;
    ll K;
    cin >> K;

    // count(x) = number of DISTINCT times t in [1..x] that are a multiple of at least one p[i].
    // Inclusion-exclusion over the (up to 3) periods. Each subset contributes
    // sign * floor(x / lcm(subset)); singletons +, pairs -, triple +.
    // Pitfall avoided: subtract the lcm-overlap terms so common multiples are counted ONCE.
    auto countLE = [&](ll x) -> ll {
        ll total = 0;
        for (int mask = 1; mask < (1 << n); ++mask) {
            ll L = 1;
            bool overflow = false;
            for (int i = 0; i < n; ++i) if (mask & (1 << i)) {
                L = lcm_sat(L, p[i]);
                if (L > CAP) { overflow = true; break; }
            }
            int bits = __builtin_popcount((unsigned)mask);
            ll contrib = overflow ? 0 : (x / L);
            if (bits & 1) total += contrib;     // odd-size subset: +
            else          total -= contrib;     // even-size subset: -
        }
        return total;
    };

    // smallest x with countLE(x) >= K. Upper bound: K * min(p) is reachable and has count >= K.
    ll mn = *min_element(p.begin(), p.end());
    ll lo = 1, hi = mn * K;             // hi <= 1e9 * 1e9 = 1e18, fits in ll
    while (lo < hi) {
        ll mid = lo + (hi - lo) / 2;
        if (countLE(mid) >= K) hi = mid;
        else lo = mid + 1;
    }

    cout << lo << "\n";
    return 0;
}
