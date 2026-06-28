#include <bits/stdc++.h>
using namespace std;

// Extended Euclid: returns g = gcd(a,b) and sets x,y with a*x + b*y = g.
// Works on non-negative a,b (we only feed moduli, which are >= 1).
long long ext_gcd(long long a, long long b, long long &x, long long &y) {
    if (b == 0) { x = 1; y = 0; return a; }
    long long x1, y1;
    long long g = ext_gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return g;
}

// Floor-mod into [0, m): handles negative a safely.
long long mod_floor(long long a, long long m) {
    long long r = a % m;
    if (r < 0) r += m;
    return r;
}

// (a * b) mod m using __int128 to avoid 64-bit overflow.
long long mulmod(long long a, long long b, long long m) {
    return (long long)((__int128)a * b % m);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int k;
    if (!(cin >> k)) return 0;

    // Running solution: x ≡ r (mod m). Start with the trivial congruence x ≡ 0 (mod 1),
    // which every integer satisfies, so the first real congruence is absorbed cleanly.
    long long r = 0, m = 1;
    bool ok = true;

    for (int i = 0; i < k; i++) {
        long long ri, mi;
        cin >> ri >> mi;
        ri = mod_floor(ri, mi); // normalize remainder into [0, mi)
        if (!ok) continue;      // already inconsistent; just consume the rest

        // Merge x ≡ r (mod m) with x ≡ ri (mod mi).
        // Need m*t ≡ (ri - r) (mod mi); solvable iff g = gcd(m,mi) divides (ri - r).
        long long p, q;
        long long g = ext_gcd(m, mi, p, q); // m*p + mi*q = g
        long long diff = ri - r;
        if (diff % g != 0) { ok = false; continue; } // contradiction

        long long lcm = m / g * mi;          // lcm(m,mi); guaranteed <= 1e18 by constraints
        // Particular t0 = p * (diff/g) (mod mi/g); then x = r + m*t0 (mod lcm).
        long long mod2 = mi / g;             // step for t
        long long t = mulmod(mod_floor(p, mod2), mod_floor(diff / g, mod2), mod2);
        // new remainder = (r + m*t) mod lcm, all done with overflow-safe mulmod/add.
        long long add = mulmod(mod_floor(m, lcm), mod_floor(t, lcm), lcm);
        r = mod_floor(r + add, lcm);
        m = lcm;
    }

    if (!ok) {
        cout << -1 << "\n";
    } else {
        cout << r << "\n"; // smallest non-negative solution, in [0, m)
    }
    return 0;
}
