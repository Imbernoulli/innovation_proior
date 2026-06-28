**Problem.** Given `k` congruences `x ≡ r_i (mod m_i)` with moduli `m_i` up to `10^9` that are **not
assumed coprime** and may be **mutually contradictory**, output the smallest non-negative `x`
satisfying all of them, or `-1` if none exists. It is guaranteed `lcm(m_i) <= 10^18`, so the answer
prints as an ordinary integer. Read `k` then the `k` pairs from stdin; print one line.

**Why the obvious approach is wrong.** Textbook CRT (`x = Σ r_i (M/m_i)(M/m_i)^{-1} mod M`) needs the
inverse `(M/m_i)^{-1} mod m_i`, which **only exists when the moduli are pairwise coprime**. With shared
factors it cannot be written: for `x ≡ 0 (mod 4)`, `x ≡ 2 (mod 6)` it asks for `6^{-1} mod 4`, but
`gcd(6,4)=2`, so no inverse — even though the system *does* solve (`x = 8`). Classical CRT also has no
`-1` verdict at all. It is structurally the wrong tool.

**Key idea — iterative pairwise merge (generalized CRT).** Fold the congruences one at a time, keeping
a single running congruence `x ≡ r (mod m)` equivalent to all seen so far. To absorb `x ≡ r_i (mod
m_i)`, write `x = r + m·t`; then `m·t ≡ (r_i − r) (mod m_i)`, a linear congruence in `t`. By the theory
of linear congruences it is solvable **iff `g = gcd(m, m_i)` divides `(r_i − r)`** — *this single
divisibility test is both the merge rule and the contradiction detector.* When solvable, extended
Euclid gives `m·p + m_i·q = g`, so

```
t ≡ p · (r_i − r)/g   (mod m_i/g),    new modulus = lcm(m, m_i) = m/g · m_i,
new remainder = (r + m·t) mod lcm.
```

Seed with `x ≡ 0 (mod 1)` (every integer satisfies it) so the first real congruence merges with no
special case. No factoring of the moduli is required — the gcd does all the work.

**Pitfalls to get right.**
1. *Overflow.* `m` can reach `10^18` and `t` up to `~10^9`, so `m·t` reaches `~10^27` — nine orders of
   magnitude past 64-bit. Every multiplication must be done modulo with `__int128` as the wide
   intermediate: `(__int128)a*b % m`. This is the dominant failure mode of a naive implementation and
   the reason the large-modulus tests exist.
2. *Negative differences.* `r_i − r` can be negative; C++ `%` truncates toward zero. Route operands
   through a floor-mod into `[0, m)` before multiplying, but test divisibility on the raw `diff`
   (`diff % g == 0` is sign-agnostic).
3. *Inconsistency control flow.* On contradiction, set a flag and **keep draining the remaining
   congruences** to end-of-input (don't `break` mid-stream); decide the verdict once after the loop.
4. *lcm formation.* Write `m/g·m_i` (divide before multiply) to keep the intermediate minimal.

**Edge cases.** Single congruence; `m_i = 1`; duplicate moduli with equal remainder (merge is a no-op)
vs conflicting (`-1`); coprime systems (reduces to classical CRT); non-coprime consistent; non-coprime
contradictory; moduli near `10^9` so the merged modulus approaches `10^18` (overflow path).

**Complexity.** One extended-gcd (`O(log m_i)`) per congruence → `O(k log(max m))` time, `O(1)` extra
space. Trivially inside 1 s / 256 MB at `k = 10^5`.

**Code.**

```cpp
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
```
