**Problem.** For an integer `x` with no leading zeros, let `len(x)` be its digit count and `S(x)` its
digit sum. Call `x` *balanced* when `len(x)` divides `S(x)`. Given `1 <= L <= R <= 10^18`, count the
balanced integers in `[L, R]`. Read `L R` from stdin, print the count. The count can reach `~5.6 * 10^16`
(over `[1, 10^18]`), so the answer and all accumulators are 64-bit.

**Reduce to a prefix count.** Let `f(N)` count balanced integers in `[1, N]`, with `f(N) = 0` for
`N <= 0`. The answer is `f(R) - f(L-1)`. Everything reduces to computing `f(N)` for one bound up to
`10^18`.

**Why the obvious approaches fail.**
- *Scan `1..N`.* Trivially correct (it is the brute oracle) but `N` is `10^18`; a linear scan is decades
  of compute. Out.
- *Plain digit DP with one modulus.* The textbook way to count `x <= N` with "digit sum divisible by a
  fixed `m`" carries `S mod m` and is `O(digits * m)`, independent of `N`. But here the divisor is
  `len(x)` — the candidate's *own* length — which a most-significant-first walk has not yet committed to
  while choosing leading digits. There is no single residue to carry, because `S mod 5` and `S mod 6`
  are different arithmetic and you don't yet know which length you'll land on. Applied directly, the DP
  counts against the wrong divisor and is simply wrong.

**Key idea — partition by length so the modulus becomes constant.** Split `[1, N]` by digit count:
numbers with exactly `1, 2, ..., D` digits, where `D = len(N)`. Within the part of exactly-`len`-digit
numbers, the modulus is the *constant* `len`. That collapses each part to the classic "digit sum
divisible by a fixed modulus" digit DP. The bounds per part are clean:
- For every `len < D`: every `len`-digit number is `< N` (fewer digits), so the bound is `99...9`
  (`len` nines) — unrestricted; count all qualifying `len`-digit numbers.
- For `len = D`: the `D`-digit numbers `<= N` are bounded by `N`'s own digits.

Each integer has exactly one length, so the parts are disjoint and cover `[1, N]` with no overlap.
Sum the per-length counts. Total cost is `O(D^2 * 10)` per prefix (`D <= 19`) — microseconds.

**Per-length DP (tight prefix + free-suffix table).** For fixed `len` and `mod = len`, count
`len`-digit numbers (no leading zero) `<= bound` with digit sum `≡ 0 (mod mod)`:
- Precompute `suf[p][s]` = number of ways to fill positions `p..len-1` with free digits `0..9` whose
  *own* digit sum is `≡ s (mod mod)`. Base `suf[len][0] = 1`. Recurrence: placing digit `d` at `p`
  leaves residue `s - d` for the rest, so `suf[p][s] = sum_d suf[p+1][(s - d) mod mod]`.
- Walk the tight prefix carrying residue `r`. At position `p`, for each digit `d` strictly below the
  bound digit (and `d >= 1` at position `0`), the suffix becomes free and contributes
  `suf[p+1][(mod - (r+d)%mod) % mod]`. Continue tight with `d = bound[p]`. Finally add `1` if the bound
  itself is reachable and totals `≡ 0`.

**Pitfalls.**
1. *Sign of the suffix recurrence.* The lookup asks for suffix-sum `≡ need`; the table build must match
   that meaning, i.e. index `suf[p+1][(s - d) mod mod]`, **not** `(s + d)`. The `+d` form builds the
   table for residue `-s` instead of `s`. By digit-set symmetry the two tables agree for many residues,
   so the bug hides on small cases and surfaces only for specific `(mod, s)` — e.g. it miscounts `1010`
   (digit sum `2`, length `4`, not balanced) as balanced, first showing up on the range `[997, 1012]`.
   Compute `((s - d) % mod + mod) % mod` so the index is non-negative in C++.
2. *No leading zero.* At position `0` the digit ranges over `1..9`, not `0..9`; otherwise you would count
   shorter numbers under a longer length.
3. *Overflow.* The count over `[1, 10^18]` is `~5.6 * 10^16`; use `long long` everywhere.
4. *Lower bound.* For `L = 1`, `f(L-1) = f(0) = 0` via the `N <= 0` guard.

**Edge cases.** Single digits (`len = 1`, divisor `1`) are all balanced: `[1,9] -> 9`. Power-of-ten
boundaries change the divisor mid-range: `[999,1000] -> 1` (only `999`). `R = 10^18` uses the
19-digit tight path. Single-point ranges `L = R` work via `f(R) - f(R-1)`.

**Complexity.** `O(D^2 * 10)` time and `O(D^2)` memory per prefix evaluation, `D = len(N) <= 19`;
independent of `N`'s magnitude. Comfortably within the `10^18` / 1-second limit.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Count integers x in [1, N] (N >= 0) whose decimal digit sum is divisible by
// the number of decimal digits of x (its length, no leading zeros).
//
// The divisor depends on the length of x, which is non-local, so we cannot run
// one digit DP with a single modulus. Instead we group by length: for every
// target length len (1..19) the modulus is exactly len and fixed, so a standard
// digit DP over numbers of that length works. We count, for each len, how many
// numbers with EXACTLY len digits (leading digit 1..9) lie in [1, N] and have
// digit sum divisible by len.

typedef long long ll;

// Count numbers of exactly `len` digits, value <= bound, no leading zero,
// digit sum divisible by `mod` (== len). `bd` is the digit array of the upper
// bound (most significant first) of length exactly `len`. Returns the count.
//
// Technique: tight-prefix walk + precomputed free-suffix counts. We precompute
// suf[p][s] = number of ways to fill positions p..len-1 with free digits 0..9
// whose own digit sum is ≡ s (mod). Then we walk the tight prefix; at each
// position, every digit strictly below the bound digit "releases" a free suffix,
// whose count we read straight out of suf[]. Finally we add the bound itself if
// it qualifies.
ll countLen(int len, int mod, const vector<int>& bd) {
    int npos = len;
    // suf[p][s] = number of digit strings for positions p..len-1 (each digit
    // 0..9) whose OWN digit sum ≡ s (mod `mod`). Recurrence: placing digit d at
    // position p needs the remaining suffix (p+1..) to have digit sum ≡ s-d.
    vector<vector<ll>> suf(npos + 1, vector<ll>(mod, 0));
    suf[npos][0] = 1; // empty suffix has digit sum 0
    for (int p = npos - 1; p >= 0; --p) {
        for (int s = 0; s < mod; ++s) {
            ll acc = 0;
            for (int d = 0; d <= 9; ++d) {
                int ns = ((s - d) % mod + mod) % mod;
                acc += suf[p + 1][ns];
            }
            suf[p][s] = acc;
        }
    }
    // Walk the tight path. At position p, residue r (digit sum so far mod `mod`).
    // We must respect: at position 0 the digit range is 1..9 (no leading zero),
    // at positions >0 it is 0..9.
    ll total = 0;
    int r = 0;        // current residue along the tight prefix
    bool feasible = true; // whether the exact-bound path is still alive
    for (int p = 0; p < npos; ++p) {
        int lo = (p == 0) ? 1 : 0;
        int hi = bd[p];
        // place digit d in [lo, hi-1] strictly less than bound digit -> free suffix
        for (int d = lo; d < hi; ++d) {
            int nr = (r + d) % mod;
            int need = (mod - nr) % mod; // suffix residue needed to total 0
            total += suf[p + 1][need];
        }
        // continue tight with d == bd[p], but only if it respects lo
        if (bd[p] < lo) { feasible = false; break; }
        r = (r + bd[p]) % mod;
    }
    if (feasible && r == 0) total += 1; // the bound itself, if its digit sum ≡ 0
    return total;
}

// Count x in [1, N] with digitSum(x) % len(x) == 0.  N >= 0.
ll countUpTo(ll N) {
    if (N <= 0) return 0;
    // decompose N into digits
    string s = to_string(N);
    int D = (int)s.size(); // number of digits of N
    ll ans = 0;
    // For lengths 1 .. D-1: all numbers of that length are fully below N's range
    // upper bound is the maximal len-digit number (all 9s), i.e. unrestricted.
    // For length D: bounded by N.
    for (int len = 1; len <= D; ++len) {
        int mod = len; // length-dependent modulus
        vector<int> bd(len);
        if (len < D) {
            // upper bound = 99...9 (len nines): every len-digit number qualifies
            for (int i = 0; i < len; ++i) bd[i] = 9;
        } else {
            for (int i = 0; i < len; ++i) bd[i] = s[i] - '0';
        }
        ans += countLen(len, mod, bd);
    }
    return ans;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    ll L, R;
    if (!(cin >> L >> R)) return 0;
    ll ans = countUpTo(R) - countUpTo(L - 1);
    cout << ans << "\n";
    return 0;
}
```
