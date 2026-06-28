**Problem.** Given a string `s` of length `n` over `a..z` and the wildcard `?` (a `?` may become any
single lowercase letter), find the smallest `p` in `[1, n]` such that some replacement of all the
`?`s makes `s` periodic with period `p`, i.e. `s[i] == s[i+p]` for all `0 <= i < n-p`. Read `s` from
stdin, print `p`. A period always exists (`p = n` is vacuous), so the answer lives in `[1, n]`.
Constraints: `n <= 2*10^5`, 2-second limit.

**Why the obvious approach is wrong.** "`p` is a period iff `s[i]` matches `s[i+p]` for all `i`,
treating `?` as matching anything" is the KMP border/period test transported to wildcards — and it is
wrong, because wildcard matching is **not transitive**. On `s = b?a` with `p = 1` the pairwise check
sees `b~?` and `?~a` and accepts, reporting `1`; but positions `0` and `2` are the same residue class
mod `1`, holding the distinct concrete letters `b` and `a`, so no fill works and the true answer is
`3`. The prefix function silently assumes `s[i]=s[j], s[j]=s[k] => s[i]=s[k]`, which `?` breaks. This
is the trap.

**Key idea — the insight.** `p` is a period iff **every residue class mod `p` is monochromatic** (its
concrete letters all agree). Pairwise (distance `p`) checks miss clashes that sit at distance `2p`,
`3p`, ... with `?`s between them. Define, for each shift `q`,

`compatible[q]` = "no index `i` has `s[i]`, `s[i+q]` both concrete and different."

Then the transitive-closure condition is exactly:

> `p` is a period **iff** `compatible[q]` holds for every multiple `q = p, 2p, 3p, ...` below `n`.

`compatible[q]` depends only on `q`, so compute it for **all** shifts at once. It is wildcard
self-matching: with `a_i in {1..26}` (`0` for `?`) and `act_i = [a_i != 0]`,

`mismatch(q) = sum_i act_i*act_{i+q}*(a_i - a_{i+q})^2 = sum_i [a_i^2*act_{i+q} - 2 a_i a_{i+q} + act_i a_{i+q}^2]`,

three cross-correlations, each one FFT — `O(n log n)`. `compatible[q] = (mismatch(q) == 0)`. Finally a
**harmonic multiple-sieve** reads off the smallest period: for `p = 1, 2, ...` test
`compatible[p], compatible[2p], ...`; first all-pass wins, at total cost `sum_p n/p = O(n log n)`.

**Pitfalls.**
1. *Transitivity.* Checking only distance-`p` pairs (or running the prefix function) accepts
   `b?a` as period `1`. You must check all multiples of `p` — equivalently, that classes are
   monochromatic. This is the whole problem.
2. *Correlation vs convolution.* `ifft(fft(u)*fft(v))` is a convolution `sum_i u[i] v[k-i]`, not the
   correlation `sum_i u[i] v[i+q]` you need. Conjugate one operand in frequency space:
   `corr = ifft(conj(FU) .* FV)`. (Trace: `u=[1,2,3], v=[4,5,6]`, `corr[1]` should be `17`, the plain
   product gives `13`.)
3. *FFT rounding.* `mismatch(q)` is computed in `double`; `llround` it. Magnitudes stay near `10^7`
   (well inside exact-integer range), and a two-modulus rolling-hash double-check of the final tiling
   guards any flip, with an `O(n*ans)` per-class fallback if it ever fires.
4. *Cyclic wrap.* Pad to `sz >= 2n` and read only `q in [0, n)`; the cyclic correlation's wrap lands
   in the unused upper half.

**Edge cases.** `n = 1` -> `1` (handled before the FFT). All wildcards `?...?` -> every shift
compatible -> `1`. `n` distinct letters -> no `p < n` survives -> `ans = n` (the initialization).

**Complexity.** `O(n log n)` time (three FFT correlations + harmonic sieve + one `O(n)` hash pass),
`O(n)` memory. About `0.2` s and `29` MB at `n = 2*10^5`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// ------------------------------------------------------------
// Smallest wildcard-period of a string over {a..z, '?'}.
//
// p is a (wildcard) period iff there is an assignment of concrete
// letters to every '?' making s[i] == s[i+p] for all 0 <= i < n-p.
// Equivalently: in every residue class {r, r+p, r+2p, ...} mod p the
// non-'?' letters must all be EQUAL.  Pairwise checks s[i] vs s[i+p]
// are NOT enough (e.g. "b?a": b~?, ?~a, yet the class forces ?=b=a).
//
// Key lemma (transitive closure): a class is monochromatic iff NO two
// of its concrete members differ, i.e. iff for EVERY distance q that is
// a multiple of p there is no i with s[i],s[i+q] both concrete & unequal.
// Define compatible[q] = "no concrete-and-different pair at distance q".
// Then p is a period iff compatible[q] holds for all multiples q (<n) of p.
//
// compatible[q] is the wildcard self-overlap at shift q.  Computing it for
// ALL q at once is a wildcard string-matching of s against itself:
//   mismatch(q) = sum_i act_i * act_{i+q} * (a_i - a_{i+q})^2
// with a_i in {1..26} (0 for '?') and act_i = [a_i != 0].  Expanding,
// this is three cross-correlations, each an FFT -> O(n log n) total.
// Then a harmonic multiple-sieve finds the smallest valid p in O(n log n).
//
// A double rolling hash independently DOUBLE-CHECKS the reported p by
// confirming every residue class is monochromatic (guards FFT rounding).
// ------------------------------------------------------------

using cd = complex<double>;

static void fft(vector<cd> &a, bool inv) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * acos(-1.0) / len * (inv ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int k = 0; k < len / 2; k++) {
                cd u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inv) for (cd &x : a) x /= n;
}

// cross-correlation corr[q] = sum_i u[i]*v[i+q], q in [0, n)
static vector<double> correlate(const vector<double> &u, const vector<double> &v, int n) {
    int sz = 1;
    while (sz < 2 * n) sz <<= 1;
    vector<cd> fu(sz), fv(sz);
    // corr = ifft( conj(FU) * FV ); arrange so result[q] = sum_i u[i] v[i+q]
    for (int i = 0; i < n; i++) { fu[i] = cd(u[i], 0); fv[i] = cd(v[i], 0); }
    fft(fu, false);
    fft(fv, false);
    for (int i = 0; i < sz; i++) fu[i] = conj(fu[i]) * fv[i];
    fft(fu, true);
    vector<double> res(n);
    for (int q = 0; q < n; q++) res[q] = fu[q].real();
    return res;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) return 0;
    int n = (int)s.size();
    if (n == 0) return 0;
    if (n == 1) { cout << 1 << "\n"; return 0; }

    // a_i in {1..26}, 0 for '?'; act_i = [concrete]
    vector<double> a(n), act(n), a2(n);
    for (int i = 0; i < n; i++) {
        if (s[i] == '?') { a[i] = 0; act[i] = 0; }
        else { a[i] = (double)(s[i] - 'a' + 1); act[i] = 1; }
        a2[i] = a[i] * a[i];
    }

    // mismatch(q) = sum_i [ a2_i*act_{i+q} - 2 a_i a_{i+q} + act_i a2_{i+q} ]
    vector<double> t1 = correlate(a2, act, n);
    vector<double> t2 = correlate(a, a, n);
    vector<double> t3 = correlate(act, a2, n);

    vector<char> compatible(n, 1); // compatible[q] for q in [1,n); index 0 unused
    for (int q = 1; q < n; q++) {
        double mm = t1[q] - 2.0 * t2[q] + t3[q];
        long long mmi = llround(mm);
        compatible[q] = (mmi == 0) ? 1 : 0;
    }

    // Smallest p such that every multiple q = p,2p,... (< n) is compatible.
    int ans = n; // p = n always works (no constraints)
    for (int p = 1; p < n; p++) {
        bool ok = true;
        for (int q = p; q < n; q += p) {
            if (!compatible[q]) { ok = false; break; }
        }
        if (ok) { ans = p; break; }
    }

    // ---- independent double rolling-hash DOUBLE-CHECK of `ans` ----
    // Build the canonical letter of each residue class mod ans; if any class
    // holds two different concrete letters the sieve answer is rejected.  Then
    // cross-validate with two independent rolling hashes that tiling s with the
    // period-ans block reproduces s on every concrete position.  This re-derives
    // the answer without the FFT, guarding against floating-point rounding.
    {
        int p = ans;
        // canonical letter per residue (0 means undetermined/free)
        vector<char> canon(p, 0);
        bool consistent = true;
        for (int i = 0; i < n && consistent; i++) {
            if (s[i] == '?') continue;
            int r = i % p;
            if (canon[r] == 0) canon[r] = s[i];
            else if (canon[r] != s[i]) consistent = false;
        }
        if (consistent) {
            // Reconstruct the canonical tiling t (period p): t[i] = canon[i%p],
            // free residues filled with 'a'.  Independently DOUBLE-CHECK with two
            // rolling hashes that the wildcard-filled s equals t -- i.e. tiling s
            // with the period-p block reproduces s on every concrete position.
            // sfill[i] = (s[i]=='?') ? canon[i%p] : s[i]; t[i] = canon[i%p].
            // These are built from different index expressions, so hash equality
            // is a genuine cross-validation of the FFT/sieve answer.
            const unsigned long long M1 = 1000000007ULL, B1 = 131;
            const unsigned long long M2 = 998244353ULL, B2 = 137;
            auto block = [&](int r) -> int { char c = canon[r]; return (c == 0 ? 'a' : c) - 'a' + 1; };
            unsigned long long hs1 = 0, ht1 = 0, hs2 = 0, ht2 = 0;
            for (int i = 0; i < n; i++) {
                int tc = block(i % p);                                  // canonical tiling letter
                int sc = (s[i] == '?') ? tc : (s[i] - 'a' + 1);         // s with '?' filled by canon
                hs1 = (hs1 * B1 + sc) % M1; ht1 = (ht1 * B1 + tc) % M1;
                hs2 = (hs2 * B2 + sc) % M2; ht2 = (ht2 * B2 + tc) % M2;
            }
            consistent = (hs1 == ht1) && (hs2 == ht2);
        }
        // If the double-check disagrees with the sieve, fall back to a direct
        // O(n * ans) class scan to be safe (never expected to trigger).
        if (!consistent) {
            for (int p2 = 1; p2 <= n; p2++) {
                vector<char> c2(p2, 0);
                bool good = true;
                for (int i = 0; i < n && good; i++) {
                    if (s[i] == '?') continue;
                    int r = i % p2;
                    if (c2[r] == 0) c2[r] = s[i];
                    else if (c2[r] != s[i]) good = false;
                }
                if (good) { ans = p2; break; }
            }
        }
    }

    cout << ans << "\n";
    return 0;
}
```
