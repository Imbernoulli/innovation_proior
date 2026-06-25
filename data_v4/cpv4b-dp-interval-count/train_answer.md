**Problem.** A label is a row of `n` cells, each painted one of `k` colors. The printer can only lay a *stripe* — a maximal block of equal-color cells — whose length is in `[A, B]`. Count the distinct labels (final color strings) of length `n` in which every maximal monochromatic run has length in `[A, B]`, modulo `M`. Read `n k A B M` from stdin, print the count. The empty label (`n = 0`) is vacuously valid, so its count is `1` (i.e. `1 % M`).

**Key idea — run-ending interval DP.** A label *is* its string, not its stripe sequence, so two adjacent runs may never share a color (or they would merge into one run). Define `f[i]` = number of valid colorings of cells `1..i` in which a maximal run **ends exactly at `i`**. Keying on "a run ends here" counts every label exactly once, at the position where its last run terminates. The last run has length `len in [A, B]` covering `(i-len+1 .. i)`; let `p = i - len`:

- `p == 0`: the whole prefix is one run, any of `k` colors — legal only when `len = i` is in `[A, B]`.
- `p >= 1`: a run ends at `p` (count `f[p]`), and the new run must take a **different** color than the previous run — `(k-1)` choices.

So `f[i] = (k-1) * sum_{p in [max(1,i-B), i-A]} f[p] + [A <= i <= B] * k`, evaluated in `O(n)` with a prefix-sum array over the sliding window `p in [i-B, i-A]`. Answer is `f[n]`.

**Pitfalls.**
1. *Double-count via the color factor.* The window sum must be multiplied by `(k-1)`, not `k`. Using `k` lets a new run repeat the previous run's color, so a merged run gets counted both as one long run (the `+k` first-run / earlier window term) and as two same-color shorter runs. Trace `n=2,k=2,A=1,B=2`: truth is `4` (all of `00,01,10,11`); the `* k` bug yields `6`, double-counting `00` and `11`.
2. *Window boundary / the phantom `p = 0`.* From `A <= i - p <= B` the previous-end positions are `p in [max(1,i-B), i-A]`; the guard `phi >= plo` with `phi = i-A` must exclude `p = 0` (that case is the first-run `+k` term, handled separately). At `i = A`, `phi = 0 < plo = 1`, so the window correctly does not fire.
3. *Overflow.* A partial sum is `< M <= 10^9` and `(k-1) < 10^9`, so the product is `< 10^18` — use `long long`; an `int` overflows immediately.

**Edge cases.** `n = 0` -> `1 % M` (vacuously valid empty label; `0` when `M = 1`). `k = 1` -> only the single-run first-case can fire, so answer is `1` iff `n in [A,B]` else `0` (the `(k-1)=0` factor kills all window terms). Wide window `A = 1, B >= n` -> every string valid, answer `k^n mod M`. `M = 1` -> always `0`. Single negative-free reductions keep `s` non-negative via `if (s < 0) s += M`.

**Complexity.** `O(n)` time, `O(n)` space (two `long long` arrays); ~0.05 s and ~34 MB at `n = 2*10^6`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, k, A, B, M;
    if (!(cin >> n >> k >> A >> B >> M)) return 0;   // empty input

    // Count strings of length n over k colors such that every MAXIMAL
    // monochromatic run has length in [A, B], modulo M.
    //
    // The empty string (n == 0) has no runs, so it is vacuously valid: 1 string.
    if (n == 0) { cout << (1 % M) << "\n"; return 0; }

    // f[i] = number of valid colorings of cells 1..i in which a maximal run
    //        ENDS exactly at position i. Keying on "a run ends here" is what
    //        prevents double counting: every valid string is counted once, at
    //        the position where its last run terminates.
    //
    // A run ending at i has length len in [A, B], so it occupies cells
    // (i-len+1 .. i). Let p = i - len be the position just before the run.
    //   - p == 0: this is the FIRST run; it may be any of the k colors.
    //             Valid only when len == i lies in [A, B].
    //   - p >= 1: a run ends at p, and the new run's color must DIFFER from the
    //             previous run's color -> (k-1) choices. Contributes
    //             f[p] * (k-1).
    //
    // Answer = f[n].
    long long kmod = k % M;
    long long km1  = (k - 1) % M;             // k >= 1, so k-1 >= 0

    vector<long long> f(n + 1, 0);
    // pref[i] = (f[0] + f[1] + ... + f[i]) mod M, with the convention f[0] = 0
    // for the prefix only; the FIRST-run case is handled separately so we never
    // confuse the empty prefix with a real "run ends at 0".
    vector<long long> pref(n + 1, 0);

    for (long long i = 1; i <= n; i++) {
        // Window of valid previous-end positions p so that len = i - p is in
        // [A, B] and p >= 1 (non-first run):  A <= i - p <= B  =>
        //   p in [i - B, i - A], intersected with [1, i-1].
        long long plo = max(1LL, i - B);
        long long phi = i - A;                // len >= A  =>  p <= i - A
        long long ways = 0;
        if (phi >= plo) {
            // sum of f[plo..phi] via prefix sums
            long long s = pref[phi] - (plo >= 1 ? pref[plo - 1] : 0);
            s %= M; if (s < 0) s += M;
            ways = s % M * (km1 % M) % M;     // (k-1) color choices for new run
        }
        // First-run case: the whole prefix 1..i is one run, length i in [A,B].
        if (i >= A && i <= B) {
            ways = (ways + kmod) % M;
        }
        f[i] = ways % M;
        pref[i] = (pref[i - 1] + f[i]) % M;
    }

    cout << (f[n] % M) << "\n";
    return 0;
}
```
