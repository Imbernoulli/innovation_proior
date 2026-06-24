**Problem.** A ring of `n` bells has signed integer mistunings `a[0..n-1]` (negative = flat, `0` = in
tune, positive = sharp). A master step of `d` cents, applied any whole number of times in either
direction, corrects bell `i` iff `d | a[i]`. For each of `q` queries `(l, r)` (1-indexed, inclusive),
report the **largest** positive `d` that corrects every bell in the block — i.e. the gcd of the block's
magnitudes — or `0` if the entire block is already in tune (all zeros). Read from stdin, write one
answer per line.

**Key idea — range GCD of magnitudes.** Since `d | a[i]` iff `d | |a[i]|`, replace every value by
`|a[i]|`. The largest common corrective step over a block is `gcd(|a[l]|, ..., |a[r]|)`. A zero bell
is already in tune, every `d` divides `0`, so `0` acts as the **identity** of the gcd fold
(`gcd(0, x) = x`); an all-zero block has no largest step and the convention `gcd(0, ..., 0) = 0`
delivers exactly the required `0`. Because gcd is associative *and idempotent* (`gcd(x, x) = x`),
overlapping covering blocks may be combined freely — so a **sparse table** answers each query in
`O(1)` after an `O(n log n)` build: cover `[l, r]` by `[l, l+2^k)` and `(r-2^k, r]` with
`k = floor(log2(r-l+1))` and gcd the two.

**Correctness.** Level 0 holds `|a[i]|`. Level `k` holds the gcd of every length-`2^k` window, built
by `gcd` of two adjacent length-`2^{k-1}` windows; by associativity each `sp[k][i]` equals the gcd of
its window. A query gcds two windows of length `2^k` whose union is exactly `[l, r]` (they overlap in
the middle when `len` is not a power of two); idempotence makes the overlap harmless, so the result is
the gcd over `[l, r]`. The seed/identity `0` makes zeros transparent and an all-zero range return `0`.

**Pitfalls.**
1. *Wrong base case (the heart).* Seed the fold with the gcd identity `0`, never `1`. Seeding with `1`
   ("the gcd is at least 1") turns an all-zero / all-in-tune block into the lie `1`; only `0` gives
   `gcd(0, x) = x` (zeros transparent) and `gcd(0, 0) = 0` (all-zero -> `0`). No element should ever be
   special-cased — the algebra is correct precisely when the seed is `0`.
2. *Sign handling.* Fold magnitudes, using `llabs` (the `long long` abs), not the `int` `abs`, so a
   flat bell `a[i] < 0` and the extremes `±10^9` are stripped to magnitude reliably.
3. *Floating `log2`.* `log2(len)` can return e.g. `2.9999999` and truncate to the wrong `k`, dropping
   elements (too-large gcd) or indexing out of bounds. Use `k = 31 - __builtin_clz(len)` (exact, and
   `len >= 1` so `clz` is never called on `0`).
4. *Table dimension off-by-one.* For power-of-two `n`, the largest query needs row `floor(log2 n)`; the
   `LOG++` headroom guarantees that row exists.

**Edge cases.** All-zero block -> `0`; all-negative block -> gcd of magnitudes (sign stripped); single
bell `(l = r)` uses `len = 1, k = 0` and reads `sp[0]` only; mixed zeros and nonzeros -> zeros drop out
(`[0,30,0] -> 30`); extremes `±10^9` fit in `long long` and never overflow (gcd only shrinks values).

**Complexity.** `O(n log n)` build, `O(1)` per query, `O(n log n)` memory. A per-query gcd scan would
be `O(q*n)` (up to `4*10^10`) and time out, so the precompute is mandatory.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// gcd on non-negative long long, with gcd(0,0)=0.
static long long g2(long long x, long long y) {
    while (y) { long long t = x % y; x = y; y = t; }
    return x; // x is non-negative here; gcd(0,0)=0
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    // Sparse table over |a[i]| with the gcd operation (idempotent, so overlap is fine).
    int LOG = 1;
    while ((1 << LOG) < max(n, 1)) LOG++;
    LOG++; // headroom
    vector<vector<long long>> sp(LOG, vector<long long>(max(n, 1), 0));
    for (int i = 0; i < n; i++) sp[0][i] = llabs(a[i]); // strip sign: d | a[i] iff d | |a[i]|
    for (int k = 1; k < LOG; k++) {
        int len = 1 << k;
        for (int i = 0; i + len <= n; i++) {
            sp[k][i] = g2(sp[k - 1][i], sp[k - 1][i + (len >> 1)]);
        }
    }

    // For a query [l, r] (1-indexed, inclusive) return gcd(|a_l|,...,|a_r|), with all-zero -> 0.
    string out;
    out.reserve(q * 12);
    for (int Q = 0; Q < q; Q++) {
        int l, r;
        cin >> l >> r;
        l--; r--; // to 0-indexed
        int len = r - l + 1;
        int k = 31 - __builtin_clz(len);
        long long ans = g2(sp[k][l], sp[k][r - (1 << k) + 1]);
        out += to_string(ans);
        out += '\n';
    }
    cout << out;
    return 0;
}
```
