**Problem.** Maintain a growing collection of non-negative 60-bit integers under up to `q = 3*10^5`
online operations: `1 x` adds `x`; `2` reports the maximum XOR over any subset; `3 x` reports whether
`x` is obtainable as a subset-XOR (`YES`/`NO`); `4 k` reports the `k`-th smallest distinct subset-XOR
value (1-indexed, `0` always present from the empty subset), or `-1` if `k` exceeds the count. Read
from stdin, write one line per type-2/3/4 query.

**Why the obvious approaches fail.** Enumerating all `2^m` subset-XOR values per query is exactly the
definition and is trivially correct, but it is exponential — unusable past about `m = 20` numbers, far
below `3*10^5`. A "sort descending, XOR-in if it helps" greedy is not even reliably the max (ordering
by *value* is not ordering by *leading bit*; once two numbers share a top bit, value order can pick
the wrong representative), and it says nothing about membership or about the `k`-th-smallest order
statistic. Two of the three query families have no greedy story at all.

**Key idea — read subset-XOR as a `GF(2)` span and maintain a linear basis.** XOR is addition in
`GF(2)`, and a 60-bit integer is a vector in `GF(2)^60`. The set of obtainable subset-XOR values is
exactly the **linear span** of the inserted numbers. Maintain that span by **online Gaussian
elimination over `GF(2)`**: a `basis[b]` array where, when non-empty, `basis[b]` is a span vector
whose highest set bit is `b` (so at most one vector per bit, ≤ 60 total). Then:

- **Max XOR** — bit-greedy: `r = 0`; for `b` high→low, if `r xor basis[b] > r`, set `r ^= basis[b]`.
- **Membership** — reduce `x` against the basis (clear its highest set bits with the pivots); `x` is
  in the span iff it reduces to `0`. `0` always reduces to `0`, so `0` is always `YES`.
- **k-th smallest** — first bring the basis to **reduced** row-echelon form (each pivot bit appears in
  exactly one vector). Then the pivots, ordered by bit ascending, behave as place values: the
  0-indexed `k`-th smallest value is obtained by XORing in the `j`-th pivot iff bit `j` of `k` is set.
  This is the query that *forces* the reduced form, and a span of rank `r` has exactly `2^r` distinct
  values.

**Pitfalls.**
1. *Reduction must be complete.* For the order query, building "reduced" form by clearing a pivot bit
   out of only the *adjacent* higher vector is wrong — lower bits stay in higher pivots and destroy
   the place-value ordering. You must clear pivot bit `b` out of **every** higher basis vector. (Trace
   `[7,2,1]`: with incomplete reduction the 4th-smallest comes out `7` instead of the correct `4`.)
2. *Reduce lazily but correctly.* `maxXor` and `representable` do not need the reduced form, so reduce
   only when an order query needs it; cache a `reduced` flag, clear it whenever a *new pivot* is
   inserted (dependent inserts leave it alone), so repeated order queries cost reduction only once.
3. *64-bit, always `1ULL`.* Values are 60-bit; use `unsigned long long` and `1ULL << b`. A plain
   `1 << b` is undefined for `b >= 31` — a silent wrong-answer.

**Edge cases.** Empty span: max `0`, only `0` is `YES`, type-4 has `2^0 = 1` value so `k=1 -> 0`,
`k>=2 -> -1`. Dependent/duplicate inserts: rank and basis unchanged, distinct count stays `2^rank`.
Out-of-range `k`: since rank ≤ 60, `2^rank <= 2^60` fits and `1ULL << rank` never overflows; `k`
past `2^rank` gives `-1`. Top bit (bit 59) handled by looping `b` from 59 down with `1ULL` shifts.
`q = 0` / empty stdin produce no output.

**Complexity.** Insert / max / membership are `O(BITS) = O(60)` each; an order query is `O(BITS)`
amortized plus an `O(BITS^2)` reduction that happens at most once per pivot-changing boundary. Total
`O(q * BITS)` with a bounded number of `O(BITS^2)` reductions — comfortably under 0.1 s at
`q = 3*10^5`. Memory `O(BITS)`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

static const int BITS = 60;

struct XorBasis {
    // basis[b] holds a vector whose highest set bit is exactly b (or 0 if empty).
    unsigned long long basis[BITS];
    int rank;                 // number of independent vectors inserted
    bool reduced;             // is the basis currently in reduced row-echelon form?
    XorBasis() { memset(basis, 0, sizeof(basis)); rank = 0; reduced = true; }

    // Insert x; returns true iff x was independent (rank increased).
    bool insert(unsigned long long x) {
        for (int b = BITS - 1; b >= 0; --b) {
            if (!((x >> b) & 1ULL)) continue;
            if (!basis[b]) {
                basis[b] = x;
                ++rank;
                reduced = false;   // a fresh pivot may need cleaning before order queries
                return true;
            }
            x ^= basis[b];
        }
        return false;              // x reduced to 0: dependent
    }

    // Reduce to row-echelon form: every pivot bit appears in exactly one basis vector.
    void makeReduced() {
        if (reduced) return;
        for (int b = 0; b < BITS; ++b) {
            if (!basis[b]) continue;
            for (int c = b + 1; c < BITS; ++c) {
                if (basis[c] && ((basis[c] >> b) & 1ULL))
                    basis[c] ^= basis[b];
            }
        }
        reduced = true;
    }

    // Maximum XOR over the span (empty subset -> 0 included automatically).
    unsigned long long maxXor() const {
        unsigned long long r = 0;
        for (int b = BITS - 1; b >= 0; --b)
            if (basis[b] && (r ^ basis[b]) > r) r ^= basis[b];
        return r;
    }

    // Is x in the span (representable as a subset-XOR)?
    bool representable(unsigned long long x) const {
        for (int b = BITS - 1; b >= 0; --b) {
            if (!((x >> b) & 1ULL)) continue;
            if (!basis[b]) return false;
            x ^= basis[b];
        }
        return x == 0;
    }

    // k-th smallest distinct value (0-indexed: k=0 -> smallest = 0).
    // Requires reduced form; valid range 0 <= k < 2^rank.
    unsigned long long kthSmallest(unsigned long long k) {
        makeReduced();
        unsigned long long res = 0;
        int idx = 0;
        for (int b = 0; b < BITS; ++b) {
            if (!basis[b]) continue;
            if ((k >> idx) & 1ULL) res ^= basis[b];
            ++idx;
        }
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;

    XorBasis B;
    string out;
    out.reserve(1 << 16);

    for (int i = 0; i < q; ++i) {
        int type;
        cin >> type;
        if (type == 1) {
            unsigned long long x;
            cin >> x;
            B.insert(x);
            // no output for an add
        } else if (type == 2) {
            // maximum subset XOR
            out += to_string(B.maxXor());
            out += '\n';
        } else if (type == 3) {
            unsigned long long x;
            cin >> x;
            out += (B.representable(x) ? "YES\n" : "NO\n");
        } else { // type == 4: k-th smallest distinct subset-XOR value, 1-indexed
            unsigned long long k;
            cin >> k;
            // #distinct values = 2^rank; rank <= 60 since values are <= 60 bits.
            unsigned long long total = (1ULL << B.rank);
            // k is 1-indexed; valid iff 1 <= k <= 2^rank, else report -1.
            if (k >= 1 && k <= total) {
                out += to_string(B.kthSmallest(k - 1));
                out += '\n';
            } else {
                out += "-1\n";
            }
        }
    }

    cout << out;
    return 0;
}
```
