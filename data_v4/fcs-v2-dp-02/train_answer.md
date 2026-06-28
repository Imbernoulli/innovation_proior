**Problem.** Given `n` items, each a `B`-bit mask, and `q` query masks, report for each query `m`
how many items `x` are **supersets** of `m`, i.e. satisfy `x & m == m`. Read `B n q`, then the `n`
item masks, then the `q` query masks from stdin; print one count per query. Limits: `B <= 20`,
`n, q <= 10^6`.

**Why the obvious approaches are too slow.** Scanning all items per query is `O(n q) = 10^{12}` —
off by ~1000x from a 2-second budget. The next idea — precompute, for every mask `m`, the superset
count `S[m]`, then answer each query by an `O(1)` array lookup — is correct, but the naive way to
build `S[]` (for each mask, enumerate its supersets and sum the multiplicities) costs `O(3^B)`
(`sum over masks of 2^{zeros(mask)} = 3^B`), about `3.5*10^9` at `B = 20`. That `3^B` factor is the
wall.

**Key idea — superset-direction Sum-Over-Subsets (zeta transform).** Build `f[t] = cnt[t]` = number
of items equal to mask `t`. Then run SOS DP in the *superset* direction: process bits one at a time,
and for each bit `b`, fold the value of every "bit-`b`-set" state down into its "bit-`b`-clear"
partner:

```
for b in 0..B-1:
    for every mask with bit b clear:
        f[mask] += f[mask | (1<<b)]
```

After all `B` bits, `f[m]` equals the sum of `cnt[t]` over all supersets `t ⊇ m` — exactly `S[m]`.
This reuses each bit's work across masks, collapsing the `O(3^B)` superset enumeration into
`O(B * 2^B) ≈ 2*10^7`. Each query is then a single `f[m]` read.

**Pitfalls to get right.**
1. *Direction and in-place hazard.* Update **only** masks with bit `b` clear, and read the partner
   `f[mask | (1<<b)]` (which has bit `b` set and is never written in that pass). A naive
   `f[mask] += f[mask ^ (1<<b)]` over *all* masks both updates bit-set states (illegal — a superset
   can't drop a set bit) and reads cells already mutated in the same pass (read-after-write), giving
   garbage. A `B=2` trace exposes it immediately.
2. *Subset vs superset.* The standard SOS folds set→clear for *superset* sums; folding clear→set
   would compute *subset* sums. Get the partner index right (`mask | bit`, not `mask & ~bit`).
3. *I/O scale.* With `q = 10^6` outputs and `n + q = 2*10^6` integers to read, use
   `sync_with_stdio(false)` and buffer the output into one string.

**Edge cases (all handled by construction).** `B = 0` → only mask `0`, bit loop empty, every query
returns `n`. `n = 0` → `f` all zero, every query `0`. Query `m = 0` (empty mask) → `S[0] = n`, since
the empty mask's supersets are everything. Query `m = 2^B - 1` (full mask) → `S[full] = cnt[full]`,
the only superset of the full mask being itself (and `f[full]` is never written, since it has no
clear bit). Duplicate item masks accumulate in `cnt[t]` and are summed correctly.

**Complexity.** `O(2^B)` to build `f`, `O(B * 2^B)` for the transform, `O(q)` to answer — overall
`O(B * 2^B + n + q)` time, `O(2^B)` space. At `B = 20`, `n = q = 10^6` this runs in ~0.3 s and
~16 MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int B, n, q;
    if (!(cin >> B >> n >> q)) return 0;

    // f[mask] starts as the multiplicity of each item mask, then becomes
    // the number of items whose mask is a SUPERSET of `mask` (item & mask == mask).
    const int SZ = 1 << B;
    vector<long long> f(SZ, 0);

    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        f[x] += 1;
    }

    // Sum-Over-Subsets (zeta transform) in the SUPERSET direction.
    // For each bit b, fold the value of the state with bit b set DOWN into the
    // state with bit b cleared. After processing every bit, f[mask] equals the
    // sum of the original f over all masks t with (t & mask) == mask, i.e. all
    // supersets of `mask`. O(B * 2^B).
    for (int b = 0; b < B; b++) {
        int bit = 1 << b;
        for (int mask = 0; mask < SZ; mask++) {
            if ((mask & bit) == 0) {
                f[mask] += f[mask | bit];
            }
        }
    }

    string out;
    out.reserve((size_t)q * 7);
    for (int i = 0; i < q; i++) {
        int m;
        cin >> m;
        out += to_string(f[m]);
        out += '\n';
    }
    cout << out;
    return 0;
}
```
