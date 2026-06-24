**Problem.** A scanner logs `q` chronological events over badges with distinct integer ids: `+ b`
(badge `b` enters, was not inside), `- b` (badge `b` leaves, was inside), and `? lo hi` (how many
badges currently inside have id in the **inclusive** range `[lo, hi]`, i.e. `lo <= b <= hi`). For each
`?`, print the count. Ids and bounds are up to `10^9`, but at most `q` distinct ids appear in `+`/`-`
events; `lo > hi` is allowed and denotes an empty range. Read from stdin, write one line per query.

**Why the naive rescan is too slow.** Keeping a hash set and counting members of `[lo, hi]` per query
is `O(size)` each, hence `O(q^2)` worst case — about `4*10^{10}` at `q = 2*10^5`. It TLEs; it is only
useful as a correctness oracle.

**Key idea — Fenwick over compressed values.** Ids reach `10^9`, so a Fenwick cannot be indexed by id
directly. Read all events first, coordinate-compress the ids that appear in `+`/`-` into `1..N`
(`N <= q`), and keep a Fenwick tree `fen` of present-counts. `+ b` does `fadd(pos(b), +1)`, `- b` does
`fadd(pos(b), -1)`. With `prefixPresent(p) = fsum(p)` = present badges among compressed positions
`1..p`, an inclusive window `[lo, hi]` is "ids `<= hi`" minus "ids `< lo`":

```
answer(lo, hi) = fsum(cntLE(hi)) - fsum(cntLT(lo)),   valid when lo <= hi
```

where `cntLE(v) = upper_bound(vals, v) - begin` (values `<= v`) and `cntLT(v) = lower_bound(vals, v)
- begin` (values `< v`). Each event is `O(log N)`; total `O(q log q)`.

**Pitfalls.**
1. *Inclusive lower bound is `<`, not `<=`.* Both endpoints are inclusive, yet the upper side uses
   `<= hi` (`upper_bound`) while the lower side must use `< lo` (`lower_bound`). Subtracting
   `cntLE(lo)` instead of `cntLT(lo)` removes the badge sitting exactly on `lo`. A trace of `+ 10;
   ? 10 50` returning `0` instead of `1` exposes exactly this — the lower endpoint gets dropped.
2. *Inverted range `lo > hi`.* The prefix-difference identity only holds when `cntLT(lo) <=
   cntLE(hi)`, i.e. `lo <= hi`; otherwise it returns a negative value. A trace of `? 60 5` returning
   `-2` shows the need for an explicit `if (lo > hi) print 0` guard.
3. *Bounds need not be ids.* `lo`/`hi` are arbitrary, so look them up with `upper_bound`/`lower_bound`
   over `vals`, never by direct index. Compress only `+`/`-` ids, not query bounds.

**Edge cases.** No insertions at all (`N = 0`: Fenwick loops never iterate, every query is `0`);
queries before any insert (live counts are zero, but `vals` already holds all ids since we compress
over the whole log); single-point window `? v v` (flows through the normal path and correctly keeps a
badge on `v`); bounds outside or between ids; re-entry `+ b; - b; + b` (same slot toggles in `{0,1}`).
Counts fit in 32 bits (`q < 2^{31}`); prefix sums use `long long` for safety.

**Complexity.** `O(q log q)` time (sort/compress plus `O(log N)` per event), `O(q)` memory. Runs in
about 50 ms on `q = 2*10^5` with ids spread over `[1, 10^9]`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int N;                 // size of the Fenwick tree (number of distinct badge values)
vector<int> fen;       // 1-indexed Fenwick storing present-counts over compressed badges

void fadd(int i, int delta) {       // i is 1-indexed
    for (; i <= N; i += i & (-i)) fen[i] += delta;
}
long long fsum(int i) {             // sum over compressed positions [1..i], i may be 0
    long long s = 0;
    for (; i > 0; i -= i & (-i)) s += fen[i];
    return s;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;

    struct Op { char type; long long x, y; };
    vector<Op> ops(q);
    vector<long long> vals;                 // badge values that actually get inserted/removed
    for (int i = 0; i < q; i++) {
        string t;
        cin >> t;
        if (t == "+" || t == "-") {
            long long b; cin >> b;
            ops[i] = {t[0], b, 0};
            vals.push_back(b);
        } else {                            // "?"
            long long lo, hi; cin >> lo >> hi;
            ops[i] = {'?', lo, hi};
        }
    }

    // Coordinate-compress only the badge values that appear in + / - events.
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    N = (int)vals.size();
    fen.assign(N + 1, 0);

    // 1-indexed compressed position of an existing badge value b (b must be in vals).
    auto posOf = [&](long long b) -> int {
        int idx = (int)(lower_bound(vals.begin(), vals.end(), b) - vals.begin());
        return idx + 1;                     // shift to 1-indexed
    };
    // Number of compressed values that are <= v (v is an arbitrary query bound).
    auto cntLE = [&](long long v) -> int {
        return (int)(upper_bound(vals.begin(), vals.end(), v) - vals.begin());
    };
    // Number of compressed values that are < v.
    auto cntLT = [&](long long v) -> int {
        return (int)(lower_bound(vals.begin(), vals.end(), v) - vals.begin());
    };

    string out;
    for (auto &op : ops) {
        if (op.type == '+') {
            fadd(posOf(op.x), +1);
        } else if (op.type == '-') {
            fadd(posOf(op.x), -1);
        } else {
            long long lo = op.x, hi = op.y;
            if (lo > hi) { out += "0\n"; continue; }
            // present badges with lo <= badge <= hi (inclusive both ends).
            // In compressed space: positions whose value is <= hi, minus those whose value is < lo.
            int rHi = cntLE(hi);            // compressed index of last value <= hi
            int rLoExclusive = cntLT(lo);   // count of values strictly < lo
            long long ans = fsum(rHi) - fsum(rLoExclusive);
            out += to_string(ans);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
```
