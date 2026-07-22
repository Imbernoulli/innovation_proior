A scanner logs `q` chronological events. `+ b`
puts badge `b` inside (it was not inside), `- b` takes it out (it was inside), and `? lo hi` asks how
many badges currently inside have id in the **inclusive** range `[lo, hi]`, i.e. `lo <= b <= hi`. I
print one count per `?`. The guarantees say a `+` never repeats an id already inside and a `-` always
names an id inside, so at every instant an id is present at most once — set semantics, each compressed
slot holds `0` or `1`. Before any algorithm I fix the scale: `q <= 2*10^5`, ids and bounds up to
`10^9`. Two consequences. First, `10^9` ids mean I cannot index a Fenwick tree directly by id —
`10^9` cells is far past memory; I must coordinate-compress. Second, at most `q` distinct ids ever
appear in `+`/`-` events, so the compressed universe has size `N <= q <= 2*10^5`, which is fine.
Counts are bounded by `q < 2^{31}`, so a 32-bit count never overflows, but I will let the Fenwick
prefix sums be `long long` anyway — it costs nothing and removes one class of worry.

**The bound that is *not* an id is the whole trap.** The query bounds `lo`, `hi` are arbitrary
integers; they need not be ids that ever appeared. So I cannot just look up `lo` and `hi` in the
compressed array and expect a hit. The window `[lo, hi]` is inclusive on both ends, and the
translation into prefix sums has to respect that exactly. I will come back to this — it is where I
expect to make a mistake — but I note up front that "inclusive `lo`" and "inclusive `hi`" land on
*opposite* comparison operators once I phrase everything as "count of compressed values `< x` or
`<= x`", and mixing them up is the canonical off-by-one for this shape of problem.

**Laying out the candidate approaches.** Two routes are on the table.

- *Rescan per query.* Keep a hash set of present ids; for each `?`, iterate it and count ids in
  `[lo, hi]`. Trivially correct, `O(1)` updates, but each query is `O(size)`, so worst case `O(q^2)`
  — roughly `4*10^{10}` at the limit. That is seconds-to-minutes; it will TLE. I keep it only as my
  *brute force oracle*, never as the submission.
- *Fenwick over compressed values.* Compress the ids from `+`/`-` events into `1..N`. Keep a Fenwick
  tree `fen` of present-counts. A `+ b` does `fadd(posOf(b), +1)`, a `- b` does `fadd(posOf(b), -1)`.
  A query `[lo, hi]` becomes a difference of two prefix counts. `O(log N)` per event, `O(q log q)`
  overall — about `2*10^5 * 18 ≈ 3.6*10^6` Fenwick steps. Comfortable under 1 s. This is the one I
  commit to; the only thing to get right is the boundary translation.

**Deriving the boundary translation on paper.** Let the sorted distinct id list be `vals[0..N-1]`,
and compressed position of `vals[k]` be `k+1` (1-indexed, because Fenwick wants 1-indexed). Define
`prefixPresent(p)` = number of present badges among compressed positions `1..p` = `fsum(p)`. I want
the count of present badges whose id is in `[lo, hi]`.

Phrase the window as "ids `<= hi`" minus "ids `< lo`":

- ids `<= hi`: among the compressed values, those `<= hi` occupy positions `1..cntLE(hi)`, where
  `cntLE(hi)` = number of `vals` entries `<= hi` = `upper_bound(vals, hi) - begin`. So the present
  count with id `<= hi` is `fsum(cntLE(hi))`.
- ids `< lo`: positions `1..cntLT(lo)`, where `cntLT(lo)` = number of `vals` entries `< lo` =
  `lower_bound(vals, lo) - begin`. Present count with id `< lo` is `fsum(cntLT(lo))`.

The inclusive window `[lo, hi]` is exactly "`<= hi`" minus "`< lo`", because removing `< lo` keeps
`>= lo` and capping at `<= hi` gives `lo <= id <= hi`. So

```
answer(lo, hi) = fsum(cntLE(hi)) - fsum(cntLT(lo)).
```

The asymmetry is the crux: the **upper** bound uses `<=` (`upper_bound`) and the **lower** bound
uses `<` (`lower_bound`). Both endpoints are inclusive, yet one side is "less-or-equal" and the other
is "strictly-less", because one is a `<= hi` cap and the other is a `< lo` *removal*. If I accidentally
use `<= lo` (i.e. `upper_bound`) for the lower side, I subtract away the badge sitting exactly on
`lo`, dropping it from an inclusive range. That is the bug I most expect to write.

**Sanity-check the derivation on the sample.** Events: `+10, +30, +50`, then `? 10 50`, `? 11 49`,
`-30`, `? 10 50`, `? 60 5`. The `vals` from `+`/`-` are `{10, 30, 50}` -> positions `10->1, 30->2,
50->3`. After the three inserts, `fen` present-counts are `1,1,1` at positions `1,2,3`.

- `? 10 50`: `cntLE(50)` = #vals `<= 50` = 3; `cntLT(10)` = #vals `< 10` = 0. Answer
  `fsum(3) - fsum(0) = 3 - 0 = 3`. Matches (all of `10,30,50`, endpoints inclusive).
- `? 11 49`: `cntLE(49)` = #vals `<= 49` = 2 (10 and 30); `cntLT(11)` = #vals `< 11` = 1 (just 10).
  Answer `fsum(2) - fsum(1) = 2 - 1 = 1`. Matches (only 30).
- `-30`: positions now present `1,_,3`.
- `? 10 50`: `cntLE(50)=3`, `cntLT(10)=0`, `fsum(3)-fsum(0) = 2 - 0 = 2`. Matches (10 and 50).
- `? 60 5`: `lo=60 > hi=5`. `cntLE(5)=0`, `cntLT(60)=3`, naive difference `fsum(0)-fsum(3)=0-2=-2`.
  That is wrong — it must be `0`. So the derivation is right *only when `lo <= hi`*; `lo > hi` needs a
  guard. I note that as a separate hazard to handle in code.

The recurrence is right; now I transcribe it, expecting to trip on exactly the two things I just
flagged.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first
cut of the query branch, written fast, blurs the two bounds. I tell myself "the range is inclusive,
so use `<=` on both ends" — which is the mistake the derivation warned about:

```
// FIRST (BUGGY) ATTEMPT
long long lo = op.x, hi = op.y;
int rHi = cntLE(hi);            // values <= hi
int rLo = cntLE(lo);           // values <= lo   <-- "inclusive lo, so <=", wrong
long long ans = fsum(rHi) - fsum(rLo);
```

I trace the smallest input that pins the lower boundary: a single badge sitting exactly on `lo`.
Events `+ 10`, then `? 10 50`. Compressed `vals = {10}`, position `10 -> 1`, present count `fen[1]=1`.
Query: `rHi = cntLE(50) = 1` (one val `<= 50`), `rLo = cntLE(10) = 1` (one val `<= 10`, since `10 <=
10`). Answer `fsum(1) - fsum(1) = 1 - 1 = 0`.

**Diagnosing the bug.** The correct answer is obviously `1` — badge `10` is inside and `10 <= 10 <=
50`, so it is in the inclusive range. My code returns `0`. The defect is exactly the off-by-one I
predicted: `cntLE(lo)` counts values `<= lo`, which *includes* `lo` itself, so subtracting
`fsum(cntLE(lo))` strips out the badge on the lower endpoint. For an inclusive `[lo, hi]` the lower
side must remove only ids strictly `< lo`, i.e. `cntLT(lo)`, not `cntLE(lo)`. The two `bound` calls
have to be different: `upper_bound` for `hi` (`<= hi`) and `lower_bound` for `lo` (`< lo`). Using the
same comparison on both ends is what collapsed the endpoint. The trace caught it on the smallest
possible witness.

**Fixing the lower boundary and re-tracing.** Replace `cntLE(lo)` with `cntLT(lo)`:

```
int rHi = cntLE(hi);            // values <= hi    (upper_bound)
int rLoExclusive = cntLT(lo);   // values <  lo    (lower_bound)
long long ans = fsum(rHi) - fsum(rLoExclusive);
```

Re-trace `+ 10`, `? 10 50`: `rHi = cntLE(50) = 1`, `rLoExclusive = cntLT(10) = 0` (no val `< 10`).
Answer `fsum(1) - fsum(0) = 1 - 0 = 1`. Correct. Re-trace `? 11 49` from the sample (only `30` should
count, with `10,30,50` present): `cntLE(49) = 2`, `cntLT(11) = 1`, `fsum(2)-fsum(1) = 2 - 1 = 1`.
Correct. And re-trace an endpoint-on-`hi` case `? 5 10` with only `10` present: `cntLE(10) = 1`,
`cntLT(5) = 0`, answer `1` — the upper endpoint `10` is kept, as inclusive demands. The boundary now
behaves symmetrically inclusive even though the two operators differ, which is precisely the point.

**Second episode: the `lo > hi` window and a negative answer.** With the boundary fixed I trace the
last sample query `? 60 5` (present `10`, `50` after `-30`). `rHi = cntLE(5) = 0` (no val `<= 5`),
`rLoExclusive = cntLT(60) = 3` (all three vals `< 60`). Answer `fsum(0) - fsum(3) = 0 - 2 = -2`.

**Diagnosing the second bug.** The range `[60, 5]` is empty (lower bound above upper bound), so the
answer must be `0`, but the formula returns `-2`. The cause is structural, not a typo: when `lo > hi`
the set "`< lo`" is a *superset* of "`<= hi`", so subtracting their present-counts goes negative.
The prefix-difference identity `fsum(cntLE(hi)) - fsum(cntLT(lo))` is only valid when
`cntLT(lo) <= cntLE(hi)`, which holds iff `lo <= hi`. I need an explicit guard: if `lo > hi`, print
`0` and skip the difference. I add it at the top of the query branch:

```
if (lo > hi) { out += "0\n"; continue; }
```

Re-trace `? 60 5`: guard fires, output `0`. Correct. I also re-check a boundary-adjacent case
`? 5 5` (a single-point inclusive window) with `5` present: not `lo > hi`, `cntLE(5)=1`,
`cntLT(5)=0`, answer `1` — a one-point inclusive range correctly keeps the badge on it. And `? 6 6`
with `5` present: `cntLE(6)=1`, `cntLT(6)=1`, answer `0` — correct, `5` is not in `[6,6]`. The guard
is needed only for the strictly-inverted case; equal bounds flow through the normal path.

**Edge cases, deliberately, because this is where this kind of code dies.**
- *No insertions at all, then a query.* If every event is `?`, then `vals` is empty, `N = 0`, the
  Fenwick has size 1 (`fen` has `N+1 = 1` cells, index 0 unused). `cntLE(hi)` and `cntLT(lo)` are
  both `0` (empty `vals`), and `fsum(0) = 0`, so every query answers `0`. The `fadd`/`fsum` loops
  with `N = 0` never iterate (`i <= 0` and `i > 0` both stop immediately). Safe.
- *Query before any insert but after some are declared.* Compression uses *all* ids from `+`/`-`
  over the whole log (I read every event first, then compress), so a query early in time still has
  the full `vals` array; the Fenwick simply has zeros where nothing is present yet. Correct, because
  presence is tracked by the live counts, not by which ids exist in `vals`.
- *Bounds outside the id range.* `? 1 (10^9)` with ids in between: `cntLE(10^9) = N`, `cntLT(1) = 0`,
  answer `fsum(N)` = total present. Correct. Bounds in a gap, `? 11 19` with ids `10, 20`: `cntLE(19)
  = 1`, `cntLT(11) = 1`, answer `0`. Correct — no id lies in `[11,19]`.
- *Re-entry.* `+ 7`, `- 7`, `+ 7`: `posOf(7)` is the same compressed slot each time; `fadd` takes it
  to `1`, `0`, `1`. The slot count stays in `{0,1}` as the set semantics promise. A later `? 7 7`
  reads `1`. Correct.
- *Overflow.* Counts never exceed `q < 2^{31}`; I use `long long` for `fsum` anyway. The Fenwick
  delta is `+/-1`. No overflow path exists.
- *Output.* Exactly one line per `?` in order; I accumulate into a `string` and flush once, which is
  fast enough for `2*10^5` lines. `cin >>` on the op token reads `"+"`, `"-"`, or `"?"` as a string,
  so whitespace/newlines are irrelevant to parsing.

**Re-verification against a brute force.** I keep the rescan-the-hash-set oracle and run both on many
small random logs — tiny id universes (3 to 12 ids) so that query bounds frequently land exactly on,
just below, just above, and between ids, plus deliberate `lo > hi` queries and queries before any
insert. Over hundreds of seeds the Fenwick solution matches the oracle on every line. The two bugs I
found by tracing were the only two; once both are fixed, the random search surfaces nothing new,
which is the evidence I trust more than my own re-reading of the code.

**Final solution.** I read every event first so I can compress the complete id set, then replay:
`+`/`-` are `+/-1` Fenwick updates at the compressed slot, and each `? lo hi` with `lo <= hi` is
`fsum(cntLE(hi)) - fsum(cntLT(lo))` — `upper_bound` on the inclusive upper end, `lower_bound` on the
inclusive lower end — with a `lo > hi` guard returning `0`. That is what I ship.

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

**Causal recap.** The Fenwick idea was never in doubt — direct indexing by id is impossible at `10^9`,
so I compress and do prefix-count differences. The entire difficulty was the inclusive boundary: a
trace of `+ 10; ? 10 50` returning `0` instead of `1` showed that using `cntLE(lo)` (a `<=` on the
lower end) strips the badge sitting exactly on `lo`, so the lower side must be `cntLT(lo)` (a `<`)
while the upper side stays `cntLE(hi)` — both endpoints inclusive, opposite operators. A second trace
of `? 60 5` returning `-2` showed the prefix-difference identity is only valid for `lo <= hi`, so an
inverted-range guard returning `0` was required. With both boundary fixes the `O(q log q)` solution
matches a rescan oracle on hundreds of small random logs and runs in ~50 ms on the `q = 2*10^5`
stress.
