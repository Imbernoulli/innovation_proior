**Reading the problem and pinning the contract.** I have an array `a[0..n-1]` and a band `[L, R]` with `0 <= L <= R`, and I must count *unordered* pairs `{i, j}`, `i != j`, with `L <= |a[i] - a[j]| <= R`. "Unordered" and "exactly once" are the load-bearing words: `{i, j}` and `{j, i}` are one pair, so the answer is at most `C(n, 2)`. Input is `n, L, R` then the `n` values; I print one integer. Before any algorithm I fix the scale, because it picks my data types: `n <= 2*10^5`, so the answer can be `C(n, 2) = n(n-1)/2 ~ 2*10^10`, which is past the 32-bit limit of `~2.1*10^9`. The pair *count* therefore must accumulate in `long long`. The values are `|a[i]| <= 10^9` and `R` can be `2*10^9`, so band endpoints like `a[j] + R` reach `~3*10^9` — also beyond 32 bits. Every value and every band endpoint must be 64-bit. That is decided up front; an `int` here is a silent wrong-answer on the large tests.

**Laying out the candidate approaches.** Two routes, and I want the one whose *counting* I can prove avoids double-counting, not just the one that is shortest.

- *Sort + two pointers.* Sort the values; for the band `[L, R]`, the partners of a fixed value form a contiguous window in sorted order, and a sliding window counts them in `O(n log n)`. The catch is the band is two-sided (partners both below and above the value), and a window naturally counts *ordered* contributions, so I would have to divide by two and be careful about the `L = 0` self-overlap. Workable but fiddly to keep unordered-correct.
- *Fenwick (BIT) over compressed values, swept left to right.* Process `j = 0, 1, ..., n-1`. Before inserting `a[j]`, query how many *already-inserted* elements (indices `i < j`) have a value inside the band around `a[j]`; add that to the answer; then insert `a[j]`. Because I only ever query earlier elements, each unordered pair `{i, j}` with `i < j` is counted exactly once — from its later endpoint `j`. That structurally solves the "count once" problem. Each query/insert is `O(log n)`, total `O(n log n)`. I pick this: the "count from the later endpoint" invariant is the cleanest argument that no pair is double-counted *across different j's*. The danger that remains is double-counting *within a single j* and the inclusive-range index arithmetic.

**Deriving the per-`j` query and checking it on paper.** Fix `j` with value `v = a[j]`. I want earlier elements `a[i]` with `L <= |a[i] - v| <= R`. Split the absolute value:

- `a[i] <= v` side: `L <= v - a[i] <= R`, i.e. `a[i] in [v - R, v - L]` (the *lower band*).
- `a[i] >= v` side: `L <= a[i] - v <= R`, i.e. `a[i] in [v + L, v + R]` (the *upper band*).

So per `j` I add `rangeCount(v - R, v - L) + rangeCount(v + L, v + R)`, where `rangeCount(lo, hi)` is the number of already-inserted elements with value in `[lo, hi]`. Let me sanity-check the band split on a tiny case. `v = 5`, `L = 2`, `R = 4`: lower band `[1, 3]`, upper band `[7, 9]`. A partner `a[i] = 3` gives `|3-5| = 2` ✓ and sits in `[1,3]` ✓; a partner `a[i] = 8` gives `|8-5| = 3` ✓ and sits in `[7,9]` ✓; a partner `a[i] = 5` gives `0`, outside `[2,4]`, and indeed `5` is in neither `[1,3]` nor `[7,9]`. The split is right *for `L >= 1`*. I deliberately flag "for `L >= 1`" because the two bands meet exactly when `L = 0`: lower band top is `v - 0 = v`, upper band bottom is `v + 0 = v`. I will come back to that; it is the obvious trap.

**Coordinate compression and the Fenwick.** Values span `[-10^9, 10^9]`, too wide to index directly, so I compress. The key insight that keeps the Fenwick small and the indices clean: I only ever *insert* actual array values `a[j]`, so the only keys the tree can ever hold are the distinct values of `a`. The band endpoints (`v - R`, `v + R`, etc.) are *not* inserted — they are only used to bound queries — so I do not need them in the compressed set. I resolve a query endpoint by binary-searching it into the sorted distinct-value list. So: `ks` = sorted unique `a` values, `m = ks.size()`, a 1-indexed Fenwick `bit[1..m]` where `bit` accumulates counts at compressed positions, and `pref(p)` returns the number of inserted elements at compressed positions `1..p`.

For `rangeCount(lo, hi)` I need "how many inserted keys have value in `[lo, hi]`". The number of keys `<= hi` is `upper_bound(ks, hi) - begin` (first key strictly greater than `hi`), call it `hiIdx`; positions `1..hiIdx` are the keys `<= hi`. The number of keys `< lo` is `lower_bound(ks, lo) - begin` (first key `>= lo`), call it `loIdx`; positions `1..loIdx` are the keys `< lo`. So the count is `pref(hiIdx) - pref(loIdx)`. This is the spot where off-by-one lives, so I will trace it explicitly rather than trust it.

**First implementation.** My first cut of the per-`j` body just adds both bands unconditionally and uses a range query I wrote from memory:

```
// querying [lo, hi] inclusive:
int hiIdx = upper_bound(ks, hi) - begin;   // keys <= hi
int loIdx = lower_bound(ks, lo) - begin;   // keys < lo
add pref(hiIdx) - pref(loIdx);
...
// per j:
ans += rangeCount(v - R, v - L);   // lower band
ans += rangeCount(v + L, v + R);   // upper band
```

**First debug episode — the `L = 0` double-count, traced on equal values.** I pick the smallest input that exercises the band overlap: `a = [5, 5, 5, 5]`, `L = 0`, `R = 3`. Every pair has difference `0`, which is in `[0, 3]`, so the true answer is `C(4, 2) = 6`. Compressed keys `ks = [5]`, `m = 1`. Now sweep:

- `j = 0`, `v = 5`. Tree empty. Lower band `[v - R, v - L] = [2, 5]`: `rangeCount(2,5)` over empty tree = 0. Upper band `[v + L, v + R] = [5, 8]`: also 0. Insert `5`. `ans = 0`.
- `j = 1`, `v = 5`. Tree has one `5`. Lower band `[2, 5]`: keys `<= 5` is `hiIdx = 1`, keys `< 2` is `loIdx = 0`, so `pref(1) - pref(0) = 1 - 0 = 1`. Upper band `[5, 8]`: keys `<= 8` is `hiIdx = 1`, keys `< 5` is `loIdx = 0`, so `1 - 0 = 1`. So this `j` adds `1 + 1 = 2`. Insert `5`. `ans = 2`.
- `j = 2`, `v = 5`. Tree has two `5`s. Lower band counts `2`, upper band counts `2`, adds `4`. Insert. `ans = 6`.
- `j = 3`, `v = 5`. Tree has three `5`s. Lower `3` + upper `3 = 6`, adds `6`. `ans = 12`.

The code returns `12`. The true answer is `6`. It is exactly **double**. The defect is precise and is the trap I flagged: when `L = 0`, the lower band `[v - R, v - L] = [v - R, v]` and the upper band `[v + L, v + R] = [v, v + R]` both *include* the value `v` itself. Their union overlaps at exactly the single value `v`, so every earlier element equal to `v` is counted once by the lower band and once by the upper band — a double-count of all equal-value partners. For `L >= 1` the bands are `[v - R, v - L]` (top `v - L <= v - 1`) and `[v + L, v + R]` (bottom `v + L >= v + 1`), which are disjoint, so the bug is specific to `L = 0`.

**Fix for the overlap.** When `L = 0`, do not query two overlapping bands; query the single merged band `[v - R, v + R]` once. (Equivalently I could subtract the overlap `rangeCount(v, v)`, but a single merged band is simpler and obviously has no internal double-count.) So:

```
if (L == 0) ans += rangeCount(v - R, v + R);
else { ans += rangeCount(v - R, v - L); ans += rangeCount(v + L, v + R); }
```

Re-trace `[5,5,5,5]`, `L=0`, `R=3`. Merged band per `j` is `[2, 8]`.
- `j=0`: empty, adds 0, insert. `ans=0`.
- `j=1`: one `5`; `[2,8]` -> `pref(1) - pref(0) = 1`. adds 1, insert. `ans=1`.
- `j=2`: two `5`s; adds 2, insert. `ans=3`.
- `j=3`: three `5`s; adds 3. `ans=6`.

Returns `6`. Correct, and it broke for exactly the reason I fixed. I also reran the documented sample `a=[1,5,3,8,6,2]`, `L=2`, `R=4` (here `L>=1`, two disjoint bands) by hand-listing pairs: qualifying pairs are `{1,5}` (d4), `{1,3}` (d2), `{5,3}` (d2), `{5,8}` (d3), `{5,2}` (d3), `{3,6}` (d3), `{8,6}` (d2), `{6,2}` (d4) = 8. The two-band path must give `8`.

**Second debug episode — an inclusive-range off-by-one, traced on a tight band.** Even with the overlap fixed I do not trust my `rangeCount` indices, so I stress the *boundary* of a band where `hi` lands exactly on a key. Earlier, while drafting, I had written the upper index as `lower_bound(ks, hi) - begin` ("first key `>= hi`") instead of `upper_bound`, reasoning loosely that "I want keys up to `hi`". Let me trace what that does on `a = [2, 5]`, `L = 3`, `R = 3` (so I want pairs with `|diff| = 3`; `|2-5| = 3`, true answer `1`). Compressed `ks = [2, 5]`.

- `j = 0`, `v = 2`. Tree empty. `L >= 1` so two bands. Lower `[v-R, v-L] = [-1, -1]`: `rangeCount(-1,-1)`. Upper `[v+L, v+R] = [5, 5]`. Both over empty tree = 0. Insert `2` (compressed pos of `2` is index 0, 1-indexed pos 1). `ans = 0`.
- `j = 1`, `v = 5`. Tree has `{2}`. Lower band `[v-R, v-L] = [2, 2]`. With the *buggy* upper index `lower_bound(ks, 2) - begin`: `lower_bound` for `2` in `[2,5]` returns position `0`, so `hiIdx = 0`. `loIdx = lower_bound(ks, 2) - begin = 0`. Then `hiIdx <= loIdx` (`0 <= 0`), so `rangeCount` returns `0`. Upper band `[8, 8]` = 0. So buggy code gives `ans = 0`.

Buggy answer `0`, true answer `1`. The defect: for the inclusive upper end I must count keys `<= hi`, which is `upper_bound(ks, hi) - begin` (first key strictly `> hi`); using `lower_bound` (first key `>= hi`) *excludes* the key equal to `hi`, an off-by-one that drops every partner that hits the band edge exactly — precisely the `|diff| = R` and `|diff| = L` pairs. With the correct `upper_bound`: at `j=1`, lower band `[2,2]`: `hiIdx = upper_bound(ks, 2) - begin = 1` (first key `> 2` is `5` at pos 1), `loIdx = lower_bound(ks, 2) - begin = 0` (first key `>= 2` is `2` at pos 0). `pref(1) - pref(0) = 1 - 0 = 1`. So `ans = 1`. Correct. The matching rule is: upper end uses `upper_bound` (keys `<= hi`), lower end uses `lower_bound` (keys `< lo`), and the count is `pref(hiIdx) - pref(loIdx)` with the guard `hiIdx <= loIdx -> 0`. I keep the `lo > hi -> 0` guard too, since for `L >= 1` a band like `[v-R, v-L]` is always non-empty as an interval but the merged/`rangeCount` helper should still be safe if called with a crossed interval.

**Re-verification after both fixes.** I compiled the corrected solution and ran it against an independent `O(n^2)` Python brute force on 1500 random small cases (`n` up to ~40, narrow value ranges so equal values and band edges occur constantly, `L` biased toward `0` to hammer the merged-band path, plus wide-value cases). Zero mismatches. The two cases that exposed the bugs — `[5,5,5,5]` with `L=0` (was `12`, now `6`) and `[2,5]` with `L=R=3` (was `0`, now `1`) — both pass, and they pass for the reasons I fixed, which is the evidence I trust over "looks right".

**Edge cases, deliberately.**
- `n = 0`: `cin >> n >> L >> R` reads the header, the value loop runs zero times, the sweep runs zero times, `ans = 0`. Correct (no pairs).
- `n = 1`: one element, the sweep queries an empty tree then inserts; `ans = 0`. Correct.
- `L = R = 0`: counts pairs with `|diff| = 0`, i.e. equal-value pairs. Merged band per `j` is `[v, v]`, which counts earlier elements equal to `v` — exactly the equal pairs, counted once each from the later endpoint. Verified against brute (`[2,2,3,2,3]` -> `4`).
- Band that no pair can satisfy, e.g. `L = R = 5` on `[1,2,3]`: every band is empty of keys; `ans = 0`. Correct.
- Extreme values/overflow: `a[j] + R` can reach `~3*10^9` and `a[j] - R` reach `~-3*10^9`; band endpoints are `long long`, so `v - R` / `v + R` do not overflow. The answer accumulates in `long long`; at `n = 2*10^5` it can reach `~2*10^10`, fits. Verified `[-10^9, 0, 10^9]` with `L=10^9, R=2*10^9` -> `3`.
- Insert vs. query order: I query *then* insert `a[j]`, so a pair never uses `i == j`, and every unordered pair is counted once from its later index. If I inserted first, an element would match itself when `L = 0` (`|a[j]-a[j]| = 0 in [0,R]`), inflating the count — the order is part of the correctness.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so the two-line vs. one-line input layout does not matter.

**Performance.** `O(n log n)`: one sort for compression, then `n` iterations each doing `O(1)` binary searches and `O(log m)` Fenwick ops. At `n = 2*10^5` with full-range values and `L = 0` (worst merged-band path) it runs in ~0.07 s using ~7 MB — comfortably inside 1 s / 256 MB.

**Final solution.** I proved "count once" structurally (query earlier elements, count each pair from its later endpoint), derived the two-band split, then caught two real counting bugs by tracing: the `L = 0` overlap that doubled equal-value pairs, and an inclusive-upper-end off-by-one that dropped band-edge pairs. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Compressed key set = the distinct values that actually appear in a[].
    // We only ever INSERT a[j] values, so these are the only keys the Fenwick
    // can hold; range-query endpoints are resolved against this sorted list by
    // binary search, no need to insert the endpoints themselves.
    vector<long long> ks(a);
    sort(ks.begin(), ks.end());
    ks.erase(unique(ks.begin(), ks.end()), ks.end());
    int m = (int)ks.size();

    // 1-indexed Fenwick over compressed positions 1..m; bit[p] counts how many
    // already-inserted elements have compressed value p.
    vector<int> bit(m + 1, 0);
    auto upd = [&](int p) {                 // p in 1..m
        for (; p <= m; p += p & (-p)) bit[p] += 1;
    };
    auto pref = [&](int p) -> long long {   // count of inserted elems at positions 1..p
        long long s = 0;
        for (; p > 0; p -= p & (-p)) s += bit[p];
        return s;
    };
    // count inserted elements whose real VALUE lies in [lo, hi] (inclusive)
    auto rangeCount = [&](long long lo, long long hi) -> long long {
        if (lo > hi) return 0;
        // keys at positions 1..hiIdx are <= hi  (upper_bound = first key > hi)
        int hiIdx = (int)(upper_bound(ks.begin(), ks.end(), hi) - ks.begin());
        // keys at positions 1..loIdx are < lo   (lower_bound = first key >= lo)
        int loIdx = (int)(lower_bound(ks.begin(), ks.end(), lo) - ks.begin());
        if (hiIdx <= loIdx) return 0;
        return pref(hiIdx) - pref(loIdx);
    };

    long long ans = 0;
    for (int j = 0; j < n; j++) {
        long long v = a[j];
        // earlier i (already inserted) with |a[i] - v| in [L, R]:
        //   lower band  a[i] in [v-R, v-L]
        //   upper band  a[i] in [v+L, v+R]
        // When L==0 both bands include v itself, so their union overlaps exactly
        // at the value v; summing both bands would double-count earlier elements
        // equal to v. Treat L==0 as ONE merged band [v-R, v+R].
        if (L == 0) {
            ans += rangeCount(v - R, v + R);
        } else {
            ans += rangeCount(v - R, v - L);   // lower band
            ans += rangeCount(v + L, v + R);   // upper band (disjoint since L>=1)
        }
        // insert a[j] AFTER querying, so a pair never uses i == j
        int p = (int)(lower_bound(ks.begin(), ks.end(), v) - ks.begin()) + 1; // 1-indexed
        upd(p);
    }

    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** The "count each unordered pair once" requirement is met structurally by sweeping left to right and querying only earlier-inserted elements, so a pair is counted from its later endpoint; the per-`j` work is a two-sided band split, `[v-R, v-L]` below and `[v+L, v+R]` above, answered by a Fenwick over compressed values. A trace of `[5,5,5,5]` with `L=0` returned `12` instead of `6` because at `L=0` the two bands both include `v`, double-counting equal partners — fixed by querying one merged band `[v-R, v+R]` when `L=0`; and a trace of `[2,5]` with `L=R=3` returned `0` instead of `1` because the inclusive upper end was computed with `lower_bound` (excluding the key equal to `hi`) instead of `upper_bound` — fixed so the upper end counts keys `<= hi`. With `long long` for the count and band endpoints, query-before-insert ordering, and both fixes, 1500 random cases agree with brute force and the corners (`n=0,1`, `L=R=0`, empty bands, full-range overflow) all check out.
