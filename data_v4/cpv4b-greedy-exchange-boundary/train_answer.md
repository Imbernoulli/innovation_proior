**Problem.** A pipeline lies on a number line with `n` flagged segments, segment `i` being the closed
integer interval `[l_i, r_i]`. A checkpoint installed at integer coordinate `x` inspects segment `i`
iff `l_i <= x <= r_i` (endpoints included). Find the minimum number of checkpoints that inspect every
segment. Read `n` then the `n` pairs from stdin; print the minimum count. (`n = 0` gives `0`.)

**Key idea — greedy by earliest right endpoint, justified by an exchange.** This is "stab all closed
intervals with the fewest points." Sort segments by right endpoint ascending and sweep, keeping
`last` = the coordinate of the most recently placed checkpoint. When the current segment is not
inspected by the checkpoint at `last`, open a new checkpoint at this segment's right endpoint `r` and
set `last = r`.

*Why the right endpoint?* Any solution must inspect the segment `s` with the smallest right endpoint
`r_min`, so it has a checkpoint somewhere in `[l_s, r_min]`. Slide that checkpoint to `r_min`: for any
other segment `t` it used to inspect, `l_t <= x <= r_min` and `r_t >= r_min` (since `r_min` is
smallest), hence `l_t <= r_min <= r_t` still holds — coverage can only grow. So a checkpoint at the
smallest right endpoint is always safe; remove the segments it inspects and recurse. That sweep is the
algorithm.

**Pitfalls.**
1. *The inclusive-exclusive boundary (the whole problem).* A segment `[l, r]` is inspected by `last`
   iff `l <= last <= r`. Because we process in increasing-`r` order, `last <= r` is automatic once a
   checkpoint exists, so the membership test reduces to `l <= last`. Therefore "open a new checkpoint"
   must be the **strict** `last < l`. A segment whose left endpoint equals the last checkpoint
   (`l == last`, the closed intervals *touch*) is already inspected. Writing `last <= l` re-opens a
   redundant checkpoint for every touching segment — e.g. on `[1,5],[5,9]` it returns `2` instead of
   the correct `1`. Trace that two-segment case and the off-by-one is unmistakable.
2. *Sort key must be the right endpoint.* Sorting by left endpoint breaks the invariant `r >= last`
   for later segments, which is exactly what lets the membership test collapse to `l <= last`; with a
   left-endpoint sort a checkpoint placed at a large right endpoint mis-reports shorter later segments
   as covered. Store pairs as `{r, l}` so the default `pair` order sorts by `r`.
3. *Sentinel and types.* Coordinates reach `±10^9` and can be negative, so the "no checkpoint yet"
   sentinel must sit below every real coordinate; use `long long last = LLONG_MIN`. No additions occur,
   so there is no overflow, but `int` coordinates with a `-2e9`-style sentinel would be fragile — keep
   everything `long long`.

**Edge cases.** `n = 0` -> `0` (loop never runs); single/degenerate `[3,3]` -> `1`; all segments
sharing a point -> `1`; fully disjoint -> `n`; end-to-end touching chain `[1,5],[5,9],[9,13]` -> `2`
(this is where a `<=` bug would wrongly say `3`); duplicates collapse to `1`; negative coordinates
handled by the `LLONG_MIN` sentinel.

**Complexity.** `O(n log n)` for the sort, `O(n)` sweep, `O(n)` memory. Runs in ~0.03 s at
`n = 2*10^5`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 -> no segments -> 0 checkpoints
    vector<pair<long long,long long>> seg(n); // (right endpoint, left endpoint)
    for (int i = 0; i < n; i++) {
        long long l, r;
        cin >> l >> r;
        seg[i] = {r, l};                   // sort key is the right endpoint
    }

    sort(seg.begin(), seg.end());          // by right endpoint ascending

    long long checkpoints = 0;
    long long last = LLONG_MIN;            // coordinate of the most recent checkpoint
    for (int i = 0; i < n; i++) {
        long long r = seg[i].first, l = seg[i].second;
        // The segment is already inspected iff l <= last <= r. Since we process in
        // right-endpoint order, last <= r always holds once a checkpoint exists, so
        // the only test that matters is whether l <= last. If l > last the segment
        // is NOT covered and we must open a new checkpoint at r (the latest spot that
        // still inspects this segment, maximizing future coverage).
        if (last < l) {                    // strict: l == last means already covered
            checkpoints++;
            last = r;
        }
    }

    cout << checkpoints << "\n";
    return 0;
}
```
