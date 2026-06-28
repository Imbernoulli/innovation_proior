**Problem.** Given `a[0..n-1]` (values may be negative) arranged on a **circle** — position `i` is adjacent to `i+1`, and position `n-1` is adjacent to position `0` — pick a subset with no two cyclically adjacent positions, maximizing the sum; the empty set is allowed, so the answer is at least `0`. Read `n` and the values from stdin, print the maximum sum.

**Why the tempting single-pass / greedy approaches are wrong.** The circle differs from the path by exactly one extra edge, the *wrap* between index `0` and index `n-1`. Two clever-looking moves fail:

- *Naive single-pass path DP over the whole circle.* It ignores the wrap and will take both endpoints. On the 6-cycle `[5, 1, 1, 5, 1, 5]` the path DP picks indices `0, 3, 5` for `15`, but `0` and `5` are wrap-adjacent, so that is illegal; the true circular optimum is indices `1, 3, 5` for `11`. A "repaired" single pass that conditions on whether index `0` was taken needs four states (each of `skip`/`take` split by `took0 = true/false`) and a final reconciliation — doable but error-prone to get right in budget, with silent wrong-answers exactly when the wrap binds.
- *Greedy by value.* Local max-grab under a global adjacency constraint is unsafe; e.g. on the line family `[8, 9, 2, 9, 9, -2, 8, -5]` greedy gets `26` while `{0,2,4,6}` reaches `27`, and this embeds into a cycle (pad with a large-negative spacer so the wrap never binds), reproducing the failure on a circle.

**Key idea — split on the wrap edge, run the proven path DP twice.** Any valid circular selection omits at least one of the two endpoints (it cannot take both `0` and `n-1`). So:

- Case A: forbid the last element → best is the *path* maximum over `a[0..n-2]`.
- Case B: forbid the first element → best is the *path* maximum over `a[1..n-1]`.

The circular answer is `max(A, B)`. This is exactly right:

- *Soundness.* A Case-A selection lives in `a[0..n-2]`, never uses `n-1`, so it cannot violate the wrap edge (which requires both `0` and `n-1`); it is a legal circular selection. Symmetric for B.
- *Completeness.* Any optimal circular selection omits `0` or omits `n-1`, hence is counted in B or in A respectively. So `max(A, B)` is at least the true optimum.

**The path DP (the building block).** Scan left to right carrying `skip` = best prefix sum with the last position **not** taken, and `take` = best with it taken:

- `take_i = skip_{i-1} + a[i]` (taking `i` forces `i-1` skipped)
- `skip_i = max(skip_{i-1}, take_{i-1})` (skipping `i` leaves `i-1` free)

Initialize `skip = 0` (empty prefix) and `take = -inf` (no last-taken state before any element); the per-range answer is `max(take, skip, 0)`. An empty range returns `0`.

**Pitfalls to get right.**
1. *In-place update.* Compute both new values from the old `(skip, take)` via temporaries. Updating `skip` first and using it for `take` builds `take` on a state that already took `i-1` — illegal adjacency. (A trace of the line `[1, 1]` returning `2` exposes exactly this.)
2. *Small-`n` corners.* `n = 0` → `0`. `n = 1` has no neighbour, so the lone element may be taken: answer `max(a[0], 0)` — must be special-cased, because the split would wrongly zero it out. `n = 2` falls out of the split correctly (`max(a[0], a[1], 0)`).
3. *Overflow.* With `n` up to `2*10^5` and `|a[i]|` up to `10^9`, sums reach `~2*10^14`; use `long long`. An `int` is a silent wrong-answer on large tests. The sentinel `LLONG_MIN/4` is only read inside a `max`, never has `a[i]` added to it, so it cannot underflow.

**Edge cases (all handled):** `n = 0` → `0`; a single negative → `0`; all negatives → `0`; the wrap-binding `[5,1,1,5,1,5]` → `11` (never the illegal `15`).

**Complexity.** `O(n)` time, `O(1)` extra space (two passes).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Maximum sum of a no-two-adjacent subset on a LINE over a[lo..hi] (inclusive).
// Empty subset allowed, so the returned value is always >= 0.
// If lo > hi the range is empty and the best sum is 0.
static long long linearBest(const vector<long long> &a, int lo, int hi) {
    long long take = LLONG_MIN / 4; // best with last position taken (impossible before any element)
    long long skip = 0;             // best with last position not taken (empty prefix -> 0)
    for (int i = lo; i <= hi; i++) {
        long long ntake = skip + a[i];     // take i => i-1 skipped
        long long nskip = max(skip, take); // skip i => i-1 either
        take = ntake;
        skip = nskip;
    }
    return max({take, skip, 0LL});
}

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }
    if (n == 1) { cout << max(a[0], 0LL) << "\n"; return 0; }

    // Circle: positions 0 and n-1 are adjacent, so they cannot both be chosen.
    // Split into two LINE subproblems that each break the wrap edge:
    //   (A) forbid the last element  -> solve line over a[0 .. n-2]
    //   (B) forbid the first element -> solve line over a[1 .. n-1]
    // Any valid circular selection avoids at least one of {first, last}, so it is
    // covered by case A or case B; conversely any selection counted in A or B is a
    // valid line selection that never uses both endpoints, hence valid on the circle.
    long long best = max(linearBest(a, 0, n - 2), linearBest(a, 1, n - 1));

    cout << best << "\n"; // empty selection always allowed (linearBest already >= 0)
    return 0;
}
```
