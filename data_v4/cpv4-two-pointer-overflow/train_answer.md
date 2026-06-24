**Problem.** Given `n` strictly positive releases `a[0..n-1]` and a cap `B`, find the maximum sum of a contiguous block `a[l..r]` whose sum is `<= B`; the empty block (sum `0`) is always allowed, so the answer is at least `0`. Read `n`, `B`, and the values from stdin; print the maximum block total.

**Key idea — two-pointer sliding window.** Because every `a[i] >= 1`, the sum of a window is monotone in its width: widening raises the sum, narrowing lowers it. Sweep `right` from `0` to `n-1` keeping a running window sum `cur = sum(a[left..right])`; whenever `cur > B`, advance `left` (dropping front elements) until `cur <= B`. For each `right`, the resulting window is the *widest* admissible block ending at `right`, and by monotonicity also the *heaviest*. Take the max of `cur` over all `right`.

**Why it is correct.** Two facts, both from positivity. (1) Among blocks ending at a fixed `right`, the one with the smallest admissible `left` has the largest sum, so the per-`right` window the sweep produces is optimal among blocks ending there; ranging `right` over all endpoints covers every contiguous block. (2) Adding the positive `a[right+1]` only tightens the budget, so the smallest admissible `left` never decreases — the left pointer moves only forward, giving `O(n)` total work. The argument depends on `a[i] >= 1`; with zeros or negatives a forward-only window would be wrong, but the contract guarantees positivity.

**Pitfalls.**
1. *Overflow (the headline).* With `n` up to `2*10^5` and `a[i]` up to `10^9`, a window sum and the cap `B` reach `~2*10^14`, far past the 32-bit `int` ceiling of `~2.1*10^9`. Keep `B`, the array, and the accumulators `cur`/`best` all `long long`. A reflexive `int cur` wraps silently: on `n=5`, `B=10^11`, `a=[10^9]*5` the correct answer is `5000000000`, but an all-`int` version prints `2000000000` — and it would still pass every small sample, only failing the large hidden tests. (Loop indices `left`/`right` may stay `int`; only the sums need widening.)
2. *Empty-window corner.* When a single release exceeds `B`, the shrink loop subtracts it and sets `left = right+1`, leaving `cur = 0`. Since `B >= 0`, the loop halts there, never indexes out of range, and the answer stays at the empty block's `0`. Initializing `best = 0` is what encodes "empty block allowed".

**Edge cases.** `n = 0` -> `0`; `n = 1` with the release over `B` -> `0`; `B` at least the whole-array total -> the full sum (`~2*10^14`, the overflow case); `B` smaller than every release -> `0`.

**Complexity.** `O(n)` time (each pointer advances at most `n` times), `O(1)` extra space beyond the input.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long B;
    if (!(cin >> n >> B)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Two-pointer / sliding window. All a[i] >= 1, so the window sum is
    // monotone in window width: as we extend the right end, the running sum
    // only grows, and shrinking the left end only shrinks it. For each right
    // we advance left just far enough that the window sum is <= B, then the
    // current window is the widest (hence, since values are positive, the
    // largest-sum) window ending at `right` that stays within budget.
    long long cur = 0;     // sum of a[left..right], must fit 64-bit
    long long best = 0;    // best window sum found so far (empty window = 0)
    int left = 0;
    for (int right = 0; right < n; right++) {
        cur += a[right];
        while (cur > B) {          // shrink from the left until within budget
            cur -= a[left];
            left++;
        }
        // now cur = sum(a[left..right]) <= B and is the max such ending here
        if (cur > best) best = cur;
    }

    cout << best << "\n";
    return 0;
}
```
