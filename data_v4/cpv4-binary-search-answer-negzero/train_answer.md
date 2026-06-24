**Problem.** Given `n` integer readings `a[0..n-1]` (each may be negative or zero) and an integer `k`, choose a contiguous block of length **at least** `k` and score it by the *minimum* reading inside it. Output the largest achievable block-minimum. If no block of length `>= k` exists (`n < k`, including `n = 0`), output the literal `INFEASIBLE`. Read `n`, `k`, and the values from stdin; print one line.

**Key idea — binary search on the answer.** "Maximize the minimum" is monotone. Guess a floor `x` and ask the decision question: *is there a run of at least `k` consecutive positions all with `a[i] >= x`?* That is a single linear scan with a run counter reset on every reading below `x`. Raising `x` can only break runs, never create them, so the predicate is non-increasing in `x`; the largest feasible `x` is the answer. Search `x` over `[min(a), max(a)]`, which is correct because the answer equals the minimum of some block and hence equals some `a[i]`.

**Why it is correct.** A block of length `>= k` with minimum `>= x` exists iff a run of `>= k` consecutive `>= x` exists (a longer all-`>= x` block contains a length-`k` one). `x = min(a)` is feasible (the whole array has length `n >= k` past the guard and every element is `>= min(a)`); `x = max(a)+1` is infeasible (no element reaches it). Monotonicity plus these bounds make the binary search land on the largest feasible floor, which is exactly the maximum block-minimum.

**Pitfalls.**
1. *Anchoring the search at 0.* Writing `lo = 0`, `hi = max(0, ...)` assumes the answer is non-negative. On an all-negative array like `[-5, -3]` the interval collapses to `[0, 0]` and the code prints the impossible `0` instead of the largest element `-3`. Fix: `lo = min(a)`, `hi = max(a)`, both allowed negative. The fingerprint of this bug is an all-negative input returning `0`.
2. *Floor midpoint with a "move `lo` up" search.* `mid = (lo + hi) / 2` truncates toward zero in C++, so it only accidentally works on negatives and **hangs** on an adjacent positive interval like `[2, 3]` (mid stays `2`, the feasible branch leaves `lo` unchanged). Fix: the upper midpoint `mid = lo + (hi - lo + 1) / 2`, which is `> lo`, `<= hi`, and sign-uniform.
3. *Infeasible / empty.* Handle `n < k` (which subsumes `n = 0` since `k >= 1`) **before** computing `min`/`max`, since folding over an empty array is undefined.

**Edge cases.** `n = 0` -> `INFEASIBLE`; `n < k` -> `INFEASIBLE`; `k = 1` -> global max; `k = n` -> global min of the whole array; all-negative -> a negative answer (not `0`); all-zero -> `0`; single element with `k = 1` -> that element.

**Complexity.** `O(n log(range))` time with range about `2*10^9` (about 31 feasibility scans), `O(n)` space for the array, `O(1)` extra.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n; long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // We must pick a contiguous block of length >= k. Its "robust floor" is the
    // minimum element of the block. Maximize that floor over all valid blocks.
    // If no block of length >= k exists (n < k), report INFEASIBLE.
    if ((long long)n < k) {
        cout << "INFEASIBLE" << "\n";
        return 0;
    }

    // Binary search on the answer x: feasible(x) = exists a run of >= k
    // consecutive positions all with a[i] >= x. feasible is monotone:
    // raising x can only shorten runs, so feasibility is non-increasing in x.
    // Bounds: any single element is >= min(a); a window's min is <= max(a).
    long long lo = LLONG_MAX, hi = LLONG_MIN;
    for (long long v : a) { lo = min(lo, v); hi = max(hi, v); }
    // lo is achievable (whole-array window of length n>=k has min >= lo),
    // hi+1 is not (no element is >= hi+1). Binary search the largest feasible x.

    auto feasible = [&](long long x) -> bool {
        long long run = 0;
        for (long long v : a) {
            if (v >= x) { run++; if (run >= k) return true; }
            else run = 0;
        }
        return false;
    };

    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2; // upper mid to avoid infinite loop
        if (feasible(mid)) lo = mid;
        else hi = mid - 1;
    }

    cout << lo << "\n";
    return 0;
}
```
