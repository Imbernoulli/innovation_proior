**Problem.** Given `n` temperature anomalies `a[0..n-1]` (each may be negative, zero, or positive) and a comfort band `[lo, hi]`, count the unordered pairs of distinct days `(i<j)` with `lo <= a[i] + a[j] <= hi`. The band may be a single value (`lo == hi`) or empty (`lo > hi`). Read `n lo hi` then the values from stdin; print the count.

**Key idea — two pointers via prefix counts.** Sort `a`. Let `countLE(K)` be the number of unordered pairs with sum `<= K`, computed by a single linear sweep: with `l = 0`, `r = n-1`, if `a[l] + a[r] <= K` then (array sorted) every pair `(l, l+1), ..., (l, r)` qualifies, so add `r - l` and advance `l`; otherwise `a[r]` is too large for any left partner, so drop it with `r--`. Then

- `answer = countLE(hi) - countLE(lo - 1)`   (pairs in `[lo, hi]` = pairs `<= hi` minus pairs `<= lo-1`),

**but only when `lo <= hi`.** The sweep pairs strictly `l < r`, so there is no self-pair to subtract and no double counting to halve.

**Pitfalls.**
1. *Empty band / sign (base case).* The identity `#[lo,hi] = countLE(hi) - countLE(lo-1)` assumes `{s <= lo-1} ⊆ {s <= hi}`, i.e. `lo <= hi`. When `lo > hi` the subtraction can go *negative* — an impossible count. Guard it: if `lo > hi`, the answer is `0`. (A trace of band `[-7,-9]` on `[-6,-5,-3]` prints `-1` without the guard.)
2. *Inner off-by-one.* When `a[l]+a[r] <= K`, the qualifying pairs with smaller index `l` are `(l, l+1) ... (l, r)` — that is `r - l` pairs, **including** `(l, r)`. Using `r - l - 1` drops a pair (trace `[1,2]`, `K=5`: true count `1`, buggy count `0`).
3. *Overflow.* `n` up to `2*10^5` gives up to `~2*10^10` pairs, and a pair sum reaches `2*10^9`; both exceed 32-bit. Use `long long` for values, endpoints, the count, and the answer. `lo - 1` with `lo` down to `-2*10^9` stays inside `long long`.

**Edge cases (all handled by the sweep + the `lo > hi` guard):** `n = 0` and `n = 1` -> `0` (the sweep's loop never runs); all-negative anomalies with a positive band -> `0` (both counts equal); empty band `lo > hi` -> `0`; degenerate band `lo == hi` -> pairs summing to exactly `lo`; duplicate values -> still counted as distinct day-pairs.

**Complexity.** `O(n log n)` for the sort, then `O(n)` per `countLE` sweep; `O(n)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Count unordered pairs (i<j) with a[i]+a[j] <= K, on a SORTED array a.
// Classic two-pointer: for each right, the leftmost l with a[l]+a[right] <= K.
static long long countLE(const vector<long long>& a, long long K) {
    int n = (int)a.size();
    long long cnt = 0;
    int l = 0, r = n - 1;
    while (l < r) {
        if (a[l] + a[r] <= K) {
            // a[l..r-1] all pair with a[r] to satisfy <= K (array sorted)
            cnt += (long long)(r - l);
            l++;
        } else {
            r--;
        }
    }
    return cnt;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    long long lo, hi;
    cin >> lo >> hi;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    sort(a.begin(), a.end());

    // An empty band (lo > hi) admits no pair. Guard it BEFORE subtracting:
    // countLE(hi) - countLE(lo-1) is the count in [lo, hi] only when lo <= hi,
    // i.e. when {s <= hi} is a superset of {s <= lo-1}. If lo > hi the
    // subtraction can go negative, so the answer must be pinned to 0 here.
    long long ans;
    if (lo > hi) {
        ans = 0;
    } else {
        // pairs with sum in [lo, hi] = countLE(hi) - countLE(lo - 1)
        ans = countLE(a, hi) - countLE(a, lo - 1);
    }

    cout << ans << "\n";
    return 0;
}
```
