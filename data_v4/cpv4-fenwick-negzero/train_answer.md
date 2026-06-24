**Problem.** Given `a[0..n-1]` (values may be negative, zero, or positive), count the contiguous
subarrays `(l, r)` with `0 <= l <= r < n` whose sum `a[l] + ... + a[r]` is **strictly** less than
zero. Read `n` and the values from stdin, print the count. Constraints: `0 <= n <= 2*10^5`,
`-10^9 <= a[i] <= 10^9`.

**Key idea — reduce to strict inversions, count with a Fenwick.** Let `P[0] = 0` and
`P[k] = a[0] + ... + a[k-1]`. Then `sum(l, r) = P[r+1] - P[l]`, so
`sum(l, r) < 0  <=>  P[j] < P[i]` with `i = l`, `j = r+1`, `0 <= i < j <= n`. The answer is the
number of *strict inversions* of the `(n+1)`-element prefix array. Coordinate-compress the prefix
sums (negatives and zeros included), then sweep `j = 0..n` in index order over a Binary Indexed
Tree of frequencies: before inserting `P[j]`, the number of already-inserted earlier values
strictly greater than `P[j]` is `inserted - (count of values <= P[j])`, where the Fenwick prefix
sum `sumPrefix(rank(P[j]))` gives the "count `<= P[j]`." Add that, then insert `P[j]`. `O(n log n)`.

**Why brute force is out.** The `O(n^2)` running-sum tally is correct but does `~2*10^10`
operations on the largest input — far over a 1-second limit. It is useful only as an oracle.

**Pitfalls to get right.**
1. *Base case `P[0] = 0` must be seeded.* Sweep from `j = 0`, not `j = 1`, and insert `P[0]`.
   Subarrays starting at `l = 0` are inversions against `P[0]`; if you never insert `0`, every
   left-anchored subarray is lost. On the all-negative `[-2, -5, -1]` this returns `3` instead of
   the correct `6` — exactly half, the left-anchored half.
2. *Strict vs non-strict at the boundary.* Use `sumPrefix(rank)` (count `<= P[j]`) so `greater`
   counts values **strictly** `> P[j]`, i.e. strictly negative subarrays. Using `sumPrefix(rank-1)`
   would count `>= P[j]` (sum `<= 0`), wrongly including zero-sum subarrays. The all-zero
   `[0, 0, 0]` separates them: correct `0` versus the wrong `6`.
3. *Overflow.* The count can reach `~2*10^10` and a prefix sum `~2*10^14`; both need `long long`.
   An `int` is a silent wrong-answer on large tests.

**Edge cases.** `n = 0` (or empty stdin) -> `0`; a single zero -> `0` (zero is not negative); a
single negative -> `1`; all-positive -> `0`; all-zero -> `0`; all-negative `n` elements ->
`n(n+1)/2`.

**Complexity.** `O(n log n)` time, `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Fenwick (Binary Indexed Tree) for prefix counts over compressed values.
struct Fenwick {
    int n;
    vector<long long> t;
    Fenwick(int n) : n(n), t(n + 1, 0) {}
    void add(int i, long long v) {            // i is 1-based
        for (; i <= n; i += i & (-i)) t[i] += v;
    }
    long long sumPrefix(int i) {              // sum over [1..i], i is 1-based
        long long s = 0;
        for (; i > 0; i -= i & (-i)) s += t[i];
        return s;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;               // empty input -> n = 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Prefix sums P[0..n], P[0] = 0, P[k] = a[0] + ... + a[k-1].
    // Subarray [l, r] sum = P[r+1] - P[l]; we want it strictly negative,
    // i.e. count pairs (i, j) with 0 <= i < j <= n and P[j] < P[i].
    vector<long long> P(n + 1);
    P[0] = 0;
    for (int k = 0; k < n; k++) P[k + 1] = P[k] + a[k];

    // Coordinate-compress all n+1 prefix-sum values (negatives/zeros included).
    vector<long long> vals(P.begin(), P.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();
    auto rankOf = [&](long long x) {          // 1-based rank in sorted unique vals
        return int(lower_bound(vals.begin(), vals.end(), x) - vals.begin()) + 1;
    };

    // Sweep j = 0..n in order. For each P[j], the number of earlier P[i] (i < j)
    // with P[i] > P[j] is (j) - (count of earlier values <= P[j]).
    Fenwick fen(m);
    long long answer = 0;
    for (int j = 0; j <= n; j++) {
        int r = rankOf(P[j]);
        long long inserted = j;               // we have inserted P[0..j-1] so far
        long long leq = fen.sumPrefix(r);     // earlier values with rank <= r (i.e. P[i] <= P[j])
        long long greater = inserted - leq;   // earlier values with P[i] > P[j]
        answer += greater;
        fen.add(r, 1);                        // now insert P[j]
    }

    cout << answer << "\n";
    return 0;
}
```
