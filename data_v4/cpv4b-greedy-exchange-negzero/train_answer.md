**Problem.** There are `n` ballast crates; loading crate `i` changes net ballast by an integer `c[i]` (positive, zero, or **negative**). The crew must load a count `t` with `L <= t <= K` (`0 <= L <= K <= n`). Maximize the total `sum of c[i]` over the chosen crates. Read `n L K` then the `n` deltas from stdin; print the maximum total.

**Key idea — greedy-exchange + a prefix-sum sweep.** For a *fixed* count `t`, the best `t` crates are provably the `t` largest deltas: if a chosen crate had a smaller delta than some unchosen one, swapping them keeps the count and strictly raises the sum (exchange argument; it never uses the sign of the deltas, so it holds even when all are negative). Therefore only prefix sums of the descending-sorted deltas matter. Let `P(t)` be the sum of the top `t` deltas. The answer is `max over t in [L, K] of P(t)`. Compute it as: sort descending, set the forced baseline `forced = P(L)` (the mandatory `L` largest deltas), then sweep `t` from `L` to `K-1`, adding `c[t]` to a running sum and tracking the maximum.

**Pitfalls.**
1. *Wrong base case / sign handling (the headline trap).* Seeding the answer at `0` smuggles in the "load nothing" option, but the empty load is only legal when `L = 0`. With `L > 0` and all-negative deltas, `max(0, negative)` returns the **forbidden** `0` instead of the forced negative sum. A trace of `n=2, L=K=2, c=[-1,-3]` returns `0` (wrong) instead of `-4`. Fix: seed `best` at the mandatory floor `forced = P(L)`. This is exactly `0` when `L = 0` (the legitimate empty load) and the forced negative sum when `L > 0` — one line handles both regimes.
2. *The `K` cap.* Even with many positive deltas, never take more than `K`; the sweep stops at `t < K`.
3. *Overflow.* With `n` up to `2*10^5` and `|c[i]|` up to `10^9`, sums reach `~2*10^14`; use `long long`. An `int` is a silent wrong-answer on large tests.

**Edge cases.** `n = 0` (then `L = K = 0`) -> `0`. `L = 0`, all negative -> `0` (load nothing). `L > 0`, all negative -> `P(L)`, the sum of the `L` least-negative deltas (still negative). `L = K` -> exactly `P(L)`, no count freedom. Zeros are a wash: adding a `0` leaves the running sum unchanged. Single forced negative (`n=1, L=1, c=[-9]`) -> `-9`.

**Complexity.** `O(n log n)` for the sort, `O(n)` for the sweep, `O(1)` extra space beyond the input.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, K;
    if (!(cin >> n >> L >> K)) return 0;   // empty input -> load nothing
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // We must load between L and K crates (inclusive), 0 <= L <= K <= n. Each
    // crate's net delta c[i] may be positive, zero, or negative. Maximize the sum
    // of the loaded crates' deltas.
    //
    // Exchange argument: for a FIXED count t, the best choice is the t largest
    // deltas (swap any omitted larger delta for a chosen smaller one to improve).
    // So sort descending and look only at prefix sums. The best count t (in
    // [L, K]) is: keep adding the next-largest delta while it is positive, but we
    // are FORCED to reach at least L crates even if that drags in negatives.
    sort(c.begin(), c.end(), greater<long long>());

    // prefix[t] = sum of the t largest deltas. Forced floor at t = L.
    long long forced = 0;          // sum of the first L crates (mandatory quota)
    for (long long i = 0; i < L; i++) forced += c[i];

    long long best = forced;       // baseline: exactly the L mandatory crates
    long long run = forced;
    for (long long t = L; t < K; t++) {
        run += c[t];               // adding the (t+1)-th largest delta
        best = max(best, run);
    }

    cout << best << "\n";
    return 0;
}
```
