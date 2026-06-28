**Problem.** Given an integer array `a[0..n-1]` (values may be negative), choose **exactly `k`** pairwise-disjoint non-empty contiguous segments to maximize the total value (sum of all covered cells). Read `n k` then the values from stdin, print the maximum total. Constraints: `n <= 2*10^5`, `1 <= k <= ceil(n/2)`, `|a[i]| <= 10^9`.

**Why the obvious DP is too slow.** The exact per-`k` DP keeps two layers `closed[i][j]` / `open[i][j]` ("position `i-1` not inside / inside the `j`-th open segment") with transitions `closed[i+1][j] = max(closed[i][j], open[i][j])` and `open[i+1][j] = max(open[i][j], closed[i][j-1]) + a[i]`. It is correct, but it is `O(nk)`; with `n = k = 2*10^5` that is `4*10^10` operations — far past a 1-second limit. The "exactly `k`" constraint is also why we cannot just take profitable segments and stop: a large `k` can force loss-making segments, a small `k` can force merging good ones.

**Key idea — the Lagrangian / "Aliens" trick.** Let `f(k)` be the optimum for exactly `k` segments. `f` is **concave**: the marginal value `s_j = f(j) - f(j-1)` of the `j`-th segment is non-increasing (the best splits/additions are used first). Concavity lets us drop the `k`-dimension:

- Charge a penalty `lambda` per opened segment and solve the **unconstrained** problem in `O(n)`: `g(lambda) = max_j ( f(j) - lambda*j )`. The penalized DP keeps just two states — `out` (no segment open) and `in` (a segment open ending here) — paying `lambda` whenever it opens a segment.
- Because the `s_j` are non-increasing, the optimal segment count is non-increasing in `lambda`. Breaking ties toward **fewer** segments makes the DP report `cntMin(lambda) = #{ j : s_j > lambda }`, which is monotone.
- **Binary-search the smallest integer `lambda` with `cntMin(lambda) <= k`.** At that `lambda`, `k` lies in the optimal plateau, every optimal `j` satisfies `f(j) = g(lambda) + lambda*j`, so `f(k) = g(lambda) + lambda*k`.

This is `O(n log(range))`, where `range` is the slope magnitude — about `10^7` operations total.

**Pitfalls to get right.**
1. *In-place `in`/`out` update.* Compute both new states from the **old** `(in, out)` pair and commit only afterward; `newOut = better(out, in)` must read the pre-update values, or you build a state on an inconsistent prefix.
2. *Overflow — the real trap.* The slopes can reach magnitude `~S = sum|a[i]|` (up to `2*10^14`), because forcing an extra segment may require bridging a long negative run; so `lambda` ranges over `[-S, S]`, *not* `[-10^9, 10^9]`. To push the optimal count up to `k = ceil(n/2)`, the search drives `lambda` to `~-10^14`, making both the penalized accumulator `g(lambda)` and the recovery term `lambda*k` reach `~10^19` — which **overflows int64** (it shows up as a negative garbage answer on the all-`10^9`, `k = ceil(n/2)` boundary). The fix: carry the penalized value and the final `lambda*k` in `__int128` (the final answer `f(k)` itself is a small segment sum `<= 2*10^14`, but the two terms summed to reach it are huge and opposite-signed). `lambda` and the inputs stay `int64`.
3. *Tie-break + search direction must be consistent.* "Prefer fewer segments" pairs with "smallest `lambda` whose count `<= k`." Mixing rules breaks monotonicity.
4. *Sentinel hygiene.* The `in = -inf` sentinel must never have `a[i]` added into it (guard the extend transition), and must be far enough below any reachable value (`(__int128)(-1) << 100`).

**Edge cases.** `k = 1` reduces to Kadane (best single subarray). All-negative with forced `k` returns the `k` least-bad single cells (e.g. `[-3,-1,-4,-1,-5]`, `k=2` → `-2`). `k = ceil(n/2)` forces a near-fixed every-other-cell layout (`[2,2,2,2,2,2]`, `k=3` → `8`). `n=1, k=1` takes the single cell even if negative. The all-`±10^9` and all-`10^9`-boundary cases exercise the `__int128` path.

**Complexity.** `O(n log S)` time, `O(1)` extra space (the `O(nk)` brute is used only as the small-case oracle).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
  Aliens-Trick Job Split.

  Given an integer array a[0..n-1] (values may be negative), pick EXACTLY k
  disjoint non-empty contiguous segments; the value of a choice is the sum of
  all covered elements. Maximize that total.

  Let f(k) be the optimum for exactly k segments. f is CONCAVE in k: the
  marginal value of the j-th segment, s_j = f(j) - f(j-1), is non-increasing in
  j. So we use the Lagrangian / "Aliens" trick.

  Charge a penalty lambda for every segment opened and solve the UNCONSTRAINED
  problem (any number of segments) in O(n):
      g(lambda) = max_j ( f(j) - lambda * j ).
  With slopes s_j non-increasing, the maximizer set is { j : s_j >= lambda }.
  Tie-breaking toward FEWER segments makes the DP report
      cntMin(lambda) = #{ j : s_j > lambda },
  which is non-increasing in lambda. We binary-search the SMALLEST integer
  lambda with cntMin(lambda) <= k. At that lambda, k lies in the optimal range
  [cntMin(lambda), cntMax(lambda)], where every optimal j satisfies
      f(j) = g(lambda) + lambda * j,
  hence f(k) = g(lambda) + lambda * k.

  Penalized DP (one pass), two states:
      out = best (penalized value, #segments) with NO segment currently open;
      in  = best (penalized value, #segments) with a segment open ending here.
  Opening a segment adds (x - lambda) and +1 to the count; extending adds x.

  Overflow note. The slopes s_j can reach magnitude ~ sum|a[i]| (forcing an
  extra segment may require absorbing a long negative run), so lambda ranges
  over [-S, +S] with S = sum|a[i]| up to 2e14. The penalized value then
  accumulates up to k * S ~ 1e5 * 2e14 = 2e19, which OVERFLOWS int64. We carry
  the penalized value and the final lambda*k in __int128.
*/

typedef long long ll;
typedef __int128 lll;

const lll NEG = (lll)(-1) << 100;  // far below any reachable penalized value

struct State {
    lll val;  // best penalized value (128-bit: may exceed int64)
    ll cnt;   // segments used to attain it (min-count tie-break)
};

// Better state for MAXIMIZATION; on equal value prefer the SMALLER count.
static inline State better(const State &a, const State &b) {
    if (a.val != b.val) return a.val > b.val ? a : b;
    return a.cnt <= b.cnt ? a : b;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    ll k;
    if (!(cin >> n >> k)) return 0;
    vector<ll> a(n);
    for (auto &x : a) cin >> x;

    // Penalized solve for a fixed integer lambda: returns (g(lambda), cntMin).
    auto solve = [&](ll lambda) -> State {
        State out{0, 0};      // empty prefix: value 0, 0 segments, nothing open
        State in{NEG, 0};     // cannot be "inside" a segment before any element
        for (int i = 0; i < n; i++) {
            lll x = a[i];
            // open a new segment at i (from an "out" state): pay lambda, cnt+1
            State openNew{out.val + x - (lll)lambda, out.cnt + 1};
            // extend the currently-open segment (only if "in" is reachable)
            State extend{in.val <= NEG ? NEG : in.val + x, in.cnt};
            State newIn = better(openNew, extend);
            // after i, not inside: either we were out, or we just closed "in"
            State newOut = better(out, in);
            in = newIn;
            out = newOut;
        }
        return better(out, in);
    };

    // Segments are non-empty and separated by >=1 unused cell, so the maximum
    // number of segments is ceil(n/2) = (n+1)/2. The statement guarantees
    // 1 <= k <= (n+1)/2, so a valid selection always exists; the guards below
    // just keep the program total for malformed input.
    if (k <= 0) { cout << 0 << "\n"; return 0; }
    if (k > (ll)(n + 1) / 2) { cout << "IMPOSSIBLE\n"; return 0; }

    // Slopes s_j live in [-S, S] with S = sum |a[i]|; search lambda over that
    // range (with a margin). cntMin(lambda) is non-increasing in lambda.
    ll S = 0;
    for (ll x : a) S += llabs(x);
    ll lo = -S - 1, hi = S + 1;          // invariant: cntMin(hi) <= k always
    // Find the smallest lambda in [lo, hi] with cntMin(lambda) <= k.
    while (lo < hi) {
        ll mid = lo + (hi - lo) / 2;     // floor midpoint, lo <= mid < hi
        if (solve(mid).cnt <= k) hi = mid;  // mid works -> answer <= mid
        else lo = mid + 1;                  // mid fails -> answer > mid
    }
    ll lambda = lo;
    lll g = solve(lambda).val;           // g(lambda)
    lll answer = g + (lll)lambda * (lll)k;   // f(k) = g(lambda) + lambda * k

    // print a (possibly large) signed __int128
    if (answer < 0) { cout << '-'; answer = -answer; }
    if (answer == 0) { cout << '0'; }
    else {
        string s;
        while (answer > 0) { int d = (int)(answer % 10); s += char('0' + d); answer /= 10; }
        reverse(s.begin(), s.end());
        cout << s;
    }
    cout << "\n";
    return 0;
}
```
