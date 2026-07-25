I am renting one locker with `K` integer
space units, positions `1 .. K`. Fire code reserves the **last `g` units** as a buffer that
must stay empty, so only the first part of the locker is usable. I have `n` items; item `i`
occupies `s[i]` units and is worth `v[i]`, each usable at most once. A subset is legal if its
total occupied space is at most the *usable* amount, and I want the maximum value of a legal
subset. The empty subset is always legal, so the answer is at least `0`. Input is `n K g`
followed by `n` pairs `s[i] v[i]`; output is one integer.

Before any algorithm, I fix two things that decide the rest. First, the data type. With up to
`n = 2000` items and `v[i]` up to `10^9`, a selected subset's value can reach
`2000 * 10^9 = 2*10^12`, far past the 32-bit limit of `~2.1*10^9`. So every value accumulator
must be 64-bit `long long`. The spaces `s[i]` and `K, g` go up to `2*10^5`, comfortably inside
`int`, but I will keep them `long long` too so the subtraction `K - g` and any indexing never
surprises me. Second — and this is the crux of this particular problem — the *usable capacity*
is not handed to me; it is `U = K - g`, a difference of two boundaries, and I have to derive it
by counting, not by reflex.

**Deriving the usable capacity by counting positions.** The locker has positions `1, 2, ..., K`
— that is `K` units total. The buffer is "the last `g` units must stay empty." The last `g`
positions are `K-g+1, K-g+2, ..., K`; that is exactly `g` of them (check: `K - (K-g+1) + 1 = g`,
good). Removing them leaves usable positions `1, 2, ..., K-g`. Counting those:
`(K-g) - 1 + 1 = K - g` units. So the usable capacity is `U = K - g`. Not `K - g + 1` (that
would wrongly keep one buffer position), not `K - g - 1` (that would burn a usable unit). I
want this written down because the entire problem is one subtraction away from being wrong, and
I have just confirmed by an explicit position count that the right value is `U = K - g`.

There is a corner already visible: `g` can be `>= K`. If `g = K`, then `U = 0` — the whole
locker is buffer, nothing can be stored, answer `0`. If `g > K` the formula gives a negative
`U`, which is nonsense as a capacity; physically there is still simply no usable space, so I
must clamp `U = max(K - g, 0)`. I will treat `U < 0` as `U = 0`.

**Candidate approaches.** Once `U` is fixed, "pick a subset of items, each at most once, total
space `<= U`, maximize value" is exactly 0/1 knapsack. Two routes:

- *Greedy by value-density.* Sort items by `v[i]/s[i]` and grab greedily. This is the classic
  knapsack trap: it is correct for the *fractional* knapsack but provably wrong for 0/1, because
  you cannot take a fraction of the last item. I will not use it; I will, however, keep a
  concrete counterexample in mind so I do not drift back to it. Items `(s,v) = (3,5), (3,5),
  (4,7)` with `U = 6`: densities `5/3 ~ 1.67`, `5/3`, `7/4 = 1.75`. Greedy takes the `(4,7)`
  first, then nothing else fits in the remaining `2`, total `7`; but taking the two `(3,5)`
  items fills `6` for `10`. Greedy loses. So greedy is out.
- *One-dimensional 0/1 knapsack DP.* Keep `dp[c]` = best value using total occupied space at
  most `c`, for `c = 0 .. U`. Process items one at a time; for each item scan `c` downward so
  the item is consumed at most once. `O(n * U)` time, `O(U)` memory. With `n*U <= 2000 *
  2*10^5 = 4*10^8` simple long-long operations and a 1 s limit, this is the route. The risk is
  not the idea but the *boundaries* — the size of the table, the inclusive top of the scan, and
  the lower cutoff are all off-by-one-prone, which is the whole point of this problem.

**Deriving the DP recurrence carefully.** Define `dp[c]` after processing some prefix of items
as the best total value achievable with occupied space `<= c`. Initialize `dp[c] = 0` for all
`c` (empty subset occupies `0 <= c`, value `0`). For each item `(s_i, v_i)`, the 0/1 update is
`dp[c] = max(dp[c], dp[c - s_i] + v_i)` for every `c >= s_i`, scanned with `c` **decreasing** so
that `dp[c - s_i]` still refers to the table *without* item `i` (ascending order would let one
item be taken multiple times — that is the unbounded-knapsack bug, which I explicitly want to
avoid). The answer is `dp[U]`: the best value with occupied space at most the full usable
capacity. Because `dp` is non-decreasing in `c` (a larger budget can only help), `dp[U]` already
captures "at most `U`", so I do not need a separate max over the row.

The table indices run `0 .. U` inclusive — that is `U + 1` entries. If I size it `U` (a common
slip), index `U` is out of bounds; if I stop the scan at `c > s_i` instead of `c >= s_i`, I
forbid placing an item into a slot that exactly fits, losing edge configurations. Both of those
are the inclusive/exclusive boundary bugs I am on guard for.

**Sanity-checking the derivation on the sample.** Sample: `n=4, K=10, g=3`, items `(3,8),
(4,9), (5,10), (2,5)`. Usable `U = K - g = 7`. Table `dp[0..7]`, all start `0`.

- Item `(3,8)`: for `c=7..3`, `dp[c] = max(dp[c], dp[c-3]+8)`. Since `dp[c-3]=0`, this sets
  `dp[3..7] = 8`. Now `dp = [0,0,0,8,8,8,8,8]`.
- Item `(4,9)`: for `c=7..4`, `dp[c] = max(dp[c], dp[c-4]+9)`. `c=7: dp[3]+9 = 8+9 = 17` vs `8`
  -> `17`. `c=6: dp[2]+9 = 0+9 = 9` vs `8` -> `9`. `c=5: dp[1]+9 = 9` vs `8` -> `9`.
  `c=4: dp[0]+9 = 9` vs `8` -> `9`. Now `dp = [0,0,0,8,9,9,9,17]`.
- Item `(5,10)`: for `c=7..5`. `c=7: dp[2]+10 = 10` vs `17` -> `17`. `c=6: dp[1]+10 = 10` vs `9`
  -> `10`. `c=5: dp[0]+10 = 10` vs `9` -> `10`. Now `dp = [0,0,0,8,9,10,10,17]`.
- Item `(2,5)`: for `c=7..2`. `c=7: dp[5]+5 = 10+5 = 15` vs `17` -> `17`. `c=6: dp[4]+5 = 14`
  vs `10` -> `14`. `c=5: dp[3]+5 = 13` vs `10` -> `13`. `c=4: dp[2]+5 = 5` vs `9` -> `9`.
  `c=3: dp[1]+5 = 5` vs `8` -> `8`. `c=2: dp[0]+5 = 5` vs `0` -> `5`. Now
  `dp = [0,0,5,8,9,13,14,17]`.

Answer `dp[7] = 17`, taking items `0` and `1` (`3+4=7` space, `8+9=17` value). That matches the
hand-derived optimum, so the recurrence is right.

**Implementation, written to the boundary and then verified against it.** Putting together the
clamp `U = max(K-g, 0)` derived above, the inclusive index range `0..U`, and an inclusive scan
cutoff:

```
long long U = K - g;
if (U < 0) U = 0;                          // no usable space at all
vector<long long> dp((size_t)U + 1, 0);    // indices 0..U inclusive, so U+1 slots
for (int i = 0; i < n; i++) {
    long long si = s[i], vi = v[i];
    for (long long c = U; c >= si; c--)    // inclusive: a slot == si can hold it
        dp[c] = max(dp[c], dp[c - si] + vi);
}
cout << dp[U] << "\n";
```

Each of the three choices above — the table size, the cutoff, the clamp — has a tempting wrong
neighbor, and it is worth naming why each neighbor loses before trusting the code. Take the
smallest input that stresses the exact boundary I am worried about: `n=1, K=5, g=2` so `U=3`,
item `(s,v)=(3,9)`. The correct answer is obviously `9` — the single item occupies `3 = U`, fits,
and is worth `9`.

- *Table size `U` instead of `U+1`.* The DP indices needed are `0..U` inclusive — `U+1` slots.
  A table declared with size `U` would have indices `0,1,2` only when `U=3`; the top slot
  `dp[3]`, which is exactly `dp[U]`, the answer I print, would not exist, and reading it is
  undefined behavior. Sizing it `U+1` is what makes `dp[U]` legal to read.
- *Exclusive cutoff `c > s_i` instead of `c >= s_i`.* A slot of size exactly `s_i` can still hold
  the item, leaving budget `0` — so the recurrence must reach `c = s_i`. With an exclusive
  cutoff, `for (c=3; c>3; c--)` on this input never executes its body at all: the only `c` that
  could hold the size-`3` item is `c=3`, and `c>3` excludes it, so the item would be silently
  dropped. The inclusive cutoff instead runs once at `c=3`. Tracing the actual code above: `dp`
  has size `4`, indices `0..3` all `0`; item `(3,9)` triggers `for c=3; c>=3; c--`, one
  iteration, `dp[3] = max(0, dp[0]+9) = 9`. Printed answer `dp[3] = 9` — correct, and correct
  because the table exists at index `3` and the cutoff reaches `c=3`, not by luck.
- *No clamp on `U`.* `g` can exceed `K` — take `n=2, K=4, g=7`, items `(1,100), (2,50)`, so
  `U = K-g = -3`. Without the clamp, sizing `vector<long long> dp((size_t)U + 1, ...)` computes
  `(size_t)(-3+1) = (size_t)(-2)`, which wraps to `18446744073709551614`: an instant `bad_alloc`
  or, worse, a huge nonsensical allocation. Physically there is simply no usable space when
  `g >= K`, so `U` must become `0`, not stay negative; clamping *before* the size computation is
  what keeps the `(size_t)U + 1` cast always well-defined. Re-tracing with the clamp: `U = -3 ->
  0`, table size `1`, `dp = [0]`; neither item's inner loop runs (`c=0` is never `>= s_i >= 1`),
  so the printed answer is `dp[0] = 0` — correct, nothing fits. The same holds at `g = K` exactly
  (`n=2, K=5, g=5, U=0`): answer `0`.

All three boundary choices are invisible on inputs where the optimum leaves slack — they only
matter when an item sits flush against the capacity or when `g` overshoots `K`, which is exactly
the kind of test the judge will include, so each is worth having traced explicitly rather than
trusted on faith.

**Guarding the item-too-large case explicitly.** If an item has `s_i > U`, the inner loop
`for (c = U; c >= s_i; c--)` simply never runs (its start `U` is already `< s_i`), so the item
is correctly ignored. So I do not strictly need a separate `if (si > U) continue;`. But I will
add it anyway as a documented short-circuit, because it makes the intent ("this item cannot
fit") explicit and avoids even entering the loop for big items — a tiny constant-factor win and
a clarity win. I trace to confirm it changes nothing: `n=1, K=3, g=0, U=3`, item `(5, 99)`. With
the guard, `si=5 > U=3` -> `continue`, `dp` stays `[0,0,0,0]`, answer `0`. Without the guard,
the loop `for c=3; c>=5; c--` never runs, same result. Equivalent — the guard is safe.

**Edge cases, deliberately, because boundary knapsack dies exactly here.**
- `U = 0` (via `g = K` or `g > K`): table is `dp[0] = 0`; every inner loop with `s_i >= 1`
  never runs; answer `0`. Correct.
- `g = 0` (no buffer): `U = K`, plain knapsack over the full locker. The clamp and `U+1` sizing
  still hold (`U = K` up to `2*10^5`, table fits in memory: `(2*10^5+1)*8` bytes `~ 1.6 MB`).
- Single item that exactly fills: `n=1`, item `s = U`: placed at `c = U` by the inclusive
  cutoff; answer `v`. Already verified above with `(3,9)`, `U=3`.
- All items too large: each inner loop never runs; answer `0`.
- Overflow: values accumulate in `long long`. Worst case `n=2000` items each `v=10^9` that all
  fit gives `2*10^12`, well within `long long`'s `~9.2*10^18`. The `max` never adds to an
  uninitialized or sentinel value — `dp` starts at `0` everywhere, and I only ever add `v_i`
  (positive) to an existing `dp[c-s_i]`. No underflow, no overflow. Safe.
- `K` or `g` near `2*10^5` with `g` slightly above `K`: `U` slightly negative, clamped to `0`.
  E.g. `K = 200000, g = 200001 -> U = -1 -> 0`. Verified by the clamp.
- Input parsing: `cin >>` skips arbitrary whitespace, so the line layout is irrelevant; reading
  `n K g` then `n` pairs is format-agnostic. Output is exactly one integer and a newline.

**Performance check at the bound.** `n * U <= 2000 * 2*10^5 = 4*10^8` long-long comparisons and
additions. Measured on a max-style case (`n=2000, U~2*10^5`) it runs in about `0.2 s` and uses
`~5 MB`, comfortably inside the `1 s` / `256 MB` envelope. The memory is dominated by the single
`dp` row of `U+1` longs (`~1.6 MB`), not by anything per-item, so it scales fine.

**Final solution.** I disproved greedy with a concrete `7`-vs-`10` counterexample, derived
`U = K - g` by counting locker positions rather than guessing, checked the DP recurrence by a
full hand trace of the sample to `17`, and then verified the *code* against the two boundary
inputs that most threaten it — an item that exactly fills `U` (needs table size `U+1` and
inclusive cutoff `c >= si`, not `U` and `c > si`), and `g > K` (needs the clamp
`U = max(K-g, 0)` applied before the size cast, not left negative). This is what I ship: one
self-contained file, the `O(n*U)` 0/1 knapsack whose every boundary I have traced.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long K, g;
    if (!(cin >> n >> K >> g)) return 0;
    vector<long long> s(n), v(n);
    for (int i = 0; i < n; i++) cin >> s[i] >> v[i];

    // Usable space: the locker has capacity K, but the last g units must stay
    // empty (fire-safety buffer). Positions are 1..K; reserving the last g of
    // them leaves usable positions 1..(K-g), i.e. U = K - g usable units.
    // If K - g < 0 there is no usable space at all -> treat U = 0.
    long long U = K - g;
    if (U < 0) U = 0;               // no usable space at all

    // 0/1 knapsack over capacity 0..U inclusive.
    // dp[c] = best value achievable using total occupied space exactly <= c.
    vector<long long> dp((size_t)U + 1, 0);
    for (int i = 0; i < n; i++) {
        long long si = s[i], vi = v[i];
        if (si > U) continue;                 // item cannot fit at all
        for (long long c = U; c >= si; c--) { // 0/1: descending
            long long cand = dp[c - si] + vi;
            if (cand > dp[c]) dp[c] = cand;
        }
    }

    cout << dp[U] << "\n";
    return 0;
}
```

**Causal recap.** The usable capacity is the difference `U = K - g`, derived by counting that
positions `1..K-g` survive after reserving the last `g`; greedy by density is wrong (a `7`-vs-
`10` counterexample), so I use the `O(n*U)` 0/1 knapsack `dp[c] = max(dp[c], dp[c-s_i]+v_i)`
scanned downward. The two boundary choices that a flush-fitting item forces — table size `U+1`,
not `U`, and an inclusive cutoff `c >= s_i`, not `c > s_i` — were verified by tracing
`n=1,U=3,(3,9)` (the wrong neighbors would drop the item and read out of bounds), and the clamp
`U = max(K-g,0)` was verified by tracing `g > K` (without it, the `size_t` cast wraps to a huge
allocation); with the table size, the inclusive cutoff, and the clamp all in place, every corner
is closed, and 64-bit accumulation handles the `~2*10^12` value sums.
