# Highest-average sustained billing window (exact fraction)

## Research question

A streaming service records `n` consecutive days of **net daily revenue** `a[0..n-1]` (a day can be
negative — refunds and chargebacks can outweigh new subscriptions). The finance team wants the single
**contiguous block of at least `L` days** whose *average* net revenue per day is as high as possible.
A minimum length `L` is imposed so the result describes a *sustained* trend rather than one lucky day.

Among all contiguous windows `[i, j)` with length `j - i >= L`, report the maximum possible average
`(a[i] + ... + a[j-1]) / (j - i)`. Because averages are rational and the inputs are huge, the answer
must be printed as an **exact reduced fraction** `p/q` (with `q > 0`) — not a rounded decimal. Two
windows tie only when their fractions are equal; otherwise the larger fraction wins unambiguously.

This is the maximum-average-subarray-of-bounded-length problem. It looks like a prefix-sum exercise,
and it is — but the moment you compare two averages you are comparing two ratios of large integers,
and the safe way to do that is to cross-multiply rather than divide. With the sums and lengths in play
here, those cross-products overflow 64-bit arithmetic, so getting the *arithmetic* right is the whole
game.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `L` (`1 <= L <= n <= 2*10^5`). The second
  line has `n` integers `a[i]` (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line `p/q`, the maximum average as a fraction in lowest terms with
  `q > 0`. (If the maximum average is an integer `k`, print `k/1`.)
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 7`, `L = 3`, `a = [1, 12, -5, 6, 7, -2, 3]` the answer is `5/1`. The window
`[1, 5)` = `12, -5, 6, 7` has sum `20` over `4` days, average `20/4 = 5`; no window of length `>= 3`
does better.

## Background

Prefix sums turn every window sum into a difference: with `S[0] = 0` and `S[k] = a[0] + ... + a[k-1]`,
the window `[i, j)` has sum `S[j] - S[i]` and average `(S[j] - S[i]) / (j - i)`. Two families of
approach are on the table before committing:

- **Binary search on the answer (float).** Guess an average `x`; a window of length `>= L` beats `x`
  iff `sum(a[i..j-1] - x) >= 0`, which is a prefix-sum-with-`min` scan. This is `O(n log(range/eps))`.
  The open question is whether *floating-point* `x` can ever distinguish the true optimum from a
  fraction that is astronomically close to it — with denominators up to `2*10^5`, two distinct
  averages can differ by less than `1/(2*10^5)^2`, far below `double` precision near `10^9`.
- **Exact geometry on prefix points.** Treat each prefix as a point `P_k = (k, S[k])`. The average of
  window `[i, j)` is exactly the *slope* of the segment `P_i -> P_j`. Maximizing slope to a fixed
  right endpoint over a moving set of left endpoints is a convex-hull-tangent problem, solvable in
  `O(n)` with **only integer comparisons** — provided every comparison is done by cross-multiplying,
  never dividing. The open question is the exact width those products need.

## Evaluation settings

Judged on hidden tests covering: tiny `n`, `L = 1`, `L = n` (only the whole array qualifies),
all-negative arrays (the answer is then a negative fraction — there is no empty window because a
window must have length `>= L >= 1`), arrays whose optimum is a long window rather than a single big
day, near-tie inputs where two windows' averages differ by a hair, and large `n = 2*10^5` with values
near `+-10^9` (so prefix sums reach `~2*10^14` and slope cross-products reach `~4*10^19`, past the
signed 64-bit ceiling of `~9.2*10^18`).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n; ll L;
    if (scanf("%d %lld", &n, &L) != 2) return 0;
    vector<ll> a(n);
    for (auto &x : a) scanf("%lld", &x);

    vector<ll> S(n + 1);
    S[0] = 0;
    for (int i = 0; i < n; i++) S[i + 1] = S[i] + a[i];

    // TODO: among windows [i, j) with j - i >= L, find the maximum average
    //       (S[j] - S[i]) / (j - i); print it as a reduced fraction p/q (q > 0).
    ll p = 0, q = 1;

    printf("%lld/%lld\n", p, q);
    return 0;
}
```
