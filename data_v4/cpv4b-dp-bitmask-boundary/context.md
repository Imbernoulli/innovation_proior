# Cheapest set of sealant strips that covers every vent

## Research question

A maintenance robot must seal a row of `m` ventilation slots numbered `0, 1, ..., m-1` on the side of
a server rack (`1 <= m <= 20`). The shop has `n` pre-cut sealant strips. Strip `i` is described by a
**half-open span** `[a_i, b_i)` and a cost `c_i`: when applied, it seals exactly the vents
`a_i, a_i+1, ..., b_i-1` (the left endpoint is included, the right endpoint is **not**). Strips may
overlap freely, you may apply any subset of them, and every vent must end up sealed.

Choose a subset of strips of **minimum total cost** whose sealed vents together cover all of
`0..m-1`, or report that no subset can cover every vent.

The vents that a single strip seals form a contiguous block, but the *union* you need is the whole
row, and the cheapest way to assemble it can mix overlapping blocks in a way no greedy left-to-right
sweep captures cleanly. With `m` small this is the textbook setting for a subset (bitmask) DP — and
the half-open `[a, b)` convention is exactly the kind of boundary that, transcribed by one position,
silently seals the wrong vents.

## Input / output contract

- Input (stdin): the first line has two integers `m` and `n` (`1 <= m <= 20`, `1 <= n <= 2*10^5`).
  Each of the next `n` lines has three integers `a_i b_i c_i` with `0 <= a_i < b_i <= m` and
  `0 <= c_i <= 10^9`. Strip `i` seals vents `a_i .. b_i-1` inclusive (half-open `[a_i, b_i)`).
- Output (stdout): a single line with the minimum total cost to seal every vent `0..m-1`, or `-1`
  if it is impossible.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for

```
4 4
0 2 5
2 4 6
0 4 12
1 3 4
```

the answer is `11`: strip 0 seals vents `{0,1}` for `5`, strip 1 seals vents `{2,3}` for `6`, and
together they cover `{0,1,2,3}` for `5 + 6 = 11`. The single strip 2 also covers everything but costs
`12`, so it loses.

## Background

The required union is an arbitrary subset of the `m` vents, so the state that matters is "which vents
are sealed so far" — a bitmask in `{0, 1, ..., 2^m - 1}`. Two families of approach are on the table
before committing:

- **Greedy / interval sweep.** Sort strips by left endpoint and repeatedly extend coverage. This is
  fast, but cost is attached to strips, not lengths, and overlaps are allowed, so the cheapest cover
  is not the one that reaches furthest right per step. Whether any greedy rule is optimal here is the
  open question.
- **Subset (bitmask) DP.** Let `dp[S]` be the minimum cost to have sealed exactly the set of vents
  `S`. Start from `dp[0] = 0` and relax by adding one strip at a time, `S -> S | mask_i`. The answer
  is `dp[(1<<m)-1]`. The open questions are the precise bit range each strip seals (the half-open
  boundary), the data type for costs, and the impossible case.

## Evaluation settings

Judged on hidden tests covering: feasible instances with overlapping strips where mixing two cheap
blocks beats one expensive full-width strip; instances that are impossible because some vent is never
inside any strip's span (`-1`); single-vent rows (`m = 1`); zero-cost strips; strips whose spans touch
at a point (`[0,2)` then `[2,4)`) to probe the inclusive/exclusive boundary; and large `m = 20` with
`n` up to `2*10^5` and costs near `10^9`, so total cost can exceed a 32-bit integer and the
`2^m * n` work must stay within time.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, n;
    if (!(cin >> m >> n)) return 0;

    vector<int> mask(n);
    vector<long long> cost(n);
    for (int i = 0; i < n; i++) {
        int a, b; long long c;
        cin >> a >> b >> c;
        // TODO: build mask[i] = set of vents this strip seals from the half-open span [a, b),
        //       then run a subset DP and print the min cost to reach the full set (or -1).
        mask[i] = 0;
        cost[i] = c;
    }

    long long answer = -1;
    cout << answer << "\n";
    return 0;
}
```
