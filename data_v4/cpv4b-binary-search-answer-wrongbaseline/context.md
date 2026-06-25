# Repeaters on a ring road (maximize the minimum cyclic clearance)

## Research question

A circular service road has integer circumference `L`. Around it sit `n` candidate mounting posts at
**distinct** integer positions `0 <= p[0] < p[1] < ... < p[n-1] < L`, measured clockwise from a fixed
origin. You must install signal repeaters on **exactly `k`** of these posts.

For a chosen set of posts, list them in clockwise order around the ring; the **clearance** of the
installation is the smallest arc distance between two *cyclically consecutive* chosen posts (the gap
that wraps from the last chosen post back to the first, going clockwise, counts like any other gap).
Choose the `k` posts so that the clearance is **as large as possible**, and output that maximum
clearance.

This is the *circular* cousin of the textbook "place `k` items to maximize the minimum spacing"
problem. On a straight line that problem is the canonical binary-search-on-the-answer exercise with a
one-pass greedy feasibility test. The ring changes the geometry: the selection forms a cycle, so the
wrap-around gap is a real constraint, and the line greedy is not the same algorithm.

## Input / output contract

- Input (stdin): the first line has three integers `n k L`
  (`2 <= k <= n <= 2000`, `n <= L <= 10^9`).
  The second line has the `n` integers `p[i]` (`0 <= p[i] < L`), strictly increasing.
- Output (stdout): a single line with the maximum achievable clearance (an integer).
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `L = 20` and posts `[0, 1, 2, 3, 9, 11, 17]` with `k = 3`, the answer is `6`
(choose posts at `3, 9, 17`: clockwise gaps are `6, 8, 6`, whose minimum is `6`; no choice of three
posts does better).

## Background

The constraint "spread the chosen posts out" makes this a max-min selection problem. Two ideas are on
the table before committing:

- **Binary search on the clearance.** Guess a target clearance `d`; ask the yes/no question "can we
  choose at least `k` posts so that every cyclically consecutive gap is `>= d`?"; and binary-search
  the largest feasible `d`. The whole method stands or falls on writing that feasibility test
  correctly for a **ring** — the open question is exactly what "place greedily" means when the
  selection closes back on itself.
- **Brute force over subsets.** Enumerate every size-`k` subset, compute its cyclic clearance, keep
  the best. Obviously correct, but `C(n, k)` is astronomical for the stated `n`, so it only survives
  as a checker on tiny inputs.

The line version of the feasibility test — "anchor the first chosen item at the leftmost candidate,
sweep left to right, take a candidate whenever its distance from the last taken is `>= d`" — is the
algorithm most people reach for. Whether that same sweep answers the ring question is the crux.

## Evaluation settings

Judged on hidden tests covering: `k = 2` (a single pair, clearance is `min(gap, L - gap)`); `k = n`
(every post forced, clearance is the smallest existing gap including the wrap); tightly clustered
posts where the wrap gap is the binding one; posts spread almost uniformly; large `L` near `10^9` so
arc arithmetic must stay in 64-bit; and `n = 2000` to stress the feasibility test's running time.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; long long k, L;
    if (!(cin >> n >> k >> L)) return 0;
    vector<long long> p(n);
    for (auto &x : p) cin >> x;
    sort(p.begin(), p.end());

    // TODO: binary-search the largest clearance d such that >= k posts can be
    //       chosen with every cyclically consecutive gap >= d (ring geometry).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
