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
problem: the selection here forms a cycle, so the wrap-around gap between the last and first chosen
post is a real constraint alongside all the others.

## Input / output contract

- Input (stdin): the first line has three integers `n k L`
  (`2 <= k <= n <= 2000`, `n <= L <= 10^9`).
  The second line has the `n` integers `p[i]` (`0 <= p[i] < L`), strictly increasing.
- Output (stdout): a single line with the maximum achievable clearance (an integer).
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `L = 20` and posts `[0, 1, 2, 3, 9, 11, 17]` with `k = 3`, the answer is `6`
(choose posts at `3, 9, 17`: clockwise gaps are `6, 8, 6`, whose minimum is `6`; no choice of three
posts does better).

## Evaluation settings

Judged on hidden tests covering: `k = 2`; `k = n`; tightly clustered posts where the wrap gap is the
binding one; posts spread almost uniformly; large `L` near `10^9`; and `n = 2000` to stress the
solution's running time.

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

    // TODO: compute the maximum clearance achievable by choosing k of the n
    //       posts (remember the selection wraps around the ring).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
