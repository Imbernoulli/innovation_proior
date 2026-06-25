# Cheapest set of uplink bursts to deliver every channel

## Research question

A satellite uplink station must deliver `m` data channels (numbered `0..m-1`) to a ground
receiver. The station has `k` pre-recorded **bursts** it can fire. Firing burst `j` costs `c[j]`
energy units and delivers exactly the set of channels described by `mask[j]` (a channel that is
already delivered being re-sent by a later burst is harmless — redundancy is allowed). The station
may fire any subset of the bursts, in any order.

Choose a subset of bursts whose delivered channels together cover **all** `m` channels, minimizing
the total energy spent. If no subset covers every channel, report that the task is impossible.

This is exactly weighted set cover over a small universe. It is the kernel that appears inside
coverage planning, test-suite minimization, and feature-flag rollout problems, so getting the exact
optimum — not a near-optimum — right on a small universe is what matters here.

## Input / output contract

- Input (stdin): the first line has two integers `m` and `k`
  (`0 <= m <= 18`, `0 <= k <= 200`).
- Then `k` lines follow, one per burst. Line `j` begins with the cost `c[j]`
  (`1 <= c[j] <= 10^6`) and a count `t[j]` (`0 <= t[j] <= m`), followed by `t[j]` distinct
  channel indices in `[0, m-1]`. A burst may deliver zero channels.
- Output (stdout): a single line with the minimum total energy to deliver all `m` channels,
  or `-1` if it is impossible. When `m = 0` every requirement is vacuously met, so the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example. With `m = 4` and three bursts —

```
4 3
5 3 0 1 2
4 2 0 1
4 2 2 3
```

the answer is `8`: fire burst 2 (channels {0,1}, cost 4) and burst 3 (channels {2,3}, cost 4).

## Background

The constraint is that the **union** of the chosen bursts' masks must equal the full channel set,
and we minimize a sum of costs. Two families of approach are on the table before committing:

- **Greedy by efficiency.** Repeatedly fire the burst with the best ratio of cost to *newly*
  covered channels, until everything is covered. It is fast and intuitive — the classic set-cover
  heuristic — and the open question is whether always grabbing the locally most efficient burst is
  actually optimal.
- **Bitmask dynamic programming.** Because `m <= 18`, a set of delivered channels is one of only
  `2^m <= 262144` subsets, which fits in an `int`. Treat each subset as a DP state and relax forward
  through bursts. This is `O(2^m * k)`; the open questions are the exact transition and the order in
  which states must be processed.

## Evaluation settings

Judged on hidden tests covering: instances where the efficiency greedy is strictly suboptimal;
impossible instances (some channel covered by no burst); `m = 0` and `k = 0` corners; bursts that
deliver zero channels or overlap heavily; large instances at `m = 18`, `k = 200` with single-channel
bursts that make almost the entire `2^m` state space reachable; and costs near `10^6` summed over
many bursts (so totals exceed a 32-bit range).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, k;
    if (!(cin >> m >> k)) return 0;

    vector<int> mask(k, 0);
    vector<long long> cost(k, 0);
    for (int j = 0; j < k; j++) {
        long long c; int t;
        cin >> c >> t;
        cost[j] = c;
        int mk = 0;
        for (int s = 0; s < t; s++) { int ch; cin >> ch; mk |= (1 << ch); }
        mask[j] = mk;
    }

    // TODO: minimum total cost of a subset of bursts whose masks cover all m channels (or -1).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
