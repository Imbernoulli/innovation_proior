# Widest reservoir-release window within a discharge budget

## Research question

A dam operator has a fixed schedule of `n` hourly water releases `a[0..n-1]`, each a strictly
positive volume (cubic metres). Downstream regulations cap the **total** volume discharged during any
single continuous run of hours at `B`. The operator wants to run the gate for one **contiguous** block
of hours that pushes as much water as possible without the block's total exceeding `B`.

Formally: among all contiguous subarrays `a[l..r]` (including the empty block, which discharges `0`),
maximize `a[l] + a[l+1] + ... + a[r]` subject to that sum being `<= B`. Output the maximum achievable
block total.

This is the "longest/heaviest window under a sum cap" shape that appears inside rate-limiting,
streaming-quota, and resource-budget problems.

## Input / output contract

- Input (stdin): the first line holds two integers `n` (`0 <= n <= 2*10^5`) and `B`
  (`0 <= B <= 2*10^14`). The second line holds `n` integers `a[i]` (`1 <= a[i] <= 10^9`),
  whitespace-separated. (When `n = 0` the second line is empty or absent.)
- Output (stdout): a single line with the maximum contiguous block total that is `<= B`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 7`, `B = 15`, `a = [4, 2, 7, 3, 1, 6, 5]` the answer is `15` — the block of hours
`3..6` discharges `3 + 1 + 6 + 5 = 15`, exactly the cap, and no contiguous block totals more than 15
without exceeding it.

## Background

The constraint "contiguous and capped" makes this a windowed-selection problem. A full `O(n^2)` scan
of every start/end pair — accumulate each window's sum, keep the largest one that fits under `B` — is
obviously correct and trivial to write, but quadratic, which is hopeless at `n = 2*10^5` under a
one-second limit.

## Evaluation settings

Judged on hidden tests covering: all-equal releases, releases with a single dominant spike, `B`
smaller than every individual release (answer `0`, the empty block), `B` at least the total of all
releases (answer = whole-array sum), the empty schedule (`n = 0`), a single hour (`n = 1`), and large
`n = 2*10^5` with releases near `10^9` and `B` near `2*10^14`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long B;
    if (!(cin >> n >> B)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: find the maximum sum of a contiguous block a[l..r] with sum <= B
    //       (the empty block, sum 0, is always allowed).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
