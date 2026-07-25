# Spreading sensors along a glacier transect (maximize the closest pair)

## Research question

A field team has drilled `n` candidate boreholes along a straight survey line across a glacier. The
line has an arbitrary origin stake; boreholes to the west of it have **negative** integer coordinates,
boreholes to the east have positive ones, and a borehole may sit exactly on the stake (coordinate `0`).
Two boreholes may even have been drilled at the **same** coordinate.

The team owns exactly `k` strain sensors (`2 <= k <= n`) and must place each sensor in a distinct
borehole. Sensors interfere when they are close, so the team wants the installation whose **closest
pair of sensors is as far apart as possible**. Formally: choose `k` of the `n` coordinates and let the
*isolation* of that choice be the minimum distance between any two chosen coordinates; report the
maximum isolation over all choices.

Because coordinates can be negative, zero, or duplicated, the corners matter: getting the
one-dimensional version exactly right — including the all-negative-coordinate and
all-identical-coordinate corners — is the point.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `k` (`2 <= k <= n <= 2*10^5`). The second
  line (or remaining whitespace-separated tokens) has the `n` coordinates `x[i]`
  (`-10^9 <= x[i] <= 10^9`); they need not be sorted and may repeat.
- Output (stdout): a single line with the maximum achievable isolation (a non-negative integer).
- Time limit: 1 second. Memory: 256 MB.

Example: for `k = 3` and coordinates `[-7, -3, 0, 0, 4, 9]` the answer is `7` (install at `-7`, `0`,
and `9`: the gaps are `7` and `9`, so the closest pair is `7` apart, and no choice of three does
better).

## Evaluation settings

Judged on hidden tests covering: all-positive coordinates, mixed signs straddling the stake,
all-negative coordinates, coordinates that include `0`, heavy duplicates, the forced case `k = n`,
the minimal case `k = 2`, and large `n = 2*10^5` with coordinates near `+-10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> x(n);
    for (auto &v : x) cin >> v;

    sort(x.begin(), x.end());

    // TODO: compute the largest distance d such that k boreholes can be chosen
    //       pairwise at least d apart; print that distance.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
