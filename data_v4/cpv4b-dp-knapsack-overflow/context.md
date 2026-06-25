# Planetarium projector array under a power budget

## Research question

A planetarium is upgrading its dome show. Along the rail above the audience sit `n` candidate
projector modules. Module `i`, if switched on, draws `w[i]` watts from the dome's single power
supply and contributes `b[i]` units of "wow factor" (brightness) to the show. The supply can deliver
at most `C` watts in total. The crew may switch on any subset of the modules whose combined wattage
does not exceed `C`. They want the subset that **maximizes the total brightness**. Output that
maximum total brightness.

This is the classic 0/1 knapsack: each module is taken at most once, weights are the wattages, the
capacity is the power budget, and the values are the brightnesses. The reason it is worth stating
carefully is the magnitude of the brightness values — a single module can already contribute up to
`10^9`, and a feasible subset can contain hundreds of them, so the answer can be far larger than any
single input number.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `C`
  (`1 <= n <= 2000`, `0 <= C <= 4000`). Then `n` lines follow; line `i` has two integers
  `w[i]` and `b[i]` (`1 <= w[i] <= 4000`, `1 <= b[i] <= 10^9`).
- Output (stdout): a single line with the maximum achievable total brightness. If no module fits
  (every `w[i] > C`, e.g. when `C = 0`), the best subset is empty and the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: with `C = 7` and modules `(w, b) = (3, 4), (4, 5), (2, 3), (5, 6)`, the answer is `9`
(switch on modules 1 and 2: wattage `3 + 4 = 7 <= 7`, brightness `4 + 5 = 9`; the pair `(2,3)+(5,6)`
also reaches `9`, and nothing does better).

## Background

The constraint is a single global budget on a sum of weights, and the objective is to maximize a sum
of values — the textbook 0/1 knapsack. Two solution shapes are worth weighing before committing:

- **Capacity dynamic programming.** Keep `dp[c]` = the best total brightness achievable with total
  wattage at most `c`, and fold in one module at a time. With `C <= 4000` and `n <= 2000` this is a
  table of size `C + 1` updated `n` times — `O(n * C)` work, about `8 * 10^6` basic operations,
  comfortably inside one second. The open question is the update direction (each module used at most
  once) and, crucially, the data type of the table entries.
- **Subset enumeration.** Try all `2^n` subsets. This is obviously correct and is the right tool for
  a brute-force oracle on tiny `n`, but at `n = 2000` it is hopeless. It exists in this writeup only
  as the independent checker.

The detail that makes this problem more than boilerplate is the size of `dp[C]`: with up to `2000`
modules each worth up to `10^9`, the maximum total brightness is on the order of `2 * 10^12`, which
overflows a signed 32-bit integer (cap `~2.1 * 10^9`) by three orders of magnitude. Every table
entry and every partial sum has to be 64-bit, or the answer silently wraps on large cases.

## Evaluation settings

Judged on hidden tests covering: tiny instances checkable by hand, instances where the optimal subset
deliberately leaves wattage unused, instances where no module fits (`C = 0` and `w[i] > C`),
single-module instances, and large instances with `n = 2000`, `C = 4000`, and brightnesses near
`10^9` (so the running total exceeds the 32-bit range and a 32-bit accumulator produces a wrong,
wrapped answer).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;

    vector<long long> w(n), b(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> b[i];

    // TODO: compute the maximum total brightness over subsets with total wattage <= C.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
