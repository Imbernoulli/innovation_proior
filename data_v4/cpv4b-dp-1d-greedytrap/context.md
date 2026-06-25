# Fording the river: cheapest legal sequence of stepping stones

## Research question

A river is crossed on a single line of `n` stepping stones, numbered `0` to `n-1` from the near
bank to the far bank. You begin on the near bank and must reach the far bank. A leap moves you
forward by **one or two** stones:

- From the near bank, your **first landing** must be on stone `0` or stone `1` (you cannot clear
  every stone in a single leap from the bank).
- From stone `i`, you may land next on stone `i+1` or stone `i+2`.
- You step off onto the far bank with a final leap from stone `n-1` or stone `n-2`; this last leap
  costs nothing.

Each stone `i` carries a **stamina cost** `c[i]` that you pay when you *land* on it. A cost may be
negative: that stone sits in a helpful eddy whose current pushes you forward and *returns* stamina.
The total cost of a crossing is the sum of `c[i]` over the stones you actually land on. Output the
**minimum** total stamina over all legal crossings.

This is the one-dimensional "which stones do I land on" problem that hides inside path-planning,
stair-climbing, and toll-route DPs. The structural catch is the `+1 / +2` reach: skipping a stone
now can force you onto a worse stone two leaps later, so a purely local choice need not be optimal.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `c[i]`
  (`-10^9 <= c[i] <= 10^9`), whitespace-separated. When `n = 0` there are no stones to print.
- Output (stdout): a single line with the minimum total stamina to cross.
- For `n = 0` the near bank *is* the far bank, so the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `c = [3, 7, 8, 6, 5, 7]` the answer is `16` — land on stones `0`, `2`, `4`
(`3 + 8 + 5`) and leap off from stone `4 = n-2` to the far bank.

## Background

The `+1 / +2` reach with a per-stone landing cost is a constrained selection problem on a line.
Two families of approach present themselves before committing to one:

- **Local greedy.** Land on the cheaper of stone `0` / stone `1`, then from each stone always hop to
  the cheaper of the next two reachable stones, until a leap clears the bank. It is `O(n)` and a few
  lines. The open question is whether "cheaper of the next two" can be defended: the reach is global
  (each landing constrains the next two), and that is exactly the setting in which a local rule may
  be cornered into an expensive future.
- **Linear dynamic programming.** Sweep the stones left to right, carrying for each stone the
  minimum total cost of any legal crossing that *stands on that stone*. This is `O(n)`, `O(1)` extra
  state if folded. The open question is the exact recurrence at the two ends (the restricted first
  landing and the cost-free final leap) and the data type for the accumulated sum.

## Evaluation settings

Judged on hidden tests covering: `n = 0` (answer `0`); `n = 1` and `n = 2` (boundary leaps);
all-non-negative costs (the pure "minimize landings" flavour); mixed signs; **all-negative** costs
(where you want to land on as many stones as the reach allows); adversarial layouts that defeat the
local greedy; and large `n = 2*10^5` with `|c[i]|` near `10^9`, so the accumulated sum can reach
about `2*10^14` and overflows a 32-bit integer.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // TODO: compute the minimum total stamina to cross, given the +1/+2 reach,
    //       the restricted first landing (stone 0 or 1), and the free final leap.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
