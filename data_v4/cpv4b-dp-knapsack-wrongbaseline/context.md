# Scheduling experiments on a relay satellite under one energy budget

## Research question

A small relay satellite has a single charged-battery window before it leaves contact. During that
window it can run a menu of `n` candidate experiments. Experiment `i` draws `e[i]` units of energy
when it runs and returns `v[i]` units of science value. The battery holds `W` units of energy total.

Each experiment is a distinct piece of hardware, so **each experiment can be run at most once** during
the window (there is exactly one copy on board). Choose a subset of experiments whose combined energy
draw does not exceed `W`, so that the **total science value is maximized**. Output that maximum value.
Running nothing is allowed, so the answer is at least `0`.

This is the **0/1 knapsack** problem dressed in a power-budget story: "at most once" is the 0/1
constraint, energy is weight, science value is profit, and the battery is the capacity. The reason it
is worth getting exactly right is that the most popular space-saving knapsack implementation — the
single rolling 1D array — is correct for one knapsack variant and silently solves a *different*
variant if one loop bound is written the natural-looking way.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `W`
  (`1 <= n <= 2000`, `0 <= W <= 2*10^5`). Then `n` lines follow; line `i` has two integers
  `e[i]` and `v[i]` (`1 <= e[i] <= 10^5`, `0 <= v[i] <= 10^9`).
- Output (stdout): a single line with the maximum total science value achievable without exceeding
  the energy budget, using each experiment at most once.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 4`, `W = 10` and the experiments `(e, v) = (3, 8), (3, 8), (4, 9), (2, 5)`, the
answer is `25` — run the first three experiments (energy `3 + 3 + 4 = 10`, value `8 + 8 + 9 = 25`).
Note that you may *not* run the `(2, 5)` experiment three times to reach `26`; there is only one of it.

## Background

Two knapsack flavours sit next to each other and differ by a single line of code:

- **Unbounded knapsack** (each item available in unlimited supply). With a 1D array `dp[c]`, the
  textbook update sweeps capacity *ascending*: `for c = e[i] .. W: dp[c] = max(dp[c], dp[c-e[i]]+v[i])`.
  Ascending order means `dp[c-e[i]]` may already include item `i`, so item `i` is allowed to repeat.
- **0/1 knapsack** (each item at most once — this problem). The same 1D array must sweep capacity
  *descending*: `for c = W .. e[i]: dp[c] = max(dp[c], dp[c-e[i]]+v[i])`, so each item touches every
  state at most once per outer iteration.

Both are `O(n*W)` time and `O(W)` memory. The trap is that the ascending version compiles, runs fast,
and looks like "the standard knapsack DP", yet it answers the wrong question for the at-most-once rule.
Constraints here are chosen so the wrong variant actually deviates: energy costs `e[i]` can be far
smaller than `W`, so an item that may repeat would be run many times and inflate the answer.

## Evaluation settings

Judged on hidden tests covering: tiny menus checkable by hand; cases where the cheapest high-value
experiment could be "reused" if the loop direction were wrong; `W = 0` (nothing can run); experiments
that individually exceed `W`; values of `0`; and large instances with `n = 2000`, `W = 2*10^5`, and
`v[i]` near `10^9` so the total value overflows 32 bits (it can reach `~2*10^12`).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long W;
    if (!(cin >> n >> W)) return 0;
    vector<long long> e(n), v(n);            // e[i] = energy cost, v[i] = science value
    for (int i = 0; i < n; i++) cin >> e[i] >> v[i];

    // TODO: maximum total value with total energy <= W, using each experiment at most once.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
