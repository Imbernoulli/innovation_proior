# Earliest moment a print shop clears its quota

## Research question

A print shop runs `n` presses on a shared paper feed. Press `i` is cyclic with period `s[i]`
seconds: it finishes its first sheet at time `s[i]`, its second at `2*s[i]`, its third at `3*s[i]`,
and so on. So by the instant `T` seconds have elapsed, press `i` has finished exactly
`floor(T / s[i])` sheets. The presses run in parallel and independently.

The shop has a quota of `P` sheets for the day. Find the **earliest integer time `T`** (in seconds)
at which the total number of finished sheets across all presses is at least `P`. Output that `T`.

This is a monotone-feasibility ("binary search the answer") problem: the total sheet count
`pages(T) = sum_i floor(T / s[i])` is nondecreasing in `T`, so there is a threshold `T*` below which
the quota is unmet and at or above which it is met. The work is to find `T*` without simulating every
second, and to do the feasibility arithmetic in a width that does not silently wrap.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `P`
  (`1 <= n <= 2*10^5`, `1 <= P <= 10^14`). The second line has `n` integers `s[i]`
  (`1 <= s[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the earliest integer time `T` at which
  `sum_i floor(T / s[i]) >= P`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 3`, `P = 10`, periods `s = [2, 3, 5]`, the answer is `10`: at `T = 10` the presses
have finished `5 + 3 + 2 = 10` sheets (just reaching the quota), while at `T = 9` they have only
`4 + 3 + 1 = 8`.

## Background

Two shapes of solution are worth weighing before committing:

- **Second-by-second simulation.** Advance `T` from `0` and recompute `pages(T)` until it reaches
  `P`. Correct and trivially obviously-correct, but `T*` can be on the order of `P * min(s)`, which is
  astronomically large here, so this is only useful as an offline checker on tiny inputs.
- **Binary search on the answer.** Exploit that `pages(T)` is monotone in `T`. Pick an upper bound
  `hi` known to satisfy the quota, then bisect `[0, hi]` for the least feasible `T`. Each feasibility
  test is an `O(n)` sum of `floor` divisions; the whole search is `O(n log hi)`. The open questions
  are the right upper bound and — sharply — the integer width of the feasibility sum and the bounds,
  because both the candidate time and the running sheet count can exceed a 32-bit integer.

## Evaluation settings

Judged on hidden tests covering: tiny inputs cross-checked against the simulator; a single press;
presses with equal periods; coprime periods; quotas that land exactly on a multiple versus strictly
between multiples; and large adversarial cases with `n = 2*10^5`, `P` near `10^14`, and small periods,
so that both the candidate time `T` and the intermediate sheet count `sum_i floor(T/s[i])` blow far
past the 32-bit range. A solution that does feasibility arithmetic in `int` is a silent wrong-answer
on exactly those large cases.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long P;
    if (!(cin >> n >> P)) return 0;
    vector<long long> s(n);
    for (auto &x : s) cin >> x;

    // TODO: binary-search the earliest integer time T with sum_i floor(T / s[i]) >= P,
    //       using an integer width wide enough that the feasibility sum cannot overflow.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
