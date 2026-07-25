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

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `c[i]`
  (`-10^9 <= c[i] <= 10^9`), whitespace-separated. When `n = 0` there are no stones to print.
- Output (stdout): a single line with the minimum total stamina to cross.
- For `n = 0` the near bank *is* the far bank, so the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `c = [3, 7, 8, 6, 5, 7]` the answer is `16` — land on stones `0`, `2`, `4`
(`3 + 8 + 5`) and leap off from stone `4 = n-2` to the far bank.

## Evaluation settings

Judged on hidden tests covering: `n = 0`; `n = 1` and `n = 2` (boundary leaps); all-non-negative
costs; mixed signs; all-negative costs; adversarial layouts; and large `n = 2*10^5` with `|c[i]|`
near `10^9`.

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
