# Stamping-press line: earliest time to reach a quota

## Research question

A factory line has `m` stamping presses running in parallel. Press `i` needs a fixed **warm-up
delay** `w[i]` milliseconds before it can emit its *first* part; after that it emits one part every
`c[i]` milliseconds. So by elapsed time `T` (milliseconds, measured from when the line is switched
on) press `i` has produced

- `0` parts if `T < w[i]`, and
- `floor((T - w[i]) / c[i]) + 1` parts if `T >= w[i]` (the `+1` counts the part stamped exactly at
  the moment `w[i]`).

All presses run simultaneously and independently. Given a quota `N`, find the **smallest** time `T`
such that the presses together have produced at least `N` parts. Output that `T`.

## Input / output contract

- Input (stdin): the first line has two integers `m` and `N`
  (`1 <= m <= 10^5`, `0 <= N <= 10^9`). Then `m` lines follow; line `i` has two integers `w[i]` and
  `c[i]` (`0 <= w[i] <= 10^9`, `1 <= c[i] <= 10^9`).
- Output (stdout): a single line with the smallest `T` (milliseconds) at which total production is at
  least `N`. If `N = 0` the answer is `0` (the quota is already met before anything is stamped).
- Time limit: 1 second. Memory: 256 MB.

Example: for `m = 3`, `N = 10` and presses `(w, c)` = `(0, 3), (2, 5), (1, 2)`, the answer is `9`.
At `T = 8` the presses have produced `3 + 2 + 4 = 9` parts (`< 10`); at `T = 9` they have produced
`4 + 2 + 5 = 11` parts (`>= 10`), and no earlier time reaches `10`.

## Evaluation settings

Judged on hidden tests covering: small hand-checkable lines; `N = 0`; `N = 1`; a single press; many
presses with tiny cycle times; and the full-scale `m = 10^5`, `N = 10^9`, `w[i], c[i]` near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m;
    long long N;
    if (!(cin >> m >> N)) return 0;
    vector<long long> w(m), c(m);
    for (int i = 0; i < m; i++) cin >> w[i] >> c[i];

    // TODO: compute the smallest time T with produced(T) >= N, where
    // produced(T) = sum over presses of (T >= w[i] ? (T - w[i]) / c[i] + 1 : 0).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
