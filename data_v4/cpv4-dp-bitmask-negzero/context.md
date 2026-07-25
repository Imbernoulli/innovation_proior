# Parcel-to-slot assignment with optional delivery (max net profit)

## Research question

A depot has `n` parcels and `m` delivery slots. Putting parcel `i` into slot `j` earns a **net
profit** `p[i][j]`, which may be **positive, zero, or negative** (a negative entry means the fuel,
handling, and penalty costs of that pairing outweigh its revenue). Each slot can hold **at most one**
parcel and each parcel goes into **at most one** slot. Crucially, delivery is *optional*: you may
leave any parcel undelivered and any slot empty.

Choose an assignment (a partial matching between parcels and slots) that **maximizes the total net
profit**. Because the empty assignment — deliver nothing — is always allowed, the answer is never
below `0`.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`0 <= n <= 18`, `1 <= m <= 18`). Then follow `n` lines, each with `m` integers; the `j`-th
  integer on line `i` is `p[i][j]` (`-10^9 <= p[i][j] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum achievable total net profit.
- Time limit: 1 second. Memory: 256 MB.

Example: for

```
3 3
5 -2 1
-3 4 0
2 1 6
```

the answer is `15` (parcel 0 -> slot 0 = 5, parcel 1 -> slot 1 = 4, parcel 2 -> slot 2 = 6).

## Evaluation settings

Judged on hidden tests covering: all-positive matrices, matrices with negatives and zeros, the empty
instance (`n = 0`), a single parcel (`n = 1`), an all-negative matrix, the case `n > m` and `n < m`,
and large `n = m = 18` with `|p[i][j]|` near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<vector<long long>> p(n, vector<long long>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) cin >> p[i][j];

    // TODO: compute the maximum achievable total net profit.
    // Deliver nothing is always allowed, so the answer is at least 0.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
