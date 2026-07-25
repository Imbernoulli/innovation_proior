# Collapsing a strip of charged tiles

## Research question

A row of `n` tiles carries integer charges `a[0..n-1]` (a charge may be negative or zero). You
collapse the row by repeatedly **fusing two adjacent tiles**: choosing neighbours with charges `x`
and `y`, you replace them with a single tile of charge `x + y`, and the fusion **releases energy
`x * y`** (which can be negative). After exactly `n - 1` fusions a single tile remains. The order in
which you choose the fusions is yours to pick, and different orders release different total energy.

Output the **maximum total energy** obtainable by collapsing the whole row to one tile. If there are
no tiles (`n = 0`) or only one (`n = 1`), no fusion ever happens and the answer is `0`.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 500`); then `n` integers `a[i]`
  (`-10^6 <= a[i] <= 10^6`), whitespace-separated.
- Output (stdout): a single line with the maximum total released energy.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, -2, 5, -1]` the answer is `-7`. (One optimal order: fuse the `5` and `-1` into
`4` releasing `-5`; fuse that `4` with `-2` into `2` releasing `-8`; fuse the leading `3` with `2`
releasing `6`; total `-5 - 8 + 6 = -7`. Every full collapse of these four tiles releases at least
`-7`.)

## Evaluation settings

Judged on hidden tests covering: all-positive rows, rows mixing negatives and zeros, all-negative
rows, the empty row (`n = 0`) and single tile (`n = 1`), rows where every fusion is forced to lose
energy, and the largest `n = 500` with charges near `10^6`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: compute the maximum total fusion energy to collapse the whole row to one tile
    //       (empty / single row -> 0).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
