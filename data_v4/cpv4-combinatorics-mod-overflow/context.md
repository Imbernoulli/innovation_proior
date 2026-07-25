# Lattice paths through a checkpoint, counted modulo a prime

## Research question

A robot walks on the integer grid. From a point it may step one unit **right** (`+1` in `x`) or one
unit **up** (`+1` in `y`); it never moves left or down. For each query you are given a **checkpoint**
`(cx, cy)` and a **destination** `(ex, ey)` with `0 <= cx <= ex` and `0 <= cy <= ey`. Count how many
monotone (right/up only) walks start at the origin `(0, 0)`, pass **through** the checkpoint
`(cx, cy)`, and end at `(ex, ey)`. The count can be astronomically large, so report it **modulo a given
prime `M`**.

## Input / output contract

- Input (stdin): the first line has two integers `q` and `M` (`1 <= q <= 10`; `M` is **prime**).
  Then `q` lines follow, each with four integers `cx cy ex ey`
  (`0 <= cx <= ex <= 10^6`, `0 <= cy <= ey <= 10^6`).
  It is guaranteed that `M` is strictly greater than every factorial index that arises, i.e.
  `M > (cx + cy)` and `M > ((ex - cx) + (ey - cy))` for every query; in particular `M` can be as large
  as about `2 * 10^9`.
- Output (stdout): for each query, one line with the number of checkpoint-respecting monotone walks,
  reduced modulo `M`.
- Time limit: 1 second. Memory: 256 MB.

Example:

```
3 998244353
2 1 4 3
0 0 5 5
1 1 2 2
```

produces

```
18
252
4
```

For the first query: walks `(0,0)->(2,1)` number `3`, walks `(2,1)->(4,3)` number `6`,
and `3 * 6 = 18`. For the second the checkpoint is the origin itself, so the first leg contributes `1`
and the answer is just `252`. For the third, `2 * 2 = 4`.

## Evaluation settings

Judged on hidden tests covering: tiny grids checked against a direct path-counting DP; checkpoints at
the origin or at the destination; checkpoints and destinations spread far apart so intermediate values
are large; small prime moduli; and large prime moduli near `2^31`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    ll MOD;
    if (!(cin >> q >> MOD)) return 0;

    // TODO: for each query (cx,cy,ex,ey), compute and print the number of
    //       checkpoint-respecting monotone walks, reduced modulo MOD.

    return 0;
}
```
