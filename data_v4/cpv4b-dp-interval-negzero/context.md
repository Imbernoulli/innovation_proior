# Collapsing a strip of charged tiles

## Research question

A row of `n` tiles carries integer charges `a[0..n-1]` (a charge may be negative or zero). You
collapse the row by repeatedly **fusing two adjacent tiles**: choosing neighbours with charges `x`
and `y`, you replace them with a single tile of charge `x + y`, and the fusion **releases energy
`x * y`** (which can be negative). After exactly `n - 1` fusions a single tile remains. The order in
which you choose the fusions is yours to pick, and different orders release different total energy.

Output the **maximum total energy** obtainable by collapsing the whole row to one tile. If there are
no tiles (`n = 0`) or only one (`n = 1`), no fusion ever happens and the answer is `0`.

This is an interval-style dynamic program: the score of collapsing a contiguous block depends only on
that block, and the last fusion always merges two sub-blocks whose combined charges are their charge
sums. The sign of `a[i]` matters everywhere — fusing two negatives releases *positive* energy, so an
all-negative row can have a large positive answer, and because fusion is mandatory the answer for
`n >= 2` can itself be negative. Getting the base case and the sign of the optimum right is the whole
game.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 500`); then `n` integers `a[i]`
  (`-10^6 <= a[i] <= 10^6`), whitespace-separated.
- Output (stdout): a single line with the maximum total released energy.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, -2, 5, -1]` the answer is `-7`. (One optimal order: fuse the `5` and `-1` into
`4` releasing `-5`; fuse that `4` with `-2` into `2` releasing `-8`; fuse the leading `3` with `2`
releasing `6`; total `-5 - 8 + 6 = -7`. Every full collapse of these four tiles releases at least
`-7`.)

## Background

The released energy of a full collapse decomposes by the **last** fusion. Whatever order you use on a
block `[i..j]`, that block is ultimately formed by fusing a left sub-block `[i..k]` with a right
sub-block `[k+1..j]` for some split `k`; at that moment the two surviving tiles carry the charge sums
`S(i,k)` and `S(k+1,j)`, so the final fusion releases `S(i,k) * S(k+1,j)`, on top of whatever each
side released internally. Charge sums never change with fusion order — fusion only adds — so the split
structure is all that matters.

Two routes are on the table before committing:

- **Greedy on fusion energy.** Repeatedly perform the adjacent fusion that releases the most energy
  right now (or, for minimisation, the least), updating charges as you go. `O(n^2)` and easy, but a
  locally best fusion changes its neighbours' charges, so the open question is whether local choices
  can be globally optimal.
- **Interval dynamic programming.** Let `dp[i][j]` be the best total energy to collapse `[i..j]` to
  one tile, and choose the last split. `O(n^3)`; the open questions are the exact recurrence, the
  base case for single tiles, and the sign of the initial optimum when fusions are forced.

## Evaluation settings

Judged on hidden tests covering: all-positive rows, rows mixing negatives and zeros, all-negative
rows (whose answer is typically *positive*), the empty row (`n = 0`) and single tile (`n = 1`), rows
where every fusion is forced to lose energy (answer negative), and the largest `n = 500` with charges
near `10^6` (so both the intermediate products `S * S` and the accumulated answer must hold in 64-bit).

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
