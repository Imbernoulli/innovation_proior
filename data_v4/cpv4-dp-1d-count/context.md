# Counting colored brick walls (1D tilings with an adjacency color rule)

## Research question

A decorator must cover a `1 x n` strip of wall with bricks laid end to end. Each brick has length
either `1` or `2` (measured along the strip), and every brick is painted in exactly one of `K`
available colors. The only aesthetic rule is **no two bricks that touch may share a color**: along
the strip, every brick must differ in color from the brick immediately before it. Two wall designs
are considered different if their sequence of (brick length, brick color) differs in any position.

Count **how many distinct valid wall designs exist** for a strip of length `n`. The number explodes
quickly, so report it **modulo `p`**. The empty strip (`n = 0`) has exactly one design — the design
that places no bricks.

This is a one-dimensional counting DP: a tiling-by-1-and-2 recurrence (the Fibonacci skeleton)
married to a per-tile color factor. The whole difficulty is fusing the two without **double-counting**
the colorings and without an **off-by-one** at the very first brick, where the "differ from the
previous brick" rule has no previous brick to differ from — exactly where a naive implementation
goes subtly wrong.

## Input / output contract

- Input (stdin): three whitespace-separated integers `n`, `K`, `p`
  (`0 <= n <= 2*10^5`, `1 <= K <= 10^9`, `1 <= p <= 10^9`).
- Output (stdout): a single line with the number of valid wall designs of a length-`n` strip,
  taken modulo `p`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 3`, `K = 3`, `p = 1000000007` the answer is `24`. (The three brick-length layouts
are `1+1+1`, `1+2`, and `2+1`; coloring them under the adjacency rule gives `3*2*2 = 12`, `3*2 = 6`,
and `3*2 = 6`, for `12 + 6 + 6 = 24`.)

## Background

The set of ways to cut a length-`n` strip into bricks of length `1` and `2` — ignoring color — is the
classic Fibonacci count: a strip of length `i` ends in either a length-`1` brick (preceded by a
length-`i-1` strip) or a length-`2` brick (preceded by a length-`i-2` strip). Color multiplies a
factor onto every brick, but the adjacency rule couples each brick to its predecessor, so the factor
is not uniform: the very first brick on the strip may be any of `K` colors, while every later brick
may be any color **except** the one immediately to its left, i.e. `K - 1` choices. Two routes are on
the table before committing to one:

- **Enumerate layouts, then color each.** List every length-`1`/`2` composition of `n` (there are a
  Fibonacci number of them), and for a layout with `t` bricks multiply `K * (K-1)^{t-1}`. This is
  obviously correct but the number of layouts is exponential in `n`, so it only works for tiny `n`.
- **Linear counting DP.** Carry one running count `g[i]` = number of valid colored designs of a
  length-`i` strip, and extend by one brick at a time, attaching the right color factor as the new
  brick is laid. This is `O(n)`; the open question is the exact recurrence and, above all, getting
  the first-brick `K` versus later-brick `K-1` factor attached to the correct predecessor.

## Evaluation settings

Judged on hidden tests covering: `n = 0` (answer `1`), `n = 1` (answer `K mod p`), `K = 1` (only the
all-length-`>=2`... actually only `n in {0,1}` admit a design, every longer strip forces two touching
same-colored bricks somewhere, so the count collapses to `0` for `n >= 2`), `K = 2`, composite and
prime moduli (including `p = 1` where every answer is `0`), and large `n = 2*10^5` with `K` near
`10^9` (so intermediate products `~10^18` must be reduced before they overflow 64-bit arithmetic).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long n, K, MOD;
    if (!(cin >> n >> K >> MOD)) return 0;

    // TODO: count valid colored 1-and-2 brick designs of a length-n strip, modulo MOD.
    // Bricks of length 1 or 2; each brick one of K colors; touching bricks differ in color;
    // the empty strip (n = 0) counts as one design.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
