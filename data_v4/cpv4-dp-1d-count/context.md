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

## Input / output contract

- Input (stdin): three whitespace-separated integers `n`, `K`, `p`
  (`0 <= n <= 2*10^5`, `1 <= K <= 10^9`, `1 <= p <= 10^9`).
- Output (stdout): a single line with the number of valid wall designs of a length-`n` strip,
  taken modulo `p`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 3`, `K = 3`, `p = 1000000007` the answer is `24`. (The three brick-length layouts
are `1+1+1`, `1+2`, and `2+1`; coloring them under the adjacency rule gives `3*2*2 = 12`, `3*2 = 6`,
and `3*2 = 6`, for `12 + 6 + 6 = 24`.)

## Evaluation settings

Judged on hidden tests covering: `n = 0`, `n = 1`, `K = 1`, `K = 2`, composite and prime moduli
(including `p = 1`), and large `n = 2*10^5` with `K` near `10^9`.

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
