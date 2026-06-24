# Lattice paths through a checkpoint, counted modulo a prime

## Research question

A robot walks on the integer grid. From a point it may step one unit **right** (`+1` in `x`) or one
unit **up** (`+1` in `y`); it never moves left or down. For each query you are given a **checkpoint**
`(cx, cy)` and a **destination** `(ex, ey)` with `0 <= cx <= ex` and `0 <= cy <= ey`. Count how many
monotone (right/up only) walks start at the origin `(0, 0)`, pass **through** the checkpoint
`(cx, cy)`, and end at `(ex, ey)`. The count can be astronomically large, so report it **modulo a given
prime `M`**.

A walk through a checkpoint factors cleanly: the number of monotone walks `(0,0) -> (ex,ey)` that visit
`(cx,cy)` equals (walks `(0,0) -> (cx,cy)`) times (walks `(cx,cy) -> (ex,ey)`), because the two legs are
independent and the checkpoint is the unique join. Each leg is a single binomial coefficient. The whole
task is therefore "binomials modulo a prime, then one product" — but the modulus is large and so are the
factorial arguments, which is exactly where a 32-bit integer quietly destroys the answer.

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

For the first query: walks `(0,0)->(2,1)` number `C(3,2)=3`, walks `(2,1)->(4,3)` need `2` rights and
`2` ups so number `C(4,2)=6`, and `3 * 6 = 18`. For the second the checkpoint is the origin itself, so
the first leg contributes `C(0,0)=1` and the answer is just `C(10,5)=252`. For the third,
`C(2,1) * C(2,1) = 2 * 2 = 4`.

## Background

The number of monotone right/up walks between two grid points depends only on how many rights and ups
are needed: a walk from `(ax,ay)` to `(bx,by)` is any arrangement of `R = bx-ax` rights among
`R + U` steps (`U = by-ay` ups), so the count is the binomial coefficient `C(R+U, R)`. Reducing a
binomial modulo a prime `M` larger than its top argument is standard: precompute factorials
`fact[i] = i! mod M`, compute one modular inverse `inv_fact[N] = fact[N]^(M-2) mod M` by Fermat's little
theorem, fold the rest down with `inv_fact[i-1] = inv_fact[i] * i mod M`, and read off
`C(n,k) = fact[n] * inv_fact[k] * inv_fact[n-k] mod M`.

The arithmetic hazard is the modular multiply itself. Two residues each below `M` can be close to
`2 * 10^9`; their product is close to `4 * 10^18`, which overflows a 32-bit integer by nine orders of
magnitude but fits comfortably inside a signed 64-bit integer (max about `9.2 * 10^18`). Every
multiplication on the path — building `fact`, the fast-exponentiation in the inverse, folding
`inv_fact`, and the final product of the two legs — must hold the intermediate product in 64 bits before
the `% M`.

## Evaluation settings

Judged on hidden tests covering: tiny grids checked against a direct path-counting DP; checkpoints at
the origin or at the destination (a leg of length zero, `C(0,0)=1`); both legs large so each binomial is
a near-`M` residue; small prime moduli where reductions actually fire on small numbers; and large prime
moduli near `2^31` where a 32-bit multiply silently overflows and a 64-bit one does not.

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

    // TODO: for each query (cx,cy,ex,ey), output
    //       C(cx+cy, cx) * C((ex-cx)+(ey-cy), ex-cx)  modulo MOD,
    //       using factorials / inverse factorials and 64-bit modular multiplication.

    return 0;
}
```
