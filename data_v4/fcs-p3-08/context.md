# Counting monotone lattice paths on a grid, modulo a prime

## Research question

A robot stands at the lower-left corner `(0, 0)` of an integer grid and wants to reach a target
corner `(a, b)`. At every step it may move **one unit right** (increasing the `x`-coordinate by 1)
or **one unit up** (increasing the `y`-coordinate by 1). It never moves left or down. Such a route is
a *monotone lattice path*.

For a single target `(a, b)`, how many distinct monotone lattice paths lead from `(0, 0)` to
`(a, b)`? The count grows astronomically — for `(a, b) = (10^6, 10^6)` it is a number with hundreds of
thousands of digits — so the answer is required **modulo the prime `p = 1 000 000 007`**.

You must answer several independent targets in one run.

## Input / output contract

- Input (stdin):
  - The first token is `q` (`1 <= q <= 2*10^5`), the number of queries.
  - Each of the next `q` lines contains two integers `a` and `b`
    (`0 <= a, b <= 10^6`), one target corner per line.
- Output (stdout): `q` lines. The `i`-th line is the number of monotone lattice paths from `(0, 0)`
  to the `i`-th target `(a, b)`, taken modulo `p = 1 000 000 007`.
- It is guaranteed that the sum of coordinates of any single query satisfies `a + b <= 2*10^6`.
- Time limit: 2 seconds. Memory limit: 256 MB.

### Example

Input:

```
4
0 0
2 3
5 5
1000000 1000000
```

Output:

```
1
10
252
192151600
```

The empty target `(0, 0)` has exactly one path (stand still). For `(2, 3)` there are
`C(5, 2) = 10` paths. For `(5, 5)` there are `C(10, 5) = 252`. The last line is
`C(2 000 000, 1 000 000) mod 1 000 000 007`.

## Background

A monotone path from `(0, 0)` to `(a, b)` consists of exactly `a` right-steps and `b` up-steps in
some order, so it is an arrangement of a multiset of `a` R's and `b` U's. The number of such
arrangements is the binomial coefficient

```
C(a + b, a) = (a + b)! / (a! * b!).
```

So the entire problem reduces to evaluating one binomial coefficient per query, modulo a prime.
Computing binomial coefficients modulo a prime is a classical topic with more than one standard
technique in the literature; picking and correctly implementing one for the constraints below is
part of the task.

## Evaluation settings

Judged on hidden tests covering: the corner targets `(0, 0)`, `(0, k)`, `(k, 0)` (all of which have
exactly one path); small square and rectangular grids; many queries (`q` near `2*10^5`); and large
targets with `a` and `b` near `10^6`, so `a + b` reaches `2*10^6`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;
    vector<int> as(q), bs(q);
    for (int i = 0; i < q; i++) cin >> as[i] >> bs[i];

    // TODO: for each query output the number of monotone lattice paths
    // from (0,0) to (as[i], bs[i]) modulo MOD, i.e. C(as[i]+bs[i], as[i]) mod MOD.

    return 0;
}
```
