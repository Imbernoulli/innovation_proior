# Counting 2xN domino tilings modulo p, for enormous N

## Research question

A `2 x N` board is to be tiled completely by `1 x 2` dominoes; each domino may be placed
horizontally (covering two side-by-side cells in one row) or vertically (covering the two cells of a
single column). Let `T(N)` be the number of distinct full tilings. You must report `T(N) mod p`.

The catch is the scale. For tiny boards the counts are a short, tidy sequence — `T(0)=1`, `T(1)=1`,
`T(2)=2`, `T(3)=3`, `T(4)=5`, `T(5)=8` — small enough that one is tempted to read the answer off a
precomputed table. But `N` ranges up to `10^18`, far beyond anything a table or a linear scan can
reach, and the modulus `p` is supplied per query, so the values must be produced by an algorithm that
is both sub-linear in `N` and modular throughout.

## Input / output contract

- Input (stdin):
  - The first line contains an integer `Q` (`1 <= Q <= 10^5`), the number of queries.
  - Each of the next `Q` lines contains two integers `N` and `p`:
    - `0 <= N <= 10^18`
    - `2 <= p <= 10^9` (a prime; the algorithm need not rely on primality)
- Output (stdout): for each query, a single line with `T(N) mod p`.
- Time limit: 2 seconds. Memory: 256 MB.

Example:

Input
```
6
0 1000000007
1 1000000007
2 1000000007
3 1000000007
4 1000000007
5 1000000007
```
Output
```
1
1
2
3
5
8
```

(An empty board, `N = 0`, has exactly one tiling: the empty tiling. Hence `T(0) = 1`.)

## Background

Two structural facts frame the work, but the exact algorithm is left to derive:

- **The counts follow a second-order linear recurrence.** Classifying tilings by how the leftmost
  column is covered yields a relation between `T(N)`, `T(N-1)`, and `T(N-2)`. Establishing that
  relation and its base cases is the first task.
- **`N` is astronomically large.** With `N` up to `10^18`, any method whose running time grows with
  `N` (a table lookup capped at small `N`, or an `O(N)` iteration of the recurrence) is hopeless on
  the hidden inputs. A linear-recurrence value at index `N` can instead be obtained in `O(log N)`
  arithmetic operations by exponentiating the recurrence's transition matrix, with every operation
  carried out modulo `p`.

The open questions to settle before committing: the precise recurrence and its base cases; the `2 x 2`
transition matrix and which of its entries is the wanted value; whether intermediate products of two
residues below `10^9` stay inside 64-bit range; and the corner behaviours at `N = 0`, `N = 1`, and
`p = 1` (if it can arise) versus the stated `p >= 2`.

## Evaluation settings

Judged on hidden tests covering: the smallest boards (`N = 0, 1, 2`); a spread of moderate `N`; many
queries at the extreme `N = 10^18` under assorted moduli (small primes, primes near `10^9`,
`998244353`, `10^9 + 7`); and stress batches with the maximum `Q = 10^5` to exercise the per-query
`O(log N)` budget. A solution that hardcodes small-`N` answers, or iterates the recurrence linearly,
will fail the large-`N` tests outright.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;
    while (q--) {
        long long n, p;
        cin >> n >> p;

        // TODO: compute T(N) mod p, where T counts 2xN domino tilings.
        // N can be up to 1e18, so this must be sub-linear in N.
        long long answer = 0;

        cout << answer << "\n";
    }
    return 0;
}
```
