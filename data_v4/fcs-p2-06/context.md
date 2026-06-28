# Matrix-chain multiplication: minimum scalar multiplications

## Research question

You are given a chain of `n` matrices `A[1], A[2], ..., A[n]` that are to be multiplied together in
the order `A[1] * A[2] * ... * A[n]`. Matrix `A[i]` has dimension `p[i-1] x p[i]`, so the chain is
described by the `n+1` dimension values `p[0], p[1], ..., p[n]` (consecutive matrices are
conformable by construction). Matrix multiplication is associative, so the *result* of the product
is fixed, but the **number of scalar multiplications** depends on how you parenthesize: multiplying a
`a x b` matrix by a `b x c` matrix with the standard algorithm costs `a * b * c` scalar
multiplications. Choose the parenthesization that **minimizes the total number of scalar
multiplications**, and output that minimum.

This is the matrix-chain ordering problem. It is a canonical setting where a locally attractive
heuristic — "at each step, multiply the adjacent pair that is cheapest right now" — looks reasonable
but is not optimal, because the dimension a product exposes to its neighbours changes the cost of
every later multiplication.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 300`), the number of matrices. Then `n+1`
  integers `p[0], p[1], ..., p[n]` (`1 <= p[i] <= 1000`), whitespace-separated, giving the matrix
  dimensions.
- Output (stdout): a single line with the minimum number of scalar multiplications needed to compute
  the product `A[1] * ... * A[n]`. With `0` or `1` matrices there is nothing to multiply, so the
  answer is `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 3` and `p = [10, 1, 100, 10]` (matrices `10x1`, `1x100`, `100x10`) the answer is
`1100`: parenthesizing as `A[1] * (A[2] * A[3])` costs `1*100*10 + 10*1*10 = 1000 + 100 = 1100`,
whereas `(A[1] * A[2]) * A[3]` costs `10*1*100 + 10*100*10 = 1000 + 10000 = 11000`.

## Background

The order of pairwise multiplications is a parenthesization of the chain, i.e. a full binary tree
over `A[1..n]`. The number of parenthesizations grows like the Catalan numbers, so brute-forcing all
of them is hopeless beyond tiny `n`. Two approaches are on the table before committing:

- **Greedy by local cost.** Repeatedly find the adjacent pair whose immediate multiplication is
  cheapest, perform that multiply, and collapse the two matrices into one (its dimensions become the
  outer dimensions of the product). Repeat until one matrix remains. This is `O(n^2)` and short to
  write; the open question is whether always taking the locally cheapest merge is globally optimal.
- **Interval dynamic programming.** For every contiguous sub-chain `A[i..j]`, compute the minimum
  cost to reduce it to a single matrix by trying every place to make the last (outermost) split.
  This is `O(n^3)`; the open question is the exact recurrence, the split bounds, and the data type.

## Evaluation settings

Judged on hidden tests covering: the empty chain (`n = 0`) and single matrix (`n = 1`); two- and
three-matrix chains where ordering does or does not matter; chains explicitly engineered to fool the
"cheapest adjacent pair first" heuristic (alternating tiny and huge dimensions); and large chains
with `n = 300` and dimensions near `1000`, where the total cost can reach roughly `3 * 10^11` and so
exceeds the range of a 32-bit integer.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> p(n + 1);
    for (auto &x : p) cin >> x;            // p[0..n]: matrix i is p[i-1] x p[i]

    // TODO: compute the minimum number of scalar multiplications to evaluate
    // TODO: the product A[1] * ... * A[n] over all parenthesizations.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
