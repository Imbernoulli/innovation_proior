# Tribonacci term modulo p with astronomically large index

## Research question

A *tribonacci-style* sequence is fixed by three seed values `f(0), f(1), f(2)` and the linear
recurrence

```
f(k) = f(k-1) + f(k-2) + f(k-3)      for k >= 3.
```

Given the three seeds, a modulus `p`, and an index `n`, output `f(n) mod p`. The catch is the scale
of `n`: it can be as large as `10^18`, so the value `f(n)` itself is an enormous number with on the
order of `10^17` digits and can never be materialized — only its residue modulo `p` is asked for, and
only that residue is tractable.

This is the canonical setting where a linear recurrence with constant coefficients must be evaluated
at an index far beyond what any term-by-term iteration could reach in time. It is the kind of kernel
that appears inside counting problems, tiling/word-counting automata, and any place a fixed-order
linear recurrence is queried at a huge step count.

## Input / output contract

- Input (stdin): the first token is `T`, the number of independent queries (`1 <= T <= 10^5`). Each
  of the next `T` lines contains five integers separated by whitespace:
  `n p f0 f1 f2`, where
  - `0 <= n <= 10^18` is the index to evaluate,
  - `2 <= p <= 4*10^18` is the modulus,
  - `0 <= f0, f1, f2 <= 10^18` are the seeds `f(0), f(1), f(2)`.
- Output (stdout): for each query, a single line with `f(n) mod p`.
- Time limit: 2 seconds. Memory: 256 MB.

The seeds are given as raw integers and may already exceed `p`; reduce them modulo `p` before use, so
e.g. `f(0) mod p` is the correct answer when `n = 0`.

### Example

Input:

```
4
0 1000000007 1 1 1
5 1000000007 1 1 1
6 1000000007 1 1 1
3 7 0 1 1
```

Output:

```
1
9
17
2
```

Explanation. With seeds `1, 1, 1`: `f(3)=1+1+1=3`, `f(4)=3+1+1=5`, `f(5)=5+3+1=9`,
`f(6)=9+5+3=17`. For the last query, seeds `0, 1, 1` give `f(3)=1+1+0=2`, and `2 mod 7 = 2`.

## Background

There are two visible families of approach.

- **Iterate the recurrence.** Carry the last three values and step forward `f(k) = f(k-1)+f(k-2)+f(k-3) mod p`
  until reaching index `n`. This is `O(n)` per query and exact. It is trivially correct but only
  finishes when `n` is small; at `n = 10^18` it cannot complete.
- **Exponentiate the transition.** A fixed-order linear recurrence advances by a constant linear map,
  so `t` steps are one matrix raised to the `t`-th power. With a `3x3` companion-style matrix and
  fast exponentiation, the cost is `O(3^3 * log n)` per query, which stays tiny even at `n = 10^18`.
  The open questions are the exact transition matrix, the orientation of the state vector, the base
  cases for `n < 3`, and the arithmetic needed so that `p` near `4*10^18` does not overflow during
  the modular multiplications.

## Evaluation settings

Judged on hidden tests covering: tiny indices `n in {0,1,2}` (pure base cases, no stepping); small
indices a brute term-by-term loop can also reach (used to differentially pin correctness); large
indices up to `n = 10^18` with both small moduli (`p = 10^9+7`, `998244353`) and large moduli near
`4*10^18` (so a 64-bit product overflows and a wider intermediate is required); seeds given larger
than `p` (must be reduced first); `p = 1`-adjacent small moduli where every residue collapses; and
many queries (`T` up to `10^5`) to exercise the per-query `log n` budget under the time limit.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    if (!(cin >> T)) return 0;
    while (T--) {
        unsigned long long n, p, f0, f1, f2;
        cin >> n >> p >> f0 >> f1 >> f2;

        // TODO: compute f(n) mod p for the recurrence
        //   f(k) = f(k-1) + f(k-2) + f(k-3), seeds f(0)=f0, f(1)=f1, f(2)=f2.
        // n can be up to 1e18, so term-by-term iteration is not viable; p can be
        // near 4e18, so a 64-bit product of two residues overflows.
        unsigned long long answer = 0;

        cout << answer << "\n";
    }
    return 0;
}
```
