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

## Evaluation settings

Judged on hidden tests covering: tiny indices `n in {0,1,2}`; small indices a brute term-by-term
loop can also reach (used to cross-check correctness); large indices up to `n = 10^18` with both
small moduli (`p = 10^9+7`, `998244353`) and large moduli near `4*10^18`; seeds given larger than
`p`; small moduli including `p = 1`; and many queries (`T` up to `10^5`) to test performance under
the time limit.

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
        unsigned long long answer = 0;

        cout << answer << "\n";
    }
    return 0;
}
```
