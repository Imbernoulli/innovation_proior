# Prefix sum of a Fibonacci-like recurrence, modulo p

## Research question

Fix a two-term linear recurrence with constant coefficients. Given starting values `f(0) = a`
and `f(1) = b`, and coefficients `c`, `d`, define

```
f(i) = c * f(i-1) + d * f(i-2)   for i >= 2.
```

For a query you are given `a, b, c, d`, an index count `N`, and a modulus `p`, and you must report

```
S(N) = ( f(0) + f(1) + ... + f(N-1) )  mod  p,
```

the sum of the **first `N` terms** of the sequence, reduced modulo `p`. The catch is the scale of
`N`: it can be as large as `10^18`, so the terms cannot be enumerated one by one.

This is the prefix-sum companion of the classic "n-th Fibonacci number mod p" exercise. The plain
n-th-term question is solved by `2x2` matrix exponentiation; here the extra requirement is to carry
the running sum *alongside* the recurrence so the whole answer still comes out in `O(log N)` per
query.

## Input / output contract

- Input (stdin): the first token is `q` (`1 <= q <= 10^5`), the number of independent queries.
  Each of the next `q` lines (whitespace-separated, possibly across lines) holds six integers:
  `a b c d N p` with
  - `-10^9 <= a, b, c, d <= 10^9`,
  - `0 <= N <= 10^18`,
  - `1 <= p <= 2*10^9`.
- Output (stdout): for each query, one line with `S(N) mod p`, a single integer in `[0, p)`.
- Time limit: 2 seconds. Memory: 256 MB.

Conventions: `S(0) = 0` (the empty prefix). When `p = 1` every residue is `0`. Starting values and
coefficients may be negative; reduce them into `[0, p)` before combining, so the printed answer is
always non-negative.

Example:

```
Input
3
1 1 1 1 10 1000000007
2 3 1 0 7 11
5 7 1 1 1 1000

Output
143
9
5
```

The first query is the ordinary Fibonacci sequence `1,1,2,3,5,8,13,21,34,55,...`; the sum of its
first ten terms is `143`. The second has `d = 0` and `c = 1`, so `f(i) = f(i-1)` for `i >= 2`, giving
`2,3,3,3,3,3,3,...`; the first seven terms sum to `2 + 3*6 = 20`, and `20 mod 11 = 9`. The third has
`N = 1`, so the answer is just `f(0) = a = 5`.

## Background

Two shapes of solution are on the table before committing to one:

- **Term-by-term accumulation.** Generate `f(2), f(3), ...` up to index `N-1`, adding each into a
  running sum mod `p`. This is the obvious reading of the statement and is trivially correct, but it
  is `O(N)` per query, which is hopeless once `N` reaches `10^18`.
- **Matrix exponentiation carrying the prefix sum.** Encode the pair `(f(i), f(i-1))` together with
  the running prefix sum in one state vector, and advance the state by a fixed linear map raised to a
  power via fast exponentiation. The open question is the exact augmented state and the exact
  transition matrix — in particular how to make a single matrix power produce the *sum* and not just
  the *term*.

A tempting third path is to special-case small `N`: the first few prefix sums have tidy closed forms
for nice parameter choices, and the worked sample only exercises small `N`. Whether such a table can
stand in for the general algorithm is exactly what the constraints (`N` up to `10^18`) put to the
test.

## Evaluation settings

Judged on hidden tests covering: tiny `N` (`0, 1, 2`), the `p = 1` degenerate case, negative
`a/b/c/d`, the `c = d = 0` and `d = 0` degeneracies, ordinary Fibonacci, and — decisively — many
queries with `N` near `10^18` under both prime and composite moduli up to `2*10^9` (so the running
products need a 128-bit intermediate and the per-query cost must be `O(log N)`).

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
        long long a, b, c, d;
        unsigned long long N;
        long long p;
        cin >> a >> b >> c >> d >> N >> p;

        // TODO: compute S(N) = (f(0)+...+f(N-1)) mod p in O(log N),
        // where f(0)=a, f(1)=b, f(i)=c*f(i-1)+d*f(i-2).
        long long answer = 0;

        cout << answer << "\n";
    }
    return 0;
}
```
