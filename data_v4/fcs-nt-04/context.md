# Integer factorization of 64-bit numbers

## Research question

You are given `q` independent queries. Each query is a single integer `n` with
`1 <= n <= 10^18`. For each `n`, output its **prime factorization**: the distinct prime
factors in increasing order, each with its multiplicity (exponent). The number `1` has no
prime factors.

The difficulty is entirely in the size of `n`. A single number can be a large prime
(no small factors to catch), a product of two primes both near `10^9` (the classic hard
case), a high power of a single prime, or a product of many primes of mixed sizes. The
solver must factor *every* such shape within the time limit.

## Input / output contract

- Input (stdin): the first token is `q` (`1 <= q <= 500`); then `q` integers `n`
  (`1 <= n <= 10^18`), whitespace-separated.
- Output (stdout): for each query, one line of the form
  `n: p1^e1 p2^e2 ... pk^ek`
  where `p1 < p2 < ... < pk` are the distinct primes dividing `n` and `ei >= 1` are their
  exponents. For `n = 1`, print exactly `1:` (the value, a colon, and nothing after).
  There is exactly one space between the colon and the first factor and between successive
  `p^e` tokens; there is no trailing space.
- Time limit: 2 seconds. Memory: 256 MB.

Example:

```
4
12
1
1000000007
1000000000000000000
```

produces

```
12: 2^2 3^1
1:
1000000007: 1000000007^1
1000000000000000000: 2^18 5^18
```

(`12 = 2^2 * 3`; `1` has no factors; `1000000007` is prime; `10^18 = 2^18 * 5^18`.)

## Background

Two routes are on the table before committing to one:

- **Trial division.** Test divisibility by every integer (or every prime) up to `sqrt(n)`,
  peeling factors as they are found. It is obviously correct and trivial to write; the open
  question is whether it is fast enough when `n` is close to `10^18` and the smallest prime
  factor is itself close to `10^9`.
- **Randomized factorization.** Use a probabilistic primality test to decide when a part is
  already prime, and a randomized factor-splitting routine (the rho method) to break a
  composite into two smaller pieces, recursing. This avoids ever scanning up to `sqrt(n)`;
  the open questions are which primality witnesses make the test exact at this scale, how to
  multiply two near-`10^18` numbers modulo a third without overflow, and how to keep the
  splitting routine from thrashing on adversarial inputs (perfect powers, twin large primes).

## Evaluation settings

Judged on hidden tests covering: `n = 1`; small `n`; large primes near `10^18`; semiprimes
`p*q` with `p, q` both near `10^9` (the worst case for any factoring method here); high
prime powers such as `2^59`, `3^37`, `5^18`; products of many small and medium primes; and a
full batch of `q = 500` maximal hard semiprimes to stress the time limit. Every factor in the
output must be genuinely prime, listed once, in increasing order, with the correct exponent,
and the product of `pi^ei` must equal `n`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    if (!(cin >> q)) return 0;
    while (q--) {
        u64 n;
        cin >> n;

        // TODO: factor n into primes with multiplicities and print
        //       "n: p1^e1 p2^e2 ... pk^ek"  (or "n:" when n == 1).

        cout << n << ":" << "\n";
    }
    return 0;
}
```
