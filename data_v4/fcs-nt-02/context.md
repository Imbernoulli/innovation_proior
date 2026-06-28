# Binomial coefficient modulo an arbitrary integer

## Research question

You are given three integers `n`, `r`, and `m`. Compute the binomial coefficient `C(n, r)` (the
number of ways to choose `r` items out of `n`, also written `nCr`) modulo `m`, and print the result.

The catch is the size and the modulus. `n` and `r` are as large as `10^18`, so `C(n, r)` itself is an
astronomically large integer that cannot be formed explicitly. And `m` is an arbitrary integer up to
`10^6` — it is **not guaranteed to be prime**, and it need not be squarefree. The standard "precompute
factorials and factorial inverses, then `C = n! * inv(r!) * inv((n-r)!)`" recipe assumes a prime
modulus so that every factorial is invertible by Fermat's little theorem. Here that assumption breaks:
modulo a composite `m`, the factorials share prime factors with `m` and are not units, so their inverses
simply do not exist. The problem is to compute the right value anyway.

## Input / output contract

- Input (stdin): a single line (or whitespace-separated tokens) with three integers
  `n r m` where `0 <= r, n <= 10^18` and `1 <= m <= 10^6`.
- Output (stdout): a single line with `C(n, r) mod m`, an integer in `[0, m)`.
- If `r > n` (or `r < 0`), then `C(n, r) = 0`, so the answer is `0 mod m`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for input `10 3 1000` the answer is `120` (since `C(10,3) = 120` and `120 mod 1000 = 120`).
For input `10 3 12` the answer is `0` (since `120 mod 12 = 0`).

## Background

`C(n, r)` modulo a prime `p` with `n` up to `10^18` is a solved, classical task: **Lucas' theorem**
expresses `C(n, r) mod p` as the product of `C(n_i, r_i) mod p` over the base-`p` digits `n_i, r_i` of
`n` and `r`, each small factor computed from precomputed factorials and Fermat inverses. That handles
the large-argument part when the modulus is prime.

The difficulty that defines this problem is that `m` is composite. Two structural facts are in play:

- A factorial like `r!` contains every prime that divides `m`, so `r!` is **not invertible** modulo `m`;
  the prime-modulus inverse trick cannot be applied directly.
- By the Chinese Remainder Theorem, knowing `C(n, r)` modulo each prime-power factor `p^e` of `m`
  determines it modulo `m`. So the composite modulus decomposes into independent prime-power
  subproblems — but each subproblem is itself a binomial coefficient modulo `p^e`, where `p` still
  divides the factorials.

The number of prime-power factors of `m <= 10^6` is tiny, and each prime power is itself at most `10^6`.

## Evaluation settings

Judged on hidden tests covering: prime `m`; prime-power `m` such as `2^e`, `3^e`, `5^e`, `7^e` (these
stress the "factor of `p` inside the factorial" handling, including the sign of full residue blocks);
squarefree composite `m` with several distinct prime factors; general composite `m` up to `10^6`;
boundary `r` values (`r = 0`, `r = n`, `r > n`); `m = 1`; and full-scale `n = r = 10^18`. Answers must
be exact.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    ll n, r, m;
    if (!(cin >> n >> r >> m)) return 0;

    // TODO: compute C(n, r) mod m for n, r up to 1e18 and arbitrary (possibly
    //       composite, non-squarefree) m up to 1e6.
    ll answer = 0;

    cout << answer << "\n";
    return 0;
}
```
