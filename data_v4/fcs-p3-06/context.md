# Derangements of n elements, modulo a prime

## Research question

A *derangement* of `{1, 2, ..., n}` is a permutation that leaves no element in its original
position: there is no index `i` with `perm[i] = i`. Let `D(n)` be the number of derangements of `n`
elements. By convention `D(0) = 1` (the empty permutation deranges everything vacuously) and
`D(1) = 0` (a single element can only map to itself).

You are given a prime modulus `p` and a list of queries. For each query value `n`, output
`D(n) mod p`. The numbers `D(n)` grow super-exponentially (roughly `n!/e`), so they are reported
modulo `p` to keep them in range.

## Input / output contract

- Input (stdin):
  - The first line contains two integers `T` and `p`: the number of queries and the prime modulus
    (`1 <= T <= 10^5`, `2 <= p <= 2^31 - 1`, `p` prime).
  - The second block contains `T` integers `n_1, n_2, ..., n_T`, whitespace-separated (they may span
    one or several lines). Each `n_i` satisfies `0 <= n_i <= 10^7`.
- Output (stdout): `T` lines; line `i` is `D(n_i) mod p`.
- Time limit: 2 seconds. Memory: 256 MB.

### Sample

Input:

```
8 1000000007
0 1 2 3 4 5 6 7
```

Output:

```
1
0
1
2
9
44
265
1854
```

The first few derangement counts are `D(0..7) = 1, 0, 1, 2, 9, 44, 265, 1854`. (Modulo `10^9 + 7`
none of these small values change.)

## Background

Two facts about derangements are worth having on the table before committing to an algorithm.

- **The small values form a short, very tidy-looking sequence.** `1, 0, 1, 2, 9, 44, 265, 1854,
  14833, 133496, ...`. For tiny `n` these are the kind of constants one is tempted to drop straight
  into a lookup table. The sample only exercises `n <= 7`.

- **There is a clean linear recurrence.** Counting derangements by where element `n` goes yields
  `D(n) = (n - 1) * (D(n - 1) + D(n - 2))` for `n >= 2`, with `D(0) = 1`, `D(1) = 0`. There is also
  the inclusion-exclusion closed form `D(n) = n! * sum_{k=0}^{n} (-1)^k / k!`, but evaluating that
  modulo a prime requires modular inverses of factorials, whereas the recurrence needs only
  additions and multiplications.

The tension the problem sets up is precisely between these two facts: the small cases look
hardcodable, but the query values range all the way to `n = 10^7`, far past any prefix one could
store.

## Evaluation settings

Judged on hidden tests covering: the tiny regime (`n <= 7`, matching the sample); moderate `n` in
the hundreds to thousands; the boundary cases `n = 0` and `n = 1`; many queries (`T` up to `10^5`)
with a single shared prime `p`; small primes such as `p = 2, 3, 5` where the answer is heavily
reduced; and large `n` up to `10^7` (so any algorithm slower than near-linear, or any finite lookup
table, fails). Correctness is an exact match on every line.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    long long p;
    if (!(cin >> t >> p)) return 0;

    vector<long long> ns(t);
    for (int i = 0; i < t; i++) cin >> ns[i];

    // TODO: for each query n_i, compute D(n_i) mod p, where D is the number of
    // derangements of n_i elements. Print one answer per line.

    return 0;
}
```
