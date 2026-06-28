# The N-th Pell number modulo a prime

## Research question

The **Pell numbers** are defined by the linear recurrence

```
P(0) = 0,
P(1) = 1,
P(n) = 2 * P(n-1) + P(n-2)   for n >= 2.
```

The first few terms are `0, 1, 2, 5, 12, 29, 70, 169, 408, 985, 2378, ...`.

Given an index `N` and a prime modulus `p`, compute `P(N) mod p`. The catch is the range of `N`:
the index can be as large as `10^18`, so the term `P(N)` itself is an astronomically large integer
(it grows like `(1 + sqrt 2)^N`) and only its residue modulo `p` is ever asked for. The number of
queries can also be large, so each query must be answered in time logarithmic in `N`.

This is the canonical "evaluate a linear recurrence at a gigantic index under a modulus" task. The
honest difficulty is entirely in the index range: the small-index values form a short, tidy,
*memorable* table, which makes a lookup look attractive — but the hidden evaluation indices live far
out of any table's reach.

## Input / output contract

- Input (stdin):
  - The first token is `T`, the number of queries (`1 <= T <= 2 * 10^5`).
  - Each of the next `T` lines contains two integers `N` and `p`, separated by whitespace:
    - `0 <= N <= 10^18`,
    - `2 <= p <= 10^18`, and `p` is prime.
- Output (stdout): for each query, a single line containing `P(N) mod p`.
- Time limit: 2 seconds. Memory: 256 MB.

### Sample input

```
8
0 1000000007
1 1000000007
2 1000000007
3 1000000007
6 1000000007
10 1000000007
1000000000000000000 1000000007
1000000000000000000 998244353
```

### Sample output

```
0
1
2
5
70
2378
3540480
425552547
```

(The first six lines are just the small Pell numbers `0, 1, 2, 5, 70, 2378` read off the table; the
last two are `P(10^18)` reduced modulo two different primes — values no table contains.)

## Background

Two broad strategies are on the table before committing to one:

- **Tabulate the small terms.** The early Pell numbers are short and easy to list, and many queries in
  any informal test set use small `N`. One could precompute `P(0..K)` for some modest `K` and answer
  by lookup. The open question is what happens when `N` exceeds `K` — and whether the evaluation
  indices stay inside any feasible `K`.
- **Logarithmic-time recurrence evaluation.** Either exponentiate the `2x2` companion matrix
  `M = [[2, 1], [1, 0]]` (so that `M^n = [[P(n+1), P(n)], [P(n), P(n-1)]]`), or use the equivalent
  *fast-doubling* identities, to jump from index `k` to index `2k`/`2k+1` directly. This is
  `O(log N)` per query. The open questions are the exact doubling identities and the modular
  arithmetic needed to multiply two residues that are each near `10^18` without overflow.

## Evaluation settings

Judged on hidden tests covering: the smallest indices (`N = 0, 1, 2, 3`); a spread of mid-range
indices; **many queries with `N` near `10^18`** (well outside any precomputable table); tiny moduli
(`p = 2, 3, 5, 7`, where residues collapse to a few values); large prime moduli near `10^18` (so two
residues multiplied together overflow 64 bits and need 128-bit intermediates); and large query counts
`T = 2 * 10^5` to enforce the per-query `O(log N)` budget.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    if (!(cin >> T)) return 0;
    while (T--) {
        unsigned long long N, p;   // 0 <= N <= 1e18 ; 2 <= p <= 1e18, prime
        cin >> N >> p;

        // TODO: compute P(N) mod p, where P(0)=0, P(1)=1, P(n)=2P(n-1)+P(n-2),
        //       in O(log N) time per query (N can be up to 1e18).
        unsigned long long answer = 0;

        cout << answer << "\n";
    }
    return 0;
}
```
