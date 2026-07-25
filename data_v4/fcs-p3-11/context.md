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

## Evaluation settings

Judged on hidden tests covering: the smallest indices (`N = 0, 1, 2, 3`); a spread of mid-range
indices; many queries with `N` near `10^18`; tiny moduli (`p = 2, 3, 5, 7`); large prime moduli near
`10^18`; and large query counts `T = 2 * 10^5` to enforce the per-query `O(log N)` budget.

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
