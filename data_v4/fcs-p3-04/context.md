# Counting binary strings with no two adjacent ones, modulo p

## Research question

A binary string is a string over the alphabet `{0, 1}`. Call a binary string *valid* if it contains no
two adjacent `1`s — that is, the substring `11` never appears. For a given length `N`, how many valid
binary strings of length `N` are there? Because that count grows exponentially, the answer is reported
modulo a given integer `p`.

You must answer this for many `(N, p)` pairs, where `N` can be astronomically large (up to `10^18`).

## Input / output contract

- Input (stdin):
  - The first line contains a single integer `T` — the number of test cases (`1 <= T <= 10^5`).
  - Each of the next `T` lines contains two integers `N` and `p`:
    - `0 <= N <= 10^18` — the string length,
    - `1 <= p <= 10^18` — the modulus. (When `p = 1`, every count is `0 mod 1`.)
- Output (stdout): for each test case, on its own line, the number of valid binary strings of length `N`,
  taken modulo `p`.
- Time limit: 2 seconds. Memory: 256 MB.

### Sample

Input:

```
6
0 1000000007
1 1000000007
2 1000000007
3 1000000007
4 1000000007
5 1000000007
```

Output:

```
1
2
3
5
8
```

(Length 0: only the empty string, count 1. Length 1: `0` and `1`, count 2. Length 2: `00`, `01`, `10`
are valid but `11` is not, count 3. The fifth line, `N = 4`, gives 8.) Notice the answers
`1, 2, 3, 5, 8, ...` — the counts for consecutive lengths form a familiar sequence.

## Background

Let `f(N)` be the number of valid strings of length `N`. The first several values are
`f(0)=1, f(1)=2, f(2)=3, f(3)=5, f(4)=8, f(5)=13, ...`. Two observations are on the table before
committing to an algorithm:

- **The small values are extremely tidy.** They are consecutive Fibonacci numbers, so for small `N`
  a hand-written table of constants would reproduce the samples perfectly. The open question is whether
  a table can possibly cover the input range.
- **A length recurrence is plausible.** Splitting on the last character of a valid string suggests
  `f(N)` is determined by a fixed number of previous values. The open questions are the exact
  recurrence and how to evaluate it when `N` is as large as `10^18` (so iterating `N` steps is far too
  slow) and `p` is as large as `10^18` (so intermediate products overflow 64-bit arithmetic).

## Evaluation settings

Judged on hidden tests covering: the small lengths `N = 0, 1, 2` (degenerate bases); a spread of small
and medium lengths; and — decisively — many cases with `N` near `10^18`. Moduli range over `p = 1`
(answer always `0`), tiny primes and composites, `10^9`-scale primes, and full-range moduli near
`10^18` where a 64-bit product of two reduced residues would overflow without a wider intermediate.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long u64;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    if (!(cin >> T)) return 0;
    while (T--) {
        u64 n, p;
        cin >> n >> p;

        // TODO: compute the number of valid binary strings of length n, modulo p.
        u64 ans = 0;

        cout << ans << "\n";
    }
    return 0;
}
```
