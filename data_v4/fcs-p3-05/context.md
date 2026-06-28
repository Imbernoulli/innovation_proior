# Counting balanced bracket sequences of length 2n, modulo a prime

## Research question

A *balanced bracket sequence* of length `2n` is a string of `n` opening brackets `(` and `n`
closing brackets `)` such that every prefix has at least as many `(` as `)` (no prefix ever closes
a bracket that was not opened). For example, for `n = 2` the balanced sequences are `()()` and
`(())`, so there are `2` of them; for `n = 3` there are `5`.

You must answer `q` independent queries. Each query gives an integer `n` and a prime `p`, and asks
for the number of balanced bracket sequences of length `2n`, **taken modulo `p`**. The count itself
is astronomically large for big `n` (it roughly quadruples every step), which is exactly why the
answer is requested modulo a prime rather than as an exact integer.

This is a clean modular-counting problem: the combinatorial object is simple, but `n` is large
enough that the count cannot be stored or enumerated directly, so the answer has to be produced
through modular arithmetic.

## Input / output contract

- Input (stdin):
  - The first line contains a single integer `q` (`1 <= q <= 10^5`), the number of queries.
  - Each of the next `q` lines contains two integers `n` and `p`:
    - `0 <= n <= 10^6`,
    - `p` is a prime with `2n < p <= 2*10^9` (so `p` is guaranteed strictly larger than `2n`).
  - The bound `2n < p` guarantees that every integer `1, 2, ..., 2n` is nonzero modulo `p` and
    therefore invertible modulo `p`; you may rely on this.
- Output (stdout): for each query, a single line with the number of balanced bracket sequences of
  length `2n`, modulo `p`.
- Time limit: 2 seconds. Memory: 256 MB.

The sum of `n` over all queries is not separately bounded beyond the per-query limits, but a
per-query `O(n)` (or `O(n + log p)`) method comfortably fits the limit.

Example:

```
Input
4
2 5
3 7
4 11
5 1000000007

Output
2
5
3
42
```

Here the length-4 count is `2`, the length-6 count is `5`, the length-8 count is `14` (and
`14 mod 11 = 3`), and the length-10 count is `42` (and `42 mod (10^9+7) = 42`).

## Background

Two facts about the object are worth having on the table before committing to a method.

- **The counts have a tidy small pattern.** For `n = 0, 1, 2, 3, 4, 5, 6, 7, ...` the counts are
  `1, 1, 2, 5, 14, 42, 132, 429, ...`. The early values are small and memorable, which makes a
  lookup table superficially attractive for "the cases that appear in examples".
- **There is a product/convolution structure.** The count for length `2(k+1)` can be written by
  splitting on the position where the first opening bracket is matched, which expresses each count
  as a sum of products of two smaller counts. Equivalently, the counts admit a closed form built
  from factorials of `n`, `n+1`, and `2n`. Either route reduces the problem to modular arithmetic
  over the integers `1..2n`, all of which are invertible modulo the given prime `p`.

## Evaluation settings

Judged on hidden tests covering: the smallest cases (`n = 0`, `n = 1`); a spread of small `n` paired
with the *tightest* legal prime `p` just above `2n` (so the result wraps heavily); moderate `n`;
many queries; and **large `n` up to `10^6`** paired with primes such as `10^9 + 7` and `998244353`.
The large-`n` cases are the decisive ones — any method whose correctness is limited to small `n`
will fail them.

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
        long long n, p;
        cin >> n >> p;

        // TODO: compute the number of balanced bracket sequences of length 2n, modulo the prime p.
        long long answer = 0;

        cout << answer << "\n";
    }
    return 0;
}
```
