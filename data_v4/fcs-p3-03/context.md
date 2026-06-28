# Counting ordered ways to climb a staircase with a given step set, modulo a prime

## Research question

You are climbing a staircase of `N` steps. In a single move you may climb any number of steps whose
size belongs to a given set `S` of positive integers. You climb until you have ascended exactly `N`
steps; you may never overshoot. Two climbs are **different** if the sequence of move-sizes differs
(order matters — climbing `1` then `2` is distinct from `2` then `1`). Count the number of distinct
climbs and output it **modulo a prime `p`**.

Formally, let `f(N)` be the number of ordered sequences `(t_1, t_2, ..., t_L)` with every `t_j` in
`S` and `t_1 + t_2 + ... + t_L = N`. The empty sequence (`L = 0`) is the unique way to climb `N = 0`.
Output `f(N) mod p`.

This is the ordered-composition / "staircase" counting problem. The headline difficulty is the scale:
`N` can be as large as `10^9`, so any method that touches every step from `1` to `N` is far too slow.

## Input / output contract

- Input (stdin), whitespace-separated:
  - Line 1: three integers `N k p`.
    - `0 <= N <= 10^9` — the number of stairs.
    - `1 <= k <= 100` — the number of listed step sizes.
    - `2 <= p <= 2*10^9`, `p` prime — the modulus.
  - Line 2: `k` integers, the step sizes `s_1, ..., s_k`, each with `1 <= s_i <= 100`.
    The listed values may contain duplicates; the actual step set `S` is the set of distinct values.
- Output (stdout): a single line with `f(N) mod p`.
- Time limit: 2 seconds. Memory: 256 MB.

Let `m = max(S)`. Then `m <= 100`, and the recurrence order is at most `m`.

### Examples

Example 1:

```
10 2 1000000007
1 2
```

Output:

```
89
```

(With `S = {1, 2}` the counts are the Fibonacci numbers `f(0)=1, f(1)=1, f(2)=2, ..., f(10)=89`.)

Example 2:

```
9 1 1000000007
2
```

Output:

```
0
```

(With `S = {2}` only even totals are reachable; `9` is odd, so there are `0` ways.)

Example 3:

```
0 2 998244353
1 5
```

Output:

```
1
```

(The empty climb is the one way to ascend `0` steps.)

## Background

The count satisfies a linear recurrence whose order is the largest allowed step. Two families of
approach are on the table before committing to one:

- **Tabulate / hardcode small cases.** For famous step sets the counts are tidy: `S = {1, 2}` gives
  the Fibonacci numbers, `S = {1, 2, 3}` gives the tribonacci numbers, `S = {2, 3}` gives a
  Padovan-like sequence. The first several values of any fixed `S` form a short, clean table. The
  open question is whether reading the answer out of such a table can possibly survive inputs where
  `N` reaches `10^9`.
- **Linear recurrence advanced by fast exponentiation.** The recurrence
  `f(n) = sum_{s in S} f(n - s)` is a constant-coefficient linear recurrence of order `m = max(S)`.
  Its state can be advanced `N` places at once with matrix exponentiation (or Kitamasa) in
  `O(m^3 log N)` time. The open question is the exact transition matrix, the base values, and the
  boundary handling for `N < m` and `N = 0`.

## Evaluation settings

Judged on hidden tests covering: tiny `N` (`0, 1`) with assorted step sets; `N` just below, equal to,
and just above `m = max(S)` (the recurrence boundary); step sets that miss `1` (so many totals are
unreachable and the answer is `0`); single-step sets (only multiples of one value are reachable);
input lists with duplicate step sizes; the maximum recurrence order `m = 100` with `N = 10^9`; and a
spread of prime moduli including small primes (where many counts collapse to `0`), `998244353`,
`10^9 + 7`, and primes near `2*10^9` (so a product of two residues exceeds 64 bits unless reduced
with care).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long N;   // number of stairs (up to 1e9)
    int k;         // number of listed step sizes
    int p;         // prime modulus
    if (!(cin >> N >> k >> p)) return 0;

    vector<int> raw(k);
    for (auto &x : raw) cin >> x;

    // TODO: count ordered compositions of N using parts from the set {raw[i]}, modulo p.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
