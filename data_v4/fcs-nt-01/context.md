# Chinese Remainder reconstruction with non-coprime moduli

## Research question

You are given `k` congruences

```
x ≡ r_1 (mod m_1)
x ≡ r_2 (mod m_2)
...
x ≡ r_k (mod m_k)
```

and must find the **smallest non-negative integer `x`** that satisfies *all* of them simultaneously,
or report that **no such `x` exists**. The catch is that the moduli `m_i` are **not assumed to be
pairwise coprime** — they may share common factors, and the system may be **inconsistent** (two
congruences can flatly contradict each other). Textbook Chinese Remainder Theorem only covers the
coprime case; here the contract is the general one, including the inconsistency verdict.

It is guaranteed that, when a solution exists, the least common multiple of all moduli (hence the
unique solution modulo that lcm) fits in a 64-bit signed integer, so the answer can be printed as an
ordinary integer.

## Input / output contract

- Input (stdin): the first token is `k` (`1 <= k <= 10^5`); then `k` lines, each containing two
  integers `r_i m_i` with `1 <= m_i <= 10^9` and `0 <= r_i < m_i`.
- It is guaranteed that `lcm(m_1, ..., m_k) <= 10^18`.
- Output (stdout): a single line — the smallest non-negative `x` satisfying every congruence, or
  `-1` if the system has no solution.
- Time limit: 1 second. Memory: 256 MB.

Example: for the three congruences `x ≡ 2 (mod 6)`, `x ≡ 2 (mod 4)`, `x ≡ 4 (mod 10)` the answer is
`14` (since `14 mod 6 = 2`, `14 mod 4 = 2`, `14 mod 10 = 4`, and no smaller non-negative integer
works).

## Background

Two routes are on the table before committing to one:

- **Plain CRT with modular inverses.** Compute `M = ∏ m_i`, and for each congruence use the inverse of
  `M/m_i` modulo `m_i`. This is the classic formula, but it is *only valid when the moduli are pairwise
  coprime* — the inverse `(M/m_i)^{-1} mod m_i` does not exist when `gcd(M/m_i, m_i) > 1`. The open
  question is what to do when the moduli share factors, and how to detect contradictions.
- **Iterative pairwise merge.** Fold the congruences one at a time, maintaining a single running
  congruence `x ≡ r (mod m)` that is equivalent to all congruences seen so far, and merging in the
  next `(r_i, m_i)` with the extended Euclidean algorithm. The open questions are the exact
  feasibility test, the combined modulus, and keeping the intermediate arithmetic from overflowing.

## Evaluation settings

Judged on hidden tests covering: single congruence; the trivial modulus `m = 1`; coprime systems
(classical CRT); **non-coprime but consistent** systems; **non-coprime contradictory** systems
(answer `-1`); duplicate moduli with equal vs conflicting remainders; many small moduli with a bounded
lcm; and a handful of congruences with moduli near `10^9` so the merged modulus approaches `10^18`
(forcing overflow-safe multiplication). `k` ranges up to `10^5`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int k;
    if (!(cin >> k)) return 0;

    long long r = 0, m = 1; // running solution x ≡ r (mod m)
    bool ok = true;

    for (int i = 0; i < k; i++) {
        long long ri, mi;
        cin >> ri >> mi;
        // TODO: merge the congruence x ≡ ri (mod mi) into (r, m);
        //       set ok = false if the system becomes inconsistent.
    }

    // TODO: print the smallest non-negative solution r, or -1 if !ok.
    cout << (ok ? r : -1) << "\n";
    return 0;
}
```
