# Unambiguous baselines: build a perfect ruler of n marks

## Research question

A radio-interferometer technician is bolting `n` antennas onto a single straight rail. Only the
*spacings* between antennas matter physically: each unordered pair of antennas forms a baseline whose
length is the distance between them, and the correlator can only disentangle the sky if **every
baseline length is different** — two pairs that happen to sit the same distance apart produce an
ambiguous measurement that corrupts the image.

Concretely: place `n` marks at integer coordinates `0 = x[0] < x[1] < ... < x[n-1] <= M` on the rail
(the first antenna is the origin by convention). The mark set must be a **perfect ruler**, also called
a **Sidon set**: all `n*(n-1)/2` pairwise differences `x[j] - x[i]` (for `i < j`) must be
**pairwise distinct**. The rail has a hard length budget `M = 8*n*n`, and you must place all `n` marks
within `[0, M]`. Output any one valid placement.

Equivalently in interval language: the marks cut the segment `[0, x[n-1]]` so that the set of lengths
of *all* sub-intervals `[x[i], x[j]]` contains no repeats. This is the canonical construction problem
where a wrong builder can look perfect on tiny inputs and silently fail once the scale grows.

## Input / output contract

- Input (stdin): a single integer `n` with `1 <= n <= 2*10^5`.
- Output (stdout): `n` integers on one line, space-separated: the mark coordinates
  `x[0] x[1] ... x[n-1]`. They must satisfy
  - `x[0] = 0`,
  - strictly increasing: `x[0] < x[1] < ... < x[n-1]`,
  - `x[n-1] <= M` where `M = 8*n*n`,
  - all pairwise differences `x[j] - x[i]` (`i < j`) distinct.
- This is a **special-judge** problem: any placement meeting all four conditions is accepted; there is
  no unique expected answer. Coordinates can exceed `2^31` (up to `~8*10^10` at the largest `n`), so
  print 64-bit values.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 6` one valid answer is `0 15 32 44 58 74`. Its 15 pairwise differences are
`12 14 15 16 17 26 29 30 32 42 43 44 58 59 74` — all distinct — and the largest mark `74 <= 8*36 = 288`.

## Background

Two ideas suggest themselves before committing:

- **Doubling / binary spacing.** Put mark `k` at `2^k`. Because every integer has a unique binary
  representation, all subset sums differ, hence all pairwise differences differ: this *is* a genuine
  Sidon set, and it is two lines to write. The open question is whether `2^(n-1)` ever exceeds the
  length budget `M = 8*n*n`.
- **An algebraic Sidon construction.** The Erdős–Turán / Singer family builds, from a prime `p`, a
  Sidon set of size `p` whose elements live in `[0, ~2*p^2)`. The open question is the exact formula,
  why it is collision-free, and how to hit an arbitrary target size `n` (not just primes) inside `M`.

The tension is precisely the trap this task is about: a construction that is *correct on the property*
but *wrong on the budget* passes every small hand-check and dies at scale.

## Evaluation settings

Judged on hidden tests covering: the degenerate `n = 1` (a single mark, no pair); the smallest real
cases `n = 2, 3`; the boundary region around `n = 11..13` where naive exponential spacing first
overflows the budget; composite `n` (so a primality-blind algebraic builder is exposed); and large
`n` up to `2*10^5`, where the output must be produced within the time limit and every coordinate must
fit the `M = 8*n*n` budget with 64-bit arithmetic. A submission is scored entirely by the special
judge checking the four contract conditions at the *test's* scale, not at the author's.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;

    // TODO: print n marks 0 = x[0] < x[1] < ... < x[n-1] <= 8*n*n whose
    //       n*(n-1)/2 pairwise differences are all distinct (a Sidon set / perfect ruler).

    return 0;
}
```
