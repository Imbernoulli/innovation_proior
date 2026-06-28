# Seating a carousel so every running total lands on a different seat

## Research question

A circular carousel has `n` numbered seats `0, 1, ..., n-1`. You must hand out the `n` boarding
tickets, which are the distinct values `1, 2, ..., n`, to the `n` riders **in some order** — that is,
you choose a permutation `p[0..n-1]` of `{1, ..., n}`. The riders board one at a time; after the
`k`-th rider boards, the **cumulative ticket total** is the prefix sum

```
S_k = p[0] + p[1] + ... + p[k-1]   (for k = 1, 2, ..., n)
```

and that rider is assigned to seat `S_k mod n`. The carousel only runs smoothly if **no two riders
are ever assigned the same seat**, i.e. the `n` prefix sums `S_1, ..., S_n` are **pairwise distinct
modulo `n`**.

Your job: output any boarding order `p` achieving this, or report that none exists.

This is a *construction* problem in the prefix-sum family: the object you emit is a sequence, and its
correctness is a global property of all `n` of its prefix sums taken modulo `n`. The danger is that a
short, pretty boarding order can satisfy the property for a handful of small `n` purely by accident
and then fail at the scale the judge actually tests.

## Input / output contract

- Input (stdin): a single integer `n` (`1 <= n <= 10^7`).
- Output (stdout):
  - If a valid boarding order exists, print the `n` values `p[0..n-1]` on one line, space-separated,
    terminated by a newline. The values must be a permutation of `1..n` whose prefix sums are pairwise
    distinct modulo `n`.
  - If no valid order exists, print a single line containing `-1`.
  - Any answer meeting the property is accepted (the judge is a checker, not an exact-match diff).
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 6` one valid answer is `6 1 4 3 2 5`. Its prefix sums are
`6, 7, 11, 14, 16, 21`, which modulo `6` are `0, 1, 5, 2, 4, 3` — all six residues, each once.
For `n = 3` the correct output is `-1`.

## Background

Two facts hover over this problem and must be settled before writing any loop:

- **When does an answer exist at all?** Asking for a permutation of `1..n` whose prefix sums hit every
  residue mod `n` exactly once is the classical question of whether the cyclic group `Z_n` is
  *sequenceable*. The boundary between "possible" and "impossible" depends on `n` in a way that is
  easy to get wrong by sampling only the smallest cases.
- **How to build one when it exists.** There is a closed-form boarding order that works for the whole
  feasible range, so no search or backtracking is needed — but a wrong-but-plausible closed form can
  reproduce the right answer for several small `n` (including several that a casual tester would try)
  before collapsing at larger `n`. The construction has to be one you can *prove*, then *verify at the
  scale the constraints demand*, not one that merely survives `n <= 8`.

## Evaluation settings

Judged on hidden tests covering: the trivial `n = 1`; small odd `n` (`3, 5, 7, 9`) where the answer is
`-1`; small even `n`; powers of two (`2, 4, 8, 16, ...`), which are exactly the sizes where a tempting
wrong construction accidentally works; and large `n` up to `10^7` of both parities. For each feasible
case the checker confirms the output is a permutation of `1..n` and that its `n` prefix sums are
distinct modulo `n`; for each infeasible case it confirms the output is exactly `-1`. Producing ~`10^7`
numbers within the time limit requires buffered output.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n;
    if (!(cin >> n)) return 0;

    // TODO: decide feasibility, then either print a permutation of 1..n whose
    //       prefix sums are pairwise distinct modulo n, or print -1.

    return 0;
}
```
