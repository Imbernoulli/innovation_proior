# Collision-free frequency gap planning (construct a Sidon set)

## Research question

A radio operator must assign integer carrier frequencies to `n` transmitters. To avoid
second-order intermodulation interference, **every pair of transmitters must have a distinct
frequency gap**: if transmitters at frequencies `x < y` and `u < v` are two different pairs, then
`y - x != v - u`. Equivalently, the multiset of all `C(n, 2)` positive pairwise differences must
contain no repeats. A set of integers with this property is called a **Sidon set** (a `B_2` set):
all pairwise differences distinct, which is the same as saying all unordered pairwise *sums* are
distinct.

The frequencies must also fit the licensed band: every assigned frequency is an integer in
`[0, 4*n*n]`. Given `n`, **output any** `n` distinct integers in that band that form a Sidon set.

The answer is not unique. The task is purely constructive: produce a witness and be sure it is
valid *at the requested scale*, not merely on the handful of small cases that are easy to eyeball.

## Input / output contract

- Input (stdin): a single integer `n` (`2 <= n <= 3000`).
- Output (stdout): `n` distinct integers, space-separated on one line, each in `[0, 4*n*n]`, whose
  set of all pairwise positive differences are pairwise distinct (a Sidon set). Any valid set is
  accepted; order of the printed values does not matter.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 5`, one valid answer is `0 11 24 34 41`. Its ten pairwise gaps are
`7, 10, 11, 13, 17, 23, 24, 30, 34, 41` — all different, and the largest value `41 <= 4*25 = 100`.

## Background

Two ideas are on the table before committing:

- **Greedy (Mian–Chowla style).** Sweep candidate values `0, 1, 2, ...` upward; keep the current
  one iff adding it introduces no repeated difference with the values kept so far. It is short to
  write and obviously produces a Sidon set. The open question is the *range*: how fast does the
  `n`-th kept value grow, and does it stay inside `4*n*n` for every `n` up to the limit?
- **Algebraic (Erdős–Turán).** For a prime `p`, the set `{ 2*p*k + (k*k mod p) : k = 0..p-1 }` is a
  Sidon set contained in `[0, 2*p*p)`. Picking the smallest prime `p >= n` and taking the first `n`
  elements yields a Sidon set whose largest value is provably below `4*n*n`. The open question is
  the correctness of the modular formula and the prime bound.

The two-pointer angle is the *certification*: once a candidate set is built and sorted, all of its
`C(n, 2)` pairwise differences can be generated, sorted, and scanned with a single two-pointer
adjacency pass that flags any duplicate — a check that must be run at the real `n`, because a
construction can be Sidon for `n = 4` and silently fail for larger `n`.

## Evaluation settings

Judged on hidden tests covering `n = 2`, `n = 3`, a spread of small `n`, and the large boundary
`n = 3000`. Each test accepts the output iff it has exactly `n` distinct integers, all within
`[0, 4*n*n]`, and the set is a genuine Sidon set. A construction that obeys the range only for small
`n` scores `0` on the large tests.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;

    // TODO: build n distinct integers in [0, 4*n*n] whose pairwise differences are
    //       all distinct (a Sidon set), and print them space-separated on one line.

    return 0;
}
```
