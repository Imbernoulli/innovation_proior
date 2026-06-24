# Exact-weight ballast loading with signed trim scores

## Research question

A trimming bench holds `n` ballast blocks. Block `i` has an integer **weight** `w[i] >= 1` and an
integer **trim score** `v[i]` that may be **negative, zero, or positive** (a block can drag the trim
the wrong way yet still be needed to hit a weight). You must load a subset of blocks whose **total
weight is exactly `C`**, and among all subsets that hit `C` exactly you want to **maximize the total
trim score**. The empty subset is a legal load: it has weight `0` and score `0`. If **no** subset of
blocks has total weight exactly `C`, the target is unreachable and you must report that.

This is the exact-capacity (subset-sum-constrained) 0/1 knapsack. Unlike the textbook "weight at
most `C`" version, the *exact* constraint plus *signed* scores means the answer can legitimately be a
negative number, and it must be kept strictly distinct from the "no valid load exists" outcome —
which is precisely where a careless base case turns into a wrong answer.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `C`
  (`0 <= n <= 2000`, `0 <= C <= 5*10^4`). Then `n` lines follow, the `i`-th containing
  `w[i]` and `v[i]` (`1 <= w[i] <= 10^9`, `-10^9 <= v[i] <= 10^9`).
- Output (stdout): if some subset has total weight exactly `C`, print the maximum total trim score
  over all such subsets (this value may be negative or zero). Otherwise print the single token
  `IMPOSSIBLE`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `C = 7` and blocks `(w,v) = (3,5), (4,-2), (2,0), (5,4)` the answer is `4`: the subsets
hitting weight `7` are `{3,4}` (score `5 + (-2) = 3`) and `{2,5}` (score `0 + 4 = 4`), and the best
is `4`.

## Background

Two families of approach are on the table before committing.

- **Meet-in-the-middle / brute enumeration.** Enumerate subsets (or split the items in half and merge
  by weight). Exhaustive enumeration is `O(2^n)` and obviously correct but only works for tiny `n`;
  meet-in-the-middle reaches `n` around 40 but is fiddly to merge under the *maximize-score-at-an-
  exact-weight* objective. Neither scales to `n = 2000`.
- **Capacity-indexed dynamic programming.** Build a table indexed by achievable total weight
  `0..C`, storing the best trim score reachable at each weight, and relax it block by block in the
  0/1 manner. This is `O(n*C)` time and `O(C)` memory. The open questions are the exact base case
  (which weights are "reachable" before any block is placed), the transition order that keeps each
  block usable at most once, and how to encode "weight not reachable" so it never masquerades as a
  real score of `0`.

## Evaluation settings

Judged on hidden tests covering: subsets that hit `C` with a mix of positive, zero, and negative
scores; cases where the only ways to reach `C` give a **negative** optimum (must print the negative
number, not `IMPOSSIBLE`); the empty-load corner `C = 0` (answer `0`, even when every block has a
negative score); `n = 0`; all-negative score arrays; targets `C` that **no** subset can reach
(answer `IMPOSSIBLE`); blocks with weight far larger than `C`; and large `n = 2000`, `C = 5*10^4`
with `|v[i]|` near `10^9` (so the accumulated score can exceed the 32-bit range).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // TODO: compute the maximum total trim score over subsets whose weight is
    //       EXACTLY C (empty subset allowed, score 0); print IMPOSSIBLE if no
    //       subset reaches weight C.
    // long long answer = ...;

    // cout << answer << "\n";   // or "IMPOSSIBLE"
    return 0;
}
```
