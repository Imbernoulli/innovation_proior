# Recover a hidden order under a sign-comparison budget

## Research question

A referee is holding a sequence of `n` boxes in a row. Box `i` contains a hidden integer key
`key[i]`, and **all the keys are distinct**. You are not allowed to look inside a box. The only thing
you can do is point at two boxes `i` and `j` and ask the one question the referee will answer:
*"is the key in box `i` smaller than the key in box `j`?"* — a single **sign comparison** that comes
back as `<` or `>`. Using only the answers to such questions, you must determine the full left-to-right
order of the boxes by increasing key, i.e. recover the permutation that sorts them.

There is a hard catch that defines the problem: you are charged **one unit per comparison you ask**,
and you are given a budget of only `B = n * ceil(log2 n)` comparisons. Recovering the order is
trivially possible if you may ask anything you like — compare every pair and read off the ranks — but
that costs `n(n-1)/2` comparisons, which is far over budget for large `n` (for `n = 2*10^5` the
all-pairs count is about `2*10^10`, while the budget is `300000`). The whole problem is to recover the
exact same order while spending no more than `B` comparisons, and to report how many you spent so the
referee can check you stayed within budget.

Because the harness is an exact, non-interactive judge, the interaction is **replayed offline**: the
hidden keys are given to your program directly on stdin (the referee's answers are then just sign
comparisons your program performs on those keys), and your program must print two things — the
recovered order, and the number of sign comparisons its comparison schedule used. The judge fixes one
canonical comparison schedule (described below) so that the spent-count is a single well-defined number
it can check, not an implementation detail.

## Input / output contract

- Input (stdin):
  - the first token is `n` (`0 <= n <= 2*10^5`);
  - then `n` integers `key[0], key[1], ..., key[n-1]` (`-10^9 <= key[i] <= 10^9`),
    whitespace-separated. The keys are **distinct**.
- Output (stdout):
  - **line 1:** the box indices `1..n` listed in ascending order of their key — the recovered
    permutation. (For `n = 0` this line is empty.)
  - **line 2:** `C`, the number of pairwise sign comparisons spent by the canonical comparison
    schedule on this input.
- The canonical comparison schedule (so that `C` is a single well-defined number): **top-down merge
  sort** on the index range `[lo, hi)`, splitting at `mid = lo + (hi - lo) / 2`, recursing on the left
  half and then the right half, and during the merge performing **exactly one** sign comparison each
  time both the left run and the right run still have an unconsumed head (the smaller key advances;
  keys are distinct so there are no ties). `C` is the total number of those merge comparisons.
- Guarantee that makes the budget meaningful: this schedule always satisfies
  `C <= n * ceil(log2 n)`, whereas an all-pairs schedule would report `n(n-1)/2`, which exceeds the
  budget for every `n >= 5`.
- Time limit: 1 second. Memory: 256 MB.

### Worked sample

Input:

```
5
3 1 4 2 5
```

Output:

```
2 4 1 3 5
7
```

The keys are `key = [3, 1, 4, 2, 5]` (1-indexed boxes `1..5`). Ascending by key the boxes are
`2 (key 1), 4 (key 2), 1 (key 3), 3 (key 4), 5 (key 5)`, so line 1 is `2 4 1 3 5`. The canonical
top-down merge sort spends `7` comparisons: it splits `[1,2,3,4,5]` into `[1,2,3]` and `[4,5]`; the
left third splits into `[1,2]` and `[3]`, costing `1` comparison to merge the pair and `2` to merge in
the singleton (`3` so far); the right half `[4,5]` costs `1`; and the final merge of the two sorted
halves of sizes `3` and `2` costs `3` more, for `3 + 1 + 3 = 7`. The budget here is
`5 * ceil(log2 5) = 5 * 3 = 15`, so `7 <= 15` is within budget.

## Background

The constraint "recover the order using only sign comparisons, counted against a budget" is exactly the
setting of **comparison-based sorting**, viewed through its query complexity. Two framings are on the
table before committing to one:

- **All-pairs ranking.** For each box, compare it against every other box; the number of boxes it beats
  is its rank, and the ranks give the order. This is dead simple and obviously correct, and it is the
  thing the phrase "find each box's place by comparing it to the others" suggests. Its cost is
  `n(n-1)/2` comparisons — `Theta(n^2)` — which is the quantity to beat.
- **A comparison schedule that reuses information.** Each comparison's answer constrains many future
  ranks at once, so a schedule that never re-compares an already-determined relationship can recover the
  full order far more cheaply. The open question is *how cheaply*, and which standard schedule realizes
  it within `n * ceil(log2 n)`.

The decision-tree lower bound says any comparison sort needs at least `ceil(log2(n!)) ~ n log2 n - 1.44 n`
comparisons in the worst case, so `n log2 n` is essentially the floor; the budget `n * ceil(log2 n)` is
that floor rounded up to a clean target. The difficulty of the problem is recognizing that the
quadratic all-pairs method is not forced — that an `O(n log n)`-comparison schedule both exists and is a
standard sorting algorithm — and then pinning the exact comparison count so the budget can be checked.

## Evaluation settings

Judged on hidden tests covering: the empty sequence (`n = 0`) and a single box (`n = 1`, zero
comparisons); already-sorted and reverse-sorted inputs (the best- and a high-cost case for the count);
the merge-sort worst-case interleave (which maximizes `C` and most tightly probes the budget); keys
spanning the full signed `10^9` range including negatives and zero; powers of two and the values just
above and below them (where `ceil(log2 n)` jumps); and large instances up to `n = 2*10^5`, where an
`O(n^2)` comparison schedule both overshoots the budget and times out.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> key(n);
    for (auto &x : key) cin >> x;

    // TODO: recover the ascending order of the boxes using a sign-comparison
    // schedule within n*ceil(log2 n) comparisons, and report how many it used.

    return 0;
}
```
