# Maximum-weight independent set on a CYCLE (no two adjacent, wrap included)

## Research question

You are given `n` integers `a[0..n-1]` (values may be negative) arranged on a **circle**: position `i`
is adjacent to position `i+1` for every `i`, and — crucially — position `n-1` is adjacent to position
`0` as well (the *wrap edge*). Choose a subset of the positions so that **no two chosen positions are
cyclically adjacent** (you may choose none), and **maximize the sum** of the chosen values. Output that
maximum sum. Because the empty subset is allowed, the answer is always at least `0`.

This is the cyclic case of maximum-weight independent set. It looks like a one-line variation on the
path version, but the wrap edge couples the two endpoints, and that coupling is exactly where a naive
"just run the path DP once" goes wrong.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum achievable sum.
- Time limit: 1 second. Memory: 256 MB.

Adjacency convention for the small cases: for `n >= 2` the cycle has edges `{i, (i+1) mod n}`. For
`n = 1` there is a single position with no neighbour, so it may be taken (the answer is `max(a[0], 0)`).
For `n = 0` the answer is `0`.

Example: for `a = [5, 1, 1, 5, 1, 5]` (a 6-cycle) the answer is `11`, taking indices `1, 3, 5`
(`1 + 5 + 5`). On the *path* the best is `15` via indices `0, 3, 5`, but on the *circle* indices `0`
and `5` are cyclically adjacent, so that selection is illegal — the wrap edge forbids using both index
`0` and index `n-1`.

### Sample

Input:

```
6
5 1 1 5 1 5
```

Output:

```
11
```

(The best legal circular selection is indices `1, 3, 5` summing to `11`. On the *path* the best would
be `15` via indices `0, 3, 5`, but on the *circle* indices `0` and `5` are adjacent, so that is illegal.)

## Background

The constraint "no two chosen cyclically adjacent" makes this a constrained selection problem on a
cycle. Several approaches are on the table before committing to one:

- **Greedy by value.** Repeatedly take the largest remaining positive element and forbid its two
  (cyclic) neighbours. `O(n log n)` and trivial to write; the open question is whether grabbing the
  largest element is optimal under a *global* adjacency constraint.
- **Single-pass circular DP.** Run the standard left-to-right path DP once over `a[0..n-1]` and hope a
  small tweak accounts for the wrap. The open question is whether one pass can correctly forbid the
  simultaneous use of both endpoints without double-counting or wrongly excluding valid selections.
- **Two linear DP passes.** Observe that a valid circular selection cannot use both endpoints, split on
  which endpoint is forbidden, and run the proven path DP twice. `O(n)`; the open question is the exact
  split and the small-`n` corners.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays with negatives and zeros, the empty array
(`n = 0`), single element (`n = 1`), a pair (`n = 2`), all-negative arrays (answer should be `0`),
selections forced apart by the wrap edge, and large `n = 2*10^5` with values near `10^9` (so the running
sum can exceed a 32-bit integer).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: compute the maximum sum of a subset with no two CYCLICALLY adjacent
    // TODO: positions (positions 0 and n-1 are adjacent); empty selection allowed.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
