# Optimal binary search tree: minimum expected search cost

## Research question

You are given `n` keys that are kept in a binary search tree (BST). The keys have a fixed sorted
order; call them key `1`, key `2`, ..., key `n` in increasing order. For each key `i` you are given a
non-negative access frequency `f[i]` (how often that key is searched for). A BST stores every key, and
because it is a search tree the **in-order traversal must list the keys in sorted order** — so the set
of legal trees is exactly the set of BST shapes over `1..n`.

The cost of looking up key `i` in a given tree is `depth(i) * f[i]`, where the **root is at depth 1**,
its children at depth 2, and so on (depth = number of nodes visited, i.e. number of comparisons). The
**total expected search cost** of a tree is `sum over i of depth(i) * f[i]`. Over all BST shapes,
output the **minimum** total expected search cost.

This is the optimal-binary-search-tree problem: choosing the tree shape that makes frequently accessed
keys shallow without violating the search-order constraint. It is the canonical setting where a
locally sensible rule — "put the most frequently accessed key at the root" — must be weighed against a
global optimum.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 500`); then `n` integers `f[1], ..., f[n]`
  (`0 <= f[i] <= 10^9`), whitespace-separated, given in sorted key order.
- Output (stdout): a single line with the minimum total expected search cost.
- For `n = 0` (no keys) the cost is `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `f = [2, 3, 4]` (keys 1, 2, 3) the answer is `15`.

## Background

The keys are fixed in sorted order, so a tree is fully determined by, recursively, **which key is the
root of each contiguous range of keys**: picking key `r` as the root of the range `[i..j]` forces keys
`i..r-1` into the left subtree and keys `r+1..j` into the right subtree (this is exactly the BST
ordering constraint). Two families of approach are on the table before committing to one:

- **Greedy "most frequent at root".** For each range, make the highest-frequency key the root, then
  recurse on the two halves. It is `O(n log n)`-ish and immediate to write; the open question is
  whether putting the most-searched key closest to the root is actually optimal once the search-order
  constraint forces the rest of the keys into fixed left/right halves.
- **Interval dynamic programming.** Let `dp[i][j]` be the minimum cost of an optimal BST on keys
  `i..j`. Try every key as the root of the range and combine the optimal sub-results. This is
  `O(n^3)` with the straightforward root scan; the open question is the exact recurrence — in
  particular how the cost of sinking both subtrees one level deeper is accounted for.

## Evaluation settings

Judged on hidden tests covering: tiny `n` (including `n = 0` and `n = 1`); all-equal frequencies;
strictly increasing and strictly decreasing frequencies; instances with a single huge frequency placed
at an end or in the middle (which most tempt the greedy root choice); all-zero frequencies; and large
`n = 500` with values near `10^9` (so the accumulated cost can exceed a 32-bit integer).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> cost 0
    vector<long long> f(n + 1);
    for (int i = 1; i <= n; i++) cin >> f[i];

    // TODO: compute the minimum total expected search cost over all BST shapes on keys 1..n,
    // where cost = sum over i of depth(i) * f[i] and the root is at depth 1.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
