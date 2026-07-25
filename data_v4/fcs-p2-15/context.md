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

## Evaluation settings

Judged on hidden tests covering: tiny `n` (including `n = 0` and `n = 1`); all-equal frequencies;
strictly increasing and strictly decreasing frequencies; instances with a single huge frequency placed
at an end or in the middle; all-zero frequencies; and large `n = 500` with values near `10^9`.

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
