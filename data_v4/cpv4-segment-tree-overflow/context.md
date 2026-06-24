# Range-add, range-sum on a fluctuating ledger

## Research question

A logistics company keeps a daily ledger of `n` warehouse balances `a[1..n]` (balances may be
negative — a warehouse can be overdrawn). Two kinds of operations stream in:

- **Adjustment** `1 l r v`: add the (possibly negative) integer `v` to every balance in the
  inclusive range `[l, r]`.
- **Audit** `2 l r`: report the *sum* of the current balances in the inclusive range `[l, r]`.

You must process all operations in order and, for every audit, print the requested range sum. The
point is to support both bulk adjustments and range audits fast enough that neither degrades to
re-scanning the array, while reporting the audited totals **exactly**.

This is the canonical range-update / range-query problem and it is the engine inside countless
larger tasks (sweep-line area, offline interval accounting, difference-array DP made online), so the
one-dimensional version has to be exactly right — including the corner where the accumulated totals
grow far past what a 32-bit integer can hold.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `q`
  (`1 <= n <= 2*10^5`, `1 <= q <= 2*10^5`). The second line has `n` integers, the initial
  balances `a[i]` (`-10^9 <= a[i] <= 10^9`). Then `q` lines follow, each an operation:
  - `1 l r v` with `1 <= l <= r <= n` and `-10^9 <= v <= 10^9` — a range adjustment.
  - `2 l r` with `1 <= l <= r <= n` — a range audit.
  All indices are **1-indexed and inclusive**.
- Output (stdout): for each audit, one line with the exact range sum.
- Time limit: 2 seconds. Memory: 256 MB.

Example: starting from `a = [10^9, 10^9, 10^9, 10^9, 10^9]`, the audit `2 1 5` reports
`5*10^9 = 5000000000` — already larger than the 32-bit signed maximum `2147483647`.

## Background

Two families of approach are on the table before committing to one:

- **Plain array with difference trick / direct loops.** Apply each adjustment by looping over
  `[l, r]` and answer each audit by looping over `[l, r]`. Trivial to write and obviously correct,
  but each operation is `O(n)`, so `q` operations cost `O(nq)` — up to `4*10^10` element touches,
  far past the time limit. Useful only as a brute-force oracle on small inputs.
- **Segment tree with lazy propagation.** Store subtree sums in a tree over the array; a range
  adjustment is pushed as a *lazy* "add to this whole segment" tag instead of being applied to every
  leaf, and the tag is propagated down only when a query or a narrower update needs to descend. Each
  operation is `O(log n)`. The open questions are the exact lazy-tag semantics (how a pending add
  affects a node's stored sum, which depends on how many elements the node covers) and the order of
  push-down versus recursion.

## Evaluation settings

Judged on hidden tests covering: all-positive and mixed-sign balances; many overlapping range
adjustments before an audit; single-element ranges (`l == r`); the whole array (`1, n`); `n = 1`;
adjustments with `v = 0`; and large instances `n, q = 2*10^5` with `|a[i]|, |v|` near `10^9`, so an
audited sum can reach roughly `2*10^5 * (10^9 + 2*10^5 * 10^9) ~ 4*10^{19}` in magnitude in the
worst arrangement — every running sum, every node aggregate, and every lazy tag must be 64-bit.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, q;
vector<long long> tree;   // subtree sums
vector<long long> lazy;   // pending per-element add to push down

// TODO: build the tree, implement lazy range-add update and range-sum query,
// then process the q operations (type 1 = adjust, type 2 = audit).

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n >> q)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // ... build and process ...

    return 0;
}
```
