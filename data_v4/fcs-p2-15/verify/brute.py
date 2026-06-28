#!/usr/bin/env python3
"""
Independent brute-force oracle for the Optimal Binary Search Tree problem.

Definition reproduced from context.md:
  - n keys with sorted-order access frequencies f[1..n].
  - A BST stores all n keys in key order (in-order traversal == sorted order).
  - The search cost of key i is depth(i) * f[i], where the root has depth 1.
  - The total expected cost of a tree is sum over i of depth(i)*f[i].
  - We want the MINIMUM total cost over all shapes of BST on these keys.

This oracle enumerates every BST shape recursively with memoized recursion that
expresses the cost directly as a recursive minimization over the choice of root
for each contiguous key interval, but computed *without* the interval-weight
trick of the DP -- instead it literally accumulates per-key depths by walking the
recursion.  To stay genuinely independent from sol.cpp we compute the cost by
recursively building the optimal subtree cost as:

    cost(i, j) = min over root r in [i..j] of
                 ( f[r]*1                      # root at depth 1 within this subtree
                   + lift(cost-with-depths of left subtree by 1 level)
                   + lift(cost-with-depths of right subtree by 1 level) )

We implement that by tracking, for an interval, the minimum total of
sum(local_depth * f) where local_depth counts the root of that subtree as depth 1.
When a subtree of weight W (sum of its frequencies) is hung one level below a
parent, every node inside gains exactly one extra unit of depth, adding W to the
total.  So:

    best(i, j), W(i,j) = sum f over [i..j]
    best(i, j) = min_r [ best(i, r-1) + best(r+1, j) ] + W(i, j)

with best(empty) = 0.  That is mathematically the same value the DP finds, but
here we ALSO cross-check it against a fully explicit shape enumeration for small
n to guard against a shared-formula mistake.
"""
import sys
from functools import lru_cache
from itertools import count


def solve_recursive(f):
    """Memoized min-cost via the interval recursion (depths accumulated by lifting)."""
    n = len(f)
    if n == 0:
        return 0
    # f is 0-indexed list of length n here.
    pref = [0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + f[i]

    from functools import lru_cache as _lru

    @_lru(maxsize=None)
    def best(i, j):
        # interval keys i..j inclusive (0-indexed); empty if i > j
        if i > j:
            return 0
        w = pref[j + 1] - pref[i]
        m = None
        for r in range(i, j + 1):
            c = best(i, r - 1) + best(r + 1, j)
            if m is None or c < m:
                m = c
        return m + w

    return best(0, n - 1)


def all_trees(keys):
    """Yield every BST shape over the contiguous key list `keys` as nested tuples.
    Returns list of trees; each tree is (root_idx_in_keys, left_tree, right_tree)
    or None for empty.  Used only for tiny n to independently verify cost."""
    if not keys:
        return [None]
    res = []
    for r in range(len(keys)):
        for L in all_trees(keys[:r]):
            for R in all_trees(keys[r + 1:]):
                res.append((keys[r], L, R))
    return res


def tree_cost(tree, f, depth=1):
    """Total cost = sum depth*f over an explicit tree shape; f indexed by key id."""
    if tree is None:
        return 0
    key, L, R = tree
    return depth * f[key] + tree_cost(L, f, depth + 1) + tree_cost(R, f, depth + 1)


def solve_explicit(f):
    """Brute force over ALL tree shapes (only feasible for small n)."""
    n = len(f)
    if n == 0:
        return 0
    keys = list(range(n))
    best = None
    for t in all_trees(keys):
        c = tree_cost(t, f)
        if best is None or c < best:
            best = c
    return best


def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    f = []
    for _ in range(n):
        f.append(int(data[idx])); idx += 1

    ans = solve_recursive(f)

    # For small n, cross-check the recursion against full explicit enumeration.
    if n <= 8:
        ans2 = solve_explicit(f)
        assert ans == ans2, f"oracle internal mismatch: rec={ans} explicit={ans2} f={f}"

    print(ans)


if __name__ == "__main__":
    main()
