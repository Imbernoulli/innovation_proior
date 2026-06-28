#!/usr/bin/env python3
"""Brute-force rooted-forest isomorphism oracle.

Reads the same input format as sol.cpp:
    n1
    par[1..n1]      (par == 0 means a forest root)
    n2
    par[1..n2]
and prints YES / NO.

Method (obviously correct, exponential): attach a virtual super-root 0 whose
children are the forest roots, turning each forest into a single rooted tree at
node 0. Two rooted trees are isomorphic iff there is a bijection of the children
of the roots and, recursively, of every node's children, preserving structure.

We test this directly by trying to match the two trees via recursive matching of
children using permutations. This is exponential in the branching factor, so it
is only valid for tiny n (the generator keeps n <= 8). It makes no clever
observation -- it just brute-forces the definition of isomorphism.
"""
import sys
from itertools import permutations


def read_forest(tokens, pos):
    n = tokens[pos]; pos += 1
    children = {0: []}
    for i in range(1, n + 1):
        children.setdefault(i, [])
    for i in range(1, n + 1):
        p = tokens[pos]; pos += 1
        children[p].append(i)
    return children, pos


def iso(children1, u, children2, v):
    """Are the subtrees rooted at u (in tree1) and v (in tree2) isomorphic?"""
    c1 = children1[u]
    c2 = children2[v]
    if len(c1) != len(c2):
        return False
    if not c1:
        return True
    # Try every way to pair u's children with v's children.
    for perm in permutations(c2):
        if all(iso(children1, a, children2, b) for a, b in zip(c1, perm)):
            return True
    return False


def main():
    data = sys.stdin.read().split()
    tokens = list(map(int, data))
    pos = 0
    f1, pos = read_forest(tokens, pos)
    f2, pos = read_forest(tokens, pos)
    ans = iso(f1, 0, f2, 0)
    print("YES" if ans else "NO")


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    main()
