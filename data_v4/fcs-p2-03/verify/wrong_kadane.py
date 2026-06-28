#!/usr/bin/env python3
# The TEMPTING-but-WRONG approach: Kadane-on-product, i.e. the max-sum greedy
# adapted to products. It only carries the running MAX (cur = max(x, cur*x))
# and never the running min, so a negative element that should multiply a
# stored large-negative into a large-positive is missed. Kept here only to
# document the counterexample; NOT the shipped solution.
import sys


def wrong(n, a):
    cur = a[0]
    best = a[0]
    for i in range(1, n):
        x = a[i]
        cur = max(x, cur * x)
        best = max(best, cur)
    return best


data = sys.stdin.read().split()
n = int(data[0])
a = [int(data[1 + k]) for k in range(n)]
print(wrong(n, a))
