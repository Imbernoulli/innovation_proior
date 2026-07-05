# TIER: invalid
# Emits an infeasible artifact: n copies of row 0 (not a permutation -- every
# mark in the same row). The checker must reject this and score 0.
import sys

n = int(sys.stdin.read().split()[0])
print(" ".join("0" for _ in range(n)))
