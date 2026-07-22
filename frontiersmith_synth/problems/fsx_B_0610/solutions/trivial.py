# TIER: trivial
# Reproduces the checker baseline: every reactor runs at max throughput every step,
# never regenerating.  Catalyst-agnostic -> scores ~0.1 .
import sys

lines = sys.stdin.read().split("\n")
T, R, Q, d, L = [int(x) for x in lines[0].split()]
row = " ".join(str(Q) for _ in range(T))
sys.stdout.write("\n".join(row for _ in range(R)) + "\n")
