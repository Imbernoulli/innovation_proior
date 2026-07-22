# TIER: invalid
# Emits throughput above the per-reactor cap (out of range) -> checker scores 0.
import sys

lines = sys.stdin.read().split("\n")
T, R, Q, d, L = [int(x) for x in lines[0].split()]
row = " ".join(str(Q + 7) for _ in range(T))  # 17 > Q -> infeasible
sys.stdout.write("\n".join(row for _ in range(R)) + "\n")
