# TIER: invalid
# Emits an asymmetric channel matrix (color[0][1] != color[1][0]) -> feasibility violation,
# must score 0.
import sys

t = sys.stdin.read().split()
n = int(t[0]); k = int(t[1])

color = [[0] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i != j:
            color[i][j] = (i + j) % 2
# break symmetry deliberately
if n >= 2:
    color[0][1] = 1 - color[0][1]

out = []
for i in range(n):
    out.append(" ".join(str(x) for x in color[i]))
sys.stdout.write("\n".join(out) + "\n")
