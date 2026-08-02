# TIER: trivial
# Reproduces the checker's own baseline construction exactly: partition vertices into
# consecutive groups of size (k-1) so no single group can contain a k-clique; edges inside
# a group get channel 0, edges between groups gi,gj get channel (gi+gj)%2. Naive, not
# clique-count-aware -> scores ~0.1 by construction (F == B).
import sys

t = sys.stdin.read().split()
n = int(t[0]); k = int(t[1])
g = max(1, k - 1)

color = [[0] * n for _ in range(n)]
for i in range(n):
    bi = i // g
    for j in range(n):
        if i == j:
            continue
        bj = j // g
        color[i][j] = 0 if bi == bj else (bi + bj) % 2

out = []
for i in range(n):
    out.append(" ".join(str(x) for x in color[i]))
sys.stdout.write("\n".join(out) + "\n")
