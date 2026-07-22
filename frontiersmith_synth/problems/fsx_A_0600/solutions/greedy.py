# TIER: greedy
# The obvious first attempt: cut the heaviest-weight edges (equivalently the edges
# that look "most infectious" by raw weight). This is the trap -- weight ignores
# where the edge sits in the spectral core.
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); m = int(next(it)); k = int(next(it))
    edges = []
    for idx in range(m):
        u = int(next(it)); v = int(next(it)); w = int(next(it))
        edges.append((w, idx))
    order = sorted(range(m), key=lambda i: (-edges[i][0], i))
    pick = order[:min(k, m)]
    print(" ".join(str(i + 1) for i in pick))

main()
