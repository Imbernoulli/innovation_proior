# TIER: trivial
# Mirrors the checker baseline: value-blind, lightest items first, stop ~1/3 cap.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); S = int(next(it)); W = int(next(it)); V = int(next(it)); k = int(next(it))
    w = [0] * N; v = [0] * N
    for i in range(N):
        w[i] = int(next(it)); v[i] = int(next(it))
        for _ in range(S):
            next(it)
    order = sorted(range(N), key=lambda i: (w[i] + v[i], i))
    sel = []
    cw = cv = 0
    for i in order:
        if cw + w[i] <= W and cv + v[i] <= V:
            sel.append(i); cw += w[i]; cv += v[i]
            if cw * 3 >= W or cv * 3 >= V:
                break
    sys.stdout.write(" ".join(str(i) for i in sel) + "\n")


main()
