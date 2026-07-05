# TIER: trivial
"""Reproduce the checker baseline: pile the whole population into the
left-most pools ("fill from the left"). Scores ~0.1."""
import sys


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0]); S = int(tok[1])
    cap = [int(x) for x in tok[2:2 + n]]
    f = [0.0] * n
    r = float(S)
    for i in range(n):
        take = cap[i] if cap[i] < r else r
        if take < 0:
            take = 0.0
        f[i] = float(take)
        r -= take
        if r <= 1e-12:
            break
    sys.stdout.write(" ".join("%.12g" % x for x in f) + "\n")


if __name__ == "__main__":
    main()
