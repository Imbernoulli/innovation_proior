# TIER: trivial
"""Baseline: emit the deterministic Sidon (Mian-Chowla) layout -- a strongly
difference-heavy set (rho ~ 0.5). This reproduces the checker's internal baseline B,
so it scores ~0.1."""
import sys


def mian_chowla(n):
    A = [0]
    diffs = set()
    x = 1
    while len(A) < n:
        ok = True
        nd = []
        for a in A:
            dd = x - a
            if dd in diffs:
                ok = False
                break
            nd.append(dd)
        if ok:
            A.append(x)
            diffs.update(nd)
        x += 1
    return A


def main():
    data = sys.stdin.read().split()
    n, V = int(data[0]), int(data[1])
    A = mian_chowla(n)
    # Sidon max is ~n^2 <= V = 8 n^2, so it fits comfortably.
    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
