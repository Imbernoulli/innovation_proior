# TIER: trivial
# No reasoning about precedence at all: rank each job by its OWN nominal
# duration (descending, ties by id), completely ignoring how much work sits
# downstream of it and ignoring the published perturbations entirely.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    dur = [0] * (N + 1)
    for i in range(1, N + 1):
        d = int(next(it)); p = int(next(it))
        for _ in range(p):
            next(it)
        dur[i] = d
    # K perturbation records are irrelevant to this baseline; skip them.
    for _ in range(K):
        num = int(next(it)); den = int(next(it)); s = int(next(it))
        for _ in range(s):
            next(it)

    order = sorted(range(1, N + 1), key=lambda j: (-dur[j], j))
    print(" ".join(map(str, order)))


main()
