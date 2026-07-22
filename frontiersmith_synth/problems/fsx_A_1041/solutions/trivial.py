# TIER: trivial
import sys


def canonical_code(N):
    W = 0
    while (1 << W) < N:
        W += 1
    x = (1 << W) - N
    codes = [None] * (N + 1)
    for i in range(1, x + 1):
        codes[i] = format(i - 1, "0{}b".format(W - 1)) if W - 1 > 0 else ""
    for j in range(N - x):
        idx = x + 1 + j
        val = 2 * x + j
        codes[idx] = format(val, "0{}b".format(W))
    return codes


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    next(it)  # Dmax, unused
    L = int(next(it))
    for _ in range(L):
        next(it)  # trace, unused: reproduce the ID-order canonical rack exactly

    codes = canonical_code(N)
    sys.stdout.write(" ".join(codes[1:]) + "\n")


if __name__ == "__main__":
    main()
