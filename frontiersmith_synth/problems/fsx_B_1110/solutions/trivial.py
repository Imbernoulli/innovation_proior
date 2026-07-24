# TIER: trivial
# Do-nothing baseline: predict each class's calm-season mean wait as a
# constant (the checker's own internal baseline). Ignores rho and the mix
# entirely -> reproduces ~0.1.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    K = int(toks[1])
    vals = toks[3:]
    wsum = [0.0] * K
    idx = 0
    for _ in range(n):
        idx += 1 + K
        for c in range(K):
            wsum[c] += float(vals[idx])
            idx += 1
    out = ["W%d = %.6f" % (c + 1, wsum[c] / n) for c in range(K)]
    print("\n".join(out))


if __name__ == "__main__":
    main()
