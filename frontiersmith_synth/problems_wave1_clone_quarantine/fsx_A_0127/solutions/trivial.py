# TIER: trivial
# Reproduces the checker's baseline: the contiguous ladder {0,...,m0-1}.
# F = |A+A| + |A-A| = (2*m0-1) + (2*m0-1) = 4*m0-2 == B, so Ratio ~= 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    k = int(data[0])
    V = int(data[1])
    m0 = min(k, V + 1)
    A = list(range(m0))
    out = [str(len(A))]
    out += [str(x) for x in A]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
