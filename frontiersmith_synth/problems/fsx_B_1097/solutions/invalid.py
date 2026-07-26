# TIER: invalid
"""Deliberately infeasible: prints the chain "0","00","000",...,"0"*Lmax. Every
shorter string here is a scattered subsequence of every longer one, so this
violates the antichain condition outright (and also is not remotely close to a
plausible offering along caps/budget). Must score Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it))
    Lmax = int(next(it))
    T = int(next(it))

    out = ["0" * l for l in range(1, Lmax + 1)]
    print(len(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
