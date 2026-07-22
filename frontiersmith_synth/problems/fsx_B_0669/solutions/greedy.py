# TIER: greedy
"""The textbook recipe: apply the famous golden angle (~137.50776 deg) as a
CONSTANT divergence step for every primordium, ignoring the growth-law
schedule (bulk/rim transition) and the disk boundary entirely."""
import sys

GOLDEN = 137.50776405003785


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    out = []
    for k in range(1, N + 1):
        out.append("%.6f" % ((k * GOLDEN) % 360.0))
    print("\n".join(out))


if __name__ == "__main__":
    main()
