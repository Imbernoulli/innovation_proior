# TIER: trivial
"""Ignore the data entirely (beyond its mean): predict the historical average
lead time as a constant. Reproduces the checker's own baseline construction."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    n = int(header[0])
    Ls = []
    for i in range(1, n + 1):
        parts = data[i].split()
        Ls.append(float(parts[2]))
    meanL = sum(Ls) / len(Ls)
    print("%.6f" % meanL)


if __name__ == "__main__":
    main()
