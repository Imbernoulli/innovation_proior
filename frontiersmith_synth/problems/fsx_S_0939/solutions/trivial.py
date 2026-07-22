# TIER: trivial
"""Reproduce the checker's own baseline: the constant mean of the training
cycle counts. No structure exploited at all."""
import sys


def main():
    data = sys.stdin.read().split()
    n_train = int(data[0])
    # each row is 9 numbers: n nA nM nL nS cLU cMF cST cycles
    vals = data[2:]
    cycles = [float(vals[9 * i + 8]) for i in range(n_train)]
    mean_c = sum(cycles) / len(cycles)
    print("%.6f" % mean_c)


if __name__ == "__main__":
    main()
