# TIER: trivial
"""Predict the constant mean of the training (censored) hold times.
Reproduces the checker's own internal baseline exactly -> Ratio ~= 0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    n_train = int(data[0])
    # header is "n_train test_id T", then n_train rows of "rho hold"
    idx = 3
    holds = []
    for _ in range(n_train):
        rho = float(data[idx]); hold = float(data[idx + 1])
        idx += 2
        holds.append(hold)
    mean_hold = sum(holds) / len(holds) if holds else 0.0
    print("%.6f" % mean_hold)


if __name__ == "__main__":
    main()
