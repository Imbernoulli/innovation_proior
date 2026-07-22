# TIER: trivial
# Do-nothing plan: commit nothing, collect the terminal value of the initial
# state of charge in every scenario.  Reproduces the checker baseline (~0.1).
import sys


def main():
    data = sys.stdin.read().split("\n")
    T, S = (int(v) for v in data[0].split()[:2])
    sys.stdout.write(" ".join("0" for _ in range(T)) + "\n")


if __name__ == "__main__":
    main()
