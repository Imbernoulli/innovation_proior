# TIER: invalid
"""Emits a syntactically well-formed but infeasible artifact: negative
feed amounts every day, for every type. Must score 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[0]); K = int(data[1])
    row = " ".join("-1.000000" for _ in range(K))
    out = "\n".join(row for _ in range(T))
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
