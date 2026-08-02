# TIER: invalid
"""Emits a malformed / infeasible artifact -- must score 0."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    N, M = map(int, data[1].split())
    ptr = 2 + M
    T = int(data[ptr]); ptr += 1
    K = int(data[ptr]); ptr += 1

    out = sys.stdout
    # wrong count AND negative/non-finite values -- clearly infeasible
    out.write("%d\n" % (K + 3))
    vals = ["-5.0", "nan", "inf"] + ["0"] * K
    out.write(" ".join(vals))
    out.write("\n")


if __name__ == "__main__":
    main()
