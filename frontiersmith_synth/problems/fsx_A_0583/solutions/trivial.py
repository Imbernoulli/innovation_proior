# TIER: trivial
# Do-nothing baseline: predict 0 for every held-out ledger line.  This is exactly
# the checker's internal "predict 0" baseline, so it reproduces Ratio ~ 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    K = int(data[1])
    sys.stdout.write("\n".join("0" for _ in range(K)) + "\n")


if __name__ == "__main__":
    main()
