# TIER: invalid
# Emits out-of-range, duplicate garbage (and the wrong token count) -> must
# score 0 under strict feasibility checking.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0]); M = int(data[1])
    # deliberately infeasible: duplicated + out-of-range posts, wrong count
    print(" ".join([str(M + 999)] * (n + 3)))


if __name__ == "__main__":
    main()
