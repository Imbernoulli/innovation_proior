# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    # a mapping that is NOT a permutation of 0..n-1 (all zeros) -- must be
    # rejected by the checker's feasibility gate before any score is given.
    mapping = ["0"] * n
    out = [str(n), " ".join(mapping), "1", "G 1"]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
