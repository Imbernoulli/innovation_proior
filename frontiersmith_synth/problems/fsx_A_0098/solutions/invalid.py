# TIER: invalid
# Emits a single all-zero survey mode -> reconstructs the zero tensor, never matches
# the (nonzero) target -> Ratio 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    I, J, K = int(toks[0]), int(toks[1]), int(toks[2])
    out = ["1", " ".join(["0"] * (I + J + K))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
