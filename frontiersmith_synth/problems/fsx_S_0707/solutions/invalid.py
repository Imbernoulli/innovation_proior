# TIER: invalid
# Claims zero ops and aliases every output to input line 1 -- wrong for any
# row that is not exactly x_1, so the checker's equivalence gate rejects it.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    m = int(next(it))
    out = ["0", " ".join(["1"] * m)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
