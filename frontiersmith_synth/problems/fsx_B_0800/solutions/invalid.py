# TIER: invalid
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    T = int(next(it))
    # Deliberately infeasible: blast every week (including the mandatory legal
    # closed weeks) with a quota far above the regulatory ceiling.
    lines = ["1000000000.0"] * T
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
