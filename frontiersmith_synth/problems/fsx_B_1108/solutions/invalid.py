# TIER: invalid
# Emits an infeasible artifact: k sink slots but every index is 0 (duplicates),
# which the feasibility check must reject -> score 0.
import sys


def main():
    it = sys.stdin.read().split()
    k = int(it[2])
    out = [str(k)] + ["0"] * k
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
