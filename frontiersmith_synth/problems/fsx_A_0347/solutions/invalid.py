# TIER: invalid
# Emits k copies of the same offset -> the distinctness check fires -> score 0.
import sys


def main():
    tok = sys.stdin.read().split()
    k, M = int(tok[0]), int(tok[1])
    out = [str(k)] + ["0"] * k  # all identical -> duplicates, infeasible
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
