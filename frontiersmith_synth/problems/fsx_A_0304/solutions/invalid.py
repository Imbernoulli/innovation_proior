# TIER: invalid
# Emits stations far outside the reef plot -> feasibility fails -> score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    out = ["5.0 5.0" for _ in range(N)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
