# TIER: invalid
# Emits an infeasible artifact: rates that overshoot RMAX (out of the
# declared [0, RMAX] range) -- must score 0.
import sys


def main():
    it = sys.stdin.read().split()
    p = 0
    N = int(it[p]); M = int(it[p + 1]); C_UNIT = int(it[p + 2]); RMAX = int(it[p + 3]); p += 4
    out = [str(RMAX + 5000)] * N
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
