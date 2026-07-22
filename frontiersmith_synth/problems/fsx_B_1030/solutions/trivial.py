# TIER: trivial
# Baseline construction: post the SAME flat rate on every field, regardless of
# difficulty, deadline, or who is arriving when.  This reproduces the
# checker's own internal baseline B exactly.
import sys


def main():
    it = sys.stdin.read().split()
    p = 0
    N = int(it[p]); M = int(it[p + 1]); C_UNIT = int(it[p + 2]); RMAX = int(it[p + 3]); p += 4
    # (fields/workers are ignored -- the flat rate does not look at them)
    R_FLAT = round(0.5 * RMAX)
    out = [str(R_FLAT)] * N
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
