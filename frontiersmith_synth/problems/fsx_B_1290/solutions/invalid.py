# TIER: invalid
# Emits an infeasible artifact: repeats the same rail id inside a cascade
# (duplicate rail) and also over-runs the attempt cap on later cells --
# the checker must reject this with Ratio: 0.0.
import sys


def main():
    it = sys.stdin.read().split()
    p = 0
    R = int(it[p]); S = int(it[p + 1]); B = int(it[p + 2]); K = int(it[p + 3]); p += 4

    out = []
    for _s in range(S):
        for _b in range(B):
            # duplicate rail 0 twice -> infeasible cascade
            out.append("2 0 0")
    print("\n".join(out))


if __name__ == "__main__":
    main()
