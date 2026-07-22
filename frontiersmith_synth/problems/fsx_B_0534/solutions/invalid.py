# TIER: invalid
# Emits an infeasible artifact: claims firebreaks on ignition cells and far more
# than the budget allows -> the checker must reject with Ratio 0.0.
import sys


def main():
    it = sys.stdin.read().split()
    N = int(it[0]); F = int(it[1]); K = int(it[2])
    # place a "firebreak" on the first ignition point (forbidden) and overshoot F
    r = int(it[3]); c = int(it[4])
    m = F + 500
    out = [str(m)]
    for _ in range(m):
        out.append("%d %d" % (r, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
