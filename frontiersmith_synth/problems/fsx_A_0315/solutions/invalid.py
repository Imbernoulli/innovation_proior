# TIER: invalid
# Emits an infeasible profile (intensities above the ceilings), which the checker
# must reject -> Ratio 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    u = [float(t) for t in toks[1:1 + N]]
    f = [u[i] * 100.0 + 5.0 for i in range(N)]  # blows past every ceiling
    sys.stdout.write(" ".join("%.6f" % x for x in f) + "\n")


if __name__ == "__main__":
    main()
