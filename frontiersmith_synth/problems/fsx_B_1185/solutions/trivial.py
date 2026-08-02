# TIER: trivial
# Baseline construction: ignore location entirely (guess the beam midpoint),
# and estimate severity as the plain average of the observed |relative
# frequency drop| across the measured modes. Always feasible; reproduces the
# checker's own internal baseline B exactly (Ratio ~= 0.1 on every case).
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    t = int(data[p]); p += 1
    L = int(data[p]); p += 1
    G = int(data[p]); p += 1
    K = int(data[p]); p += 1
    modes = [int(data[p + i]) for i in range(K)]; p += K
    f0 = [float(data[p + i]) for i in range(K)]; p += K
    fdam = [float(data[p + i]) for i in range(K)]; p += K
    # gauge positions and mode-shape rows are not needed by this baseline

    S_MAX_OUT = 0.5
    x_hat = L / 2.0
    rel = [abs(1.0 - fdam[i] / f0[i]) for i in range(K)]
    s_hat = max(0.0, min(S_MAX_OUT, sum(rel) / len(rel)))

    print("%.6f %.6f" % (x_hat, s_hat))


if __name__ == "__main__":
    main()
