# TIER: trivial
# Reproduces the checker's own internal baseline construction: delay lengths are
# ascending multiples of Lmin (all share the common factor Lmin -> sparse, aliased
# echo arrivals -- the density descriptor is not addressed at all), and a single
# global gain fit crudely from the last checkpoint's target level (the decay
# descriptor gets the most obvious one-line fit and nothing more). Should score ~0.1.
import sys
GAIN_MIN, GAIN_MAX = 1e-3, 0.99


def main():
    tok = sys.stdin.read().split()
    p = 0
    N = int(tok[p]); p += 1
    T = int(tok[p]); p += 1
    Lmin = int(tok[p]); p += 1
    Lmax = int(tok[p]); p += 1
    K = int(tok[p]); p += 1
    ts = [int(tok[p + j]) for j in range(K)]; p += K
    target_db = [float(tok[p + j]) for j in range(K)]; p += K
    # target_density, weights: unused by this baseline recipe
    p += K
    p += 2

    L = []
    for i in range(N):
        Li = min(max(Lmin * (i + 1), Lmin), Lmax)
        L.append(Li)
    Lavg = sum(L) / N
    slope = target_db[-1] / max(1, ts[-1])
    g0 = 10.0 ** (slope * Lavg / 20.0)
    g0 = min(GAIN_MAX, max(GAIN_MIN, g0))
    g = [g0] * N

    out = []
    out.append(" ".join(str(x) for x in L))
    out.append(" ".join("%.6f" % x for x in g))
    sys.stdout.write("\n".join(out) + "\n")


main()
