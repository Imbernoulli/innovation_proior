# TIER: trivial
"""Leading-only inverse-square fit (Newtonian point mass, short-range correction
ignored): F = c/r^2 with c by exact least squares on the training band.  This
reproduces the grader's internal baseline predictor -> Ratio ~ 0.1."""
import sys


def main():
    vals = [float(t) for t in sys.stdin.read().split()]
    rows = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    num = den = 0.0
    for (r, f) in rows:
        phi = 1.0 / (r * r)
        num += f * phi
        den += phi * phi
    c = num / den if den > 0 else 0.0
    # space-separate tokens so bare numeric coefficients are standalone
    sys.stdout.write("%.10f / r**2\n" % c)


if __name__ == "__main__":
    main()
