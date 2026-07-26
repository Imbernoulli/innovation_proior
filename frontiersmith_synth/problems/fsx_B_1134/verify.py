#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for rigid-origami fold-to-target.
Reads the crease pattern + targets from <in>, the participant's fold angles from <out>.
Validates feasibility strictly (token count, finiteness, angle range) -- any violation
prints Ratio: 0.0. Otherwise scores: achieved-vs-target distance PLUS a loop-closure
("tearing") penalty, normalized against the checker's own flat (theta=0) baseline.
"""
import sys, math
import riglib as R

ANGLE_BOUND = 3.0     # solvers must submit angles strictly inside (-pi,pi); this margin
                       # keeps Rodrigues well away from the antipodal singularity.
LAMBDA = 0.8           # tearing-penalty weight (radians -> length-scale units)


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    it = iter(inp)
    try:
        K = int(next(it))
        modules = []
        for _ in range(K):
            a = [float(next(it)) for _ in range(4)]
            L = [float(next(it)) for _ in range(4)]
            modules.append({'a': a, 'L': L})
        targets = []
        for _ in range(K):
            targets.append(tuple(float(next(it)) for _ in range(3)))
    except Exception:
        fail("malformed input (checker bug?)")

    n_expected = R.n_angles(K)

    try:
        out_toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    if len(out_toks) != n_expected:
        fail("expected %d angles, got %d" % (n_expected, len(out_toks)))

    angles = []
    for tok in out_toks:
        try:
            v = float(tok)
        except ValueError:
            fail("non-numeric token %r" % tok)
        if not math.isfinite(v):
            fail("non-finite angle %r" % tok)
        if abs(v) > ANGLE_BOUND:
            fail("angle %.6f out of range [-%.2f,%.2f]" % (v, ANGLE_BOUND, ANGLE_BOUND))
        angles.append(v)

    # ---- internal baseline B: the flat (unfolded, theta=0) construction ----
    zero = [0.0] * n_expected
    Bval, Bdist, Bpen, Btips, Bcl = R.objective(modules, zero, targets, LAMBDA)
    Bval = max(1e-6, Bval)

    Fval, Fdist, Fpen, tips, closures = R.objective(modules, angles, targets, LAMBDA)

    sc = min(1000.0, 100.0 * Bval / max(1e-9, Fval))
    print("K=%d dist=%.6f tear_pen=%.6f B=%.6f F=%.6f Ratio: %.6f"
          % (K, Fdist, Fpen, Bval, Fval, sc / 1000.0))


if __name__ == "__main__":
    main()
