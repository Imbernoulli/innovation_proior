# TIER: greedy
# The obvious recipe: "a conserved quantity should look roughly CONSTANT over
# my data."  So build the feature vector [X, Y, XX, XY, YY] at every TRAIN
# STATE (never looking at the transition pairs at all), and take the
# direction of SMALLEST VALUE-VARIANCE across the sample (the bottom
# principal component of the raw feature values) as the "invariant".
#
# This is exactly a FIT OF THE STATE DISTRIBUTION, not of the dynamics.  On
# the narrow training ring (radius close to 1) this reliably locks onto
# X^2+Y^2 (Euclidean radius-squared is almost constant purely because the ring
# is narrow) -- a value-space coincidence of the SAMPLING, not a property of
# the map.  It looks great on paper (near-zero value variance!) but the map
# is a hyperbolic boost, which does NOT conserve Euclidean radius, so this
# candidate drifts hard once evaluated on the (unseen) larger-radius regime.
import sys
import numpy as np

FEATURES = ["X", "Y", "XX", "XY", "YY"]


def feat_vec(x, y):
    return [x, y, x * x, x * y, y * y]


def format_expr(coefs):
    # Always print an explicit numeric coefficient per kept term (never a
    # bare feature name) -- keeps at least one numeric literal in the output
    # regardless of which library direction is found.
    # NOTE: coefficient and '*' and feature name are kept as SEPARATE
    # whitespace tokens ("0.973095 * XX", not "0.973095*XX") so the numeric
    # literal is its own token -- both for readability and so a token-level
    # fuzz of the output (nan/inf substitution) can actually hit it.
    m = max(abs(c) for c in coefs)
    if m < 1e-12:
        return "1.000000 * XY"
    coefs = [c / m for c in coefs]
    terms = []
    for c, name in zip(coefs, FEATURES):
        if abs(c) < 0.08:
            continue
        terms.append(("+" if c > 0 else "-", "%.6f * %s" % (abs(c), name)))
    if not terms:
        return "1.000000 * XY"
    s = terms[0][1] if terms[0][0] == "+" else "-" + terms[0][1]
    for sign, tname in terms[1:]:
        s += " %s %s" % (sign, tname)
    return s


def main():
    data = sys.stdin.read().split()
    if not data:
        print("XY")
        return
    n = int(data[0])
    vals = data[2:]
    states = []
    for i in range(n):
        x = float(vals[4 * i])
        y = float(vals[4 * i + 1])
        states.append((x, y))

    V = np.array([feat_vec(x, y) for (x, y) in states], dtype=float)
    Vc = V - V.mean(axis=0, keepdims=True)
    cov = (Vc.T @ Vc) / len(states)
    evals, evecs = np.linalg.eigh(cov)   # ascending eigenvalues
    w = evecs[:, 0]                      # smallest-variance direction
    print(format_expr(list(w)))


if __name__ == "__main__":
    main()
