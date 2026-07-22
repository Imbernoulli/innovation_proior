# TIER: strong
# The insight: a conserved quantity is a linear functional w . theta(state)
# whose CHANGE across each observed TRANSITION is zero -- i.e. w lies in the
# (right) NULL SPACE of the matrix of one-step feature VARIATIONS
#     Delta_i = theta(next_i) - theta(state_i)      theta = [X,Y,XX,XY,YY]
# This is a completely different object from "fit a curve to next_state" or
# "find a quasi-constant direction of the state values": it never asks what
# theta LOOKS LIKE on the sample, only how theta MOVES under the map.  Solve
# it by SVD of the stacked Delta matrix and take the right-singular vector
# with the SMALLEST singular value (the best least-squares null vector).
# Because that vector's defining property (zero change per transition) is a
# LAW, not a sampling artifact, it holds just as well far outside the training
# ring.
import sys
import numpy as np

FEATURES = ["X", "Y", "XX", "XY", "YY"]


def feat_vec(x, y):
    return [x, y, x * x, x * y, y * y]


def format_expr(coefs):
    m = max(abs(c) for c in coefs)
    if m < 1e-12:
        return "XY"
    coefs = [c / m for c in coefs]
    terms = []
    for c, name in zip(coefs, FEATURES):
        if abs(c) < 0.02:
            continue
        if abs(c - 1) < 0.02:
            terms.append(("+", name))
        elif abs(c + 1) < 0.02:
            terms.append(("-", name))
        else:
            terms.append(("+" if c > 0 else "-", "%.6f*%s" % (abs(c), name)))
    if not terms:
        return "XY"
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
    rows = []
    for i in range(n):
        x = float(vals[4 * i])
        y = float(vals[4 * i + 1])
        xp = float(vals[4 * i + 2])
        yp = float(vals[4 * i + 3])
        rows.append((x, y, xp, yp))

    M = np.array([np.array(feat_vec(xp, yp)) - np.array(feat_vec(x, y))
                  for (x, y, xp, yp) in rows], dtype=float)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    w = Vt[-1]
    print(format_expr(list(w)))


if __name__ == "__main__":
    main()
