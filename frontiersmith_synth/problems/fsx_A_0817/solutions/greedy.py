# TIER: greedy
"""The obvious/textbook move: pool ALL points (ignore the group/stencil
structure entirely) and fit one global linear regression label ~ b0+b1*x+b2*y.
The direction along which the fitted surface is flat (perpendicular to the
fitted gradient (b1,b2)) is declared as a pure-translation invariance
direction. This is a single recipe applied uniformly regardless of whether the
true symmetry is a translation, a rotation about some center, or a scaling
about some center -- so it has no way to represent a center at all."""
import sys, math


def solve3(A, rhs):
    def det3(M):
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    D = det3(A)
    if abs(D) < 1e-12:
        return [0.0, 0.0, 0.0]
    res = []
    for col in range(3):
        M2 = [row[:] for row in A]
        for r in range(3):
            M2[r][col] = rhs[r]
        res.append(det3(M2) / D)
    return res


def main():
    data = sys.stdin.read().split()
    pos = 0
    tid = int(data[pos]); pos += 1
    G = int(data[pos]); pos += 1
    eps = float(data[pos]); pos += 1
    pts = []
    for _ in range(G):
        group = []
        for _ in range(5):
            x = float(data[pos]); y = float(data[pos + 1]); lab = float(data[pos + 2])
            pos += 3
            group.append((x, y, lab))
        pts.extend(group)

    Sxx = Sxy = Sx = Syy = Sy = S1 = 0.0
    Sl = Slx = Sly = 0.0
    for (x, y, l) in pts:
        Sx += x; Sy += y; S1 += 1
        Sxx += x * x; Syy += y * y; Sxy += x * y
        Sl += l; Slx += l * x; Sly += l * y
    A = [[S1, Sx, Sy], [Sx, Sxx, Sxy], [Sy, Sxy, Syy]]
    rhs = [Sl, Slx, Sly]
    beta = solve3(A, rhs)
    b1, b2 = beta[1], beta[2]
    norm = math.hypot(b1, b2)
    if norm < 1e-9:
        print("0 0 0 0 1 0")
        return
    dx, dy = -b2 / norm, b1 / norm
    print("0 0 0 0 %.9g %.9g" % (dx, dy))


if __name__ == "__main__":
    main()
