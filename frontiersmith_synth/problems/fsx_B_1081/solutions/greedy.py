# TIER: greedy
"""The textbook balloon-gore answer: cut the skin into K equal-width vertical
meridian strips, spending as much of the seam budget as an equal-spaced
cutting pattern allows, WITHOUT ever looking at where the curvature actually
is. This is the classic developable-surface recipe (equal gores are optimal
for a surface of revolution) applied blindly to a surface that is NOT
rotationally symmetric -- exactly the trap the family is built around.
Whichever K equal-spaced boundaries fit under budget B, all curvature peaks
that don't happen to sit on one of those specific boundary columns stay
fully unrelieved."""
import math
import sys


def vid(i, j, C):
    return i * (C + 1) + j


def main():
    data = sys.stdin.read().split()
    ptr = 0
    R = int(data[ptr]); ptr += 1
    C = int(data[ptr]); ptr += 1
    B = float(data[ptr]); ptr += 1
    h = [[0.0] * (C + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        for j in range(C + 1):
            h[i][j] = float(data[ptr]); ptr += 1

    def vlen(i, j):
        dz = h[i + 1][j] - h[i][j]
        return math.sqrt(1.0 + dz * dz)

    def cut_cost(j0):
        return sum(vlen(i, j0) for i in range(R))

    def boundaries_for(K):
        bs = sorted(set(round(k * C / K) for k in range(1, K)))
        bs = [b for b in bs if 1 <= b <= C - 1]
        return bs

    best_K = 1
    best_boundaries = []
    for K in range(1, C + 1):
        bs = boundaries_for(K)
        cost = sum(cut_cost(b) for b in bs)
        if cost <= B + 1e-9:
            best_K = K
            best_boundaries = bs
        else:
            break

    # assign each cell-column j (0..C-1) a strip id by which boundary bracket it falls in
    bnds = best_boundaries
    strip_of_col = []
    for j in range(C):
        sid = 0
        for b in bnds:
            if j >= b:
                sid += 1
        strip_of_col.append(sid)

    out = []
    for i in range(R):
        for j in range(C):
            pid = strip_of_col[j]
            out.append(str(pid))
            out.append(str(pid))
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
