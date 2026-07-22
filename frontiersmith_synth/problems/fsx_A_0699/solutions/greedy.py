# TIER: greedy
# The "obvious" rotationally-symmetric rosette: a per-ring grid of k equal
# angular wedges (one face per ring-per-wedge cell), checkerboard-coloured with
# whichever 2 palette colours look best together by the harmony table. This is
# exactly k-fold symmetric and looks like a natural stained-glass lattice -- but
# it never questions whether k is odd: the ring of k wedges is then an odd
# cycle, and no pair of 2 colours (however well matched) can properly colour an
# odd cycle, regardless of how large the palette p is.
import sys


def main():
    data = sys.stdin.read().split("\n")
    R, A, k, p = [int(x) for x in data[0].split()]
    H = []
    for r in range(p):
        H.append([float(x) for x in data[2 + r].split()])

    best_pair = (1, 2)
    best_h = -1.0
    for c1 in range(1, p + 1):
        for c2 in range(c1 + 1, p + 1):
            if H[c1 - 1][c2 - 1] > best_h:
                best_h = H[c1 - 1][c2 - 1]
                best_pair = (c1, c2)

    step = A // k  # wedge width in sectors
    out = []
    for r in range(R):
        for a in range(A):
            wedge = a // step
            fid = r * k + wedge
            color = best_pair[0] if (r + wedge) % 2 == 0 else best_pair[1]
            out.append("%d %d" % (fid, color))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
