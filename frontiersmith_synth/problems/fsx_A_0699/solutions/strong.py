# TIER: strong
# Insight: a k-wedge design forces a k-length adjacency cycle around the centre,
# which is properly 2-colourable only when k is EVEN (odd cycles need >=3 colours).
# Rather than pick the geometry first and discover the obstruction afterward, we
# co-design geometry and colour: always work at DOUBLE the angular resolution
# (m = 2k arcs per ring instead of k). 2k is even for every k, so a simple
# checkerboard 2-colouring is ALWAYS valid, sidestepping the parity trap for any
# k while also giving finer (more numerous) faces for entropy. We then search a
# small set of radial band counts (extra face granularity) and the best colour
# PAIR from the palette (by harmony), scoring each candidate with the exact
# closed-form entropy + boundary-crossing count, and materialise the winner.
import sys, math


def main():
    data = sys.stdin.read().split("\n")
    R, A, k, p = [int(x) for x in data[0].split()]
    we, wh = [float(x) for x in data[1].split()]
    H = []
    for r in range(p):
        H.append([float(x) for x in data[2 + r].split()])

    m = 2 * k                      # always even -> checkerboard always valid
    arcw = A // m

    b_candidates = sorted(set(b for b in range(1, R + 1) if R % b == 0))

    best = None  # (F, b, c1, c2)
    for b in b_candidates:
        faces = m * b
        Hent = math.log2(faces)
        cross_edges = R * m + (b - 1) * A
        for c1 in range(1, p + 1):
            for c2 in range(c1 + 1, p + 1):
                harmony = cross_edges * H[c1 - 1][c2 - 1]
                F = we * Hent + wh * harmony
                if best is None or F > best[0]:
                    best = (F, b, c1, c2)

    _, b, c1, c2 = best
    band_height = R // b

    out = []
    for r in range(R):
        band = r // band_height
        for a in range(A):
            arc = a // arcw
            fid = band * m + arc
            color = c1 if (band + arc) % 2 == 0 else c2
            out.append("%d %d" % (fid, color))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
