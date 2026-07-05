import sys, random

# Bakery supply-chain sampling: choose M test points in the unit square
# [flour-fraction] x [proof-time] to cover the recipe space as evenly as
# possible (low star discrepancy). K of the points are already-fixed
# "signature recipes" (anchors) that MUST appear in the submitted set.

def main():
    i = int(sys.argv[1])
    rng = random.Random(9100 + i)

    d = 2  # (flour fraction, proof time) -- fixed 2-D recipe space
    # difficulty ladder: number of sample points grows small -> larger
    M_ladder = [8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
    M = M_ladder[(i - 1) % len(M_ladder)]
    # number of fixed signature recipes (anchors) that must be included
    K = 2 + ((i - 1) % 3)  # 2,3,4 cycling
    if K > M:
        K = M

    # anchors deterministically drawn inside (0.05, 0.95)^2
    anchors = []
    for _ in range(K):
        ax = round(0.05 + 0.90 * rng.random(), 6)
        ay = round(0.05 + 0.90 * rng.random(), 6)
        anchors.append((ax, ay))

    out = ["%d %d %d" % (d, M, K)]
    for (ax, ay) in anchors:
        out.append("%.6f %.6f" % (ax, ay))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
