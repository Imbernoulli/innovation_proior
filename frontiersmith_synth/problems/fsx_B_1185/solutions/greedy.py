# TIER: greedy
# The "obvious first instinct": trust a SINGLE mode -- the lowest-index
# measured mode m1 (in practice the fundamental/easiest mode to measure
# reliably) -- and nothing else. Locate the crack by peak-picking the
# largest observed shape distortion |damaged - undamaged| for mode m1 across
# the coarse gauge points, then back out severity by inverting mode m1's OWN
# frequency-shift equation at that location. This is a textbook single-mode
# damage index -- it never looks at any other measured mode, and never
# fuses the shape and frequency channels of a DIFFERENT mode. It is exactly
# what the "node-line blindness" trap punishes: if the crack sits on mode
# m1's node line, BOTH of mode m1's channels go dark there simultaneously
# (frequency shift ~0 and local shape distortion ~0, since damage severity
# multiplies the already-near-zero mode-1 shape amplitude), so this method
# either reports ~0 severity or amplifies pure noise into a wild guess.
import sys, math


def main():
    data = sys.stdin.read().split()
    p = 0
    t = int(data[p]); p += 1
    L = int(data[p]); p += 1
    G = int(data[p]); p += 1
    K = int(data[p]); p += 1
    modes = [int(data[p + i]) for i in range(K)]; p += K
    f0 = [float(data[p + i]) for i in range(K)]; p += K
    fdam = [float(data[p + i]) for i in range(K)]; p += K
    gpts = [float(data[p + i]) for i in range(G)]; p += G
    shape_u = []
    for i in range(K):
        shape_u.append([float(data[p + j]) for j in range(G)]); p += G
    shape_d = []
    for i in range(K):
        shape_d.append([float(data[p + j]) for j in range(G)]); p += G

    S_MAX_OUT = 0.5
    m1 = modes[0]

    delta = [abs(shape_d[0][g] - shape_u[0][g]) for g in range(G)]
    gi = max(range(G), key=lambda g: delta[g])
    x_hat = gpts[gi]

    r1 = 1.0 - fdam[0] / f0[0]
    c = math.cos(m1 * math.pi * x_hat / L) ** 2
    if c < 0.05:
        s_hat = 0.0  # mode 1 is (near) its own node here -- "no damage found"
    else:
        s_hat = max(0.0, min(S_MAX_OUT, r1 / c))

    print("%.6f %.6f" % (x_hat, s_hat))


if __name__ == "__main__":
    main()
