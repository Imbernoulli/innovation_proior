# TIER: trivial
# Reproduces the checker's airspace-aware equal-radius grid baseline exactly -> Ratio ~= 0.1.
import sys, math


def baseline_disks(N, cx, cy, R, zones):
    gx = int(math.ceil(math.sqrt(N)))
    if gx < 1:
        gx = 1
    gy = int(math.ceil(N / float(gx)))
    cw = (2.0 * R) / gx
    ch = (2.0 * R) / gy
    r = min(cw, ch) * 0.5 * 0.999
    x0 = cx - R
    y0 = cy - R
    disks = []
    for j in range(gy):
        for i in range(gx):
            if len(disks) >= N:
                break
            px = x0 + (i + 0.5) * cw
            py = y0 + (j + 0.5) * ch
            if math.hypot(px - cx, py - cy) + r > R:
                continue
            ok = True
            for (zx, zy, zr) in zones:
                if math.hypot(px - zx, py - zy) < r + zr:
                    ok = False
                    break
            if not ok:
                continue
            for (ox, oy, orr) in disks:
                if math.hypot(px - ox, py - oy) < r + orr - 1e-12:
                    ok = False
                    break
            if ok:
                disks.append((px, py, r))
    return disks


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); cx = float(toks[1]); cy = float(toks[2])
    R = float(toks[3]); K = int(toks[4])
    zones = []
    idx = 5
    for _ in range(K):
        zones.append((float(toks[idx]), float(toks[idx + 1]), float(toks[idx + 2])))
        idx += 3
    disks = baseline_disks(N, cx, cy, R, zones)
    out = [str(len(disks))]
    for (x, y, r) in disks:
        out.append("%r %r %r" % (x, y, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
