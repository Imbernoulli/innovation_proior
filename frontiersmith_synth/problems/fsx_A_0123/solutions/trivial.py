# TIER: trivial
# Reproduces the checker's obstacle-aware equal-radius grid baseline exactly -> Ratio ~= 0.1.
import sys, math


def dist_pt_rect(px, py, rect):
    x0, y0, x1, y1 = rect
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def baseline_disks(N, W, H, racks):
    r = min(W / (2.0 * N), H * 0.5) * 0.999
    cy = H * 0.5
    disks = []
    for i in range(N):
        cx = (i + 0.5) * W / N
        if cx - r < 0 or cx + r > W or cy - r < 0 or cy + r > H:
            continue
        ok = True
        for rk in racks:
            if dist_pt_rect(cx, cy, rk) < r:
                ok = False
                break
        if ok:
            disks.append((cx, cy, r))
    return disks


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); W = float(toks[1]); H = float(toks[2]); K = int(toks[3])
    racks = []
    idx = 4
    for _ in range(K):
        racks.append((float(toks[idx]), float(toks[idx + 1]),
                      float(toks[idx + 2]), float(toks[idx + 3])))
        idx += 4
    disks = baseline_disks(N, W, H, racks)
    out = [str(len(disks))]
    for (x, y, r) in disks:
        out.append("%r %r %r" % (x, y, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
