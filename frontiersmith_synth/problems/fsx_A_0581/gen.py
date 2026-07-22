#!/usr/bin/env python3
# Deterministic generator for "One Artery Feeds Every Organ".
#   python3 gen.py <testId>   ->  one instance on stdout, seeded by testId only.
#
# Layout: an arterial source at the origin and several organ CLUSTERS strung along
# ONE narrow directional CORRIDOR at DISTINCT radii (all roughly "downstream" of the
# source). Because every cluster shares a direction, the cheap network is a HIERARCHY
# / SPINE that SPLITS LATE: one thick trunk leaves the source, pools the flow up the
# corridor, taps each cluster in turn, and only then fans out to terminals (Murray
# material is concave in flow, so pooling the long near-source haul is where the money
# is). The trap: a FLAT star of per-cluster hubs -- the obvious "one hub per cluster"
# move -- makes every cluster pay the shared near-source haul on its own and never
# pools the corridor, so it lands far below the spine. Obstacles never block a
# straight source->organ line, so the star is always feasible.
import sys, math, random, os

# testId -> (num clusters, organs target, cone half-width rad, radius spread, n_obstacles)
# Clusters sit along ONE narrow corridor at DISTINCT radii: the cheap network runs a
# single trunk up the corridor that pools flow and taps each cluster in turn (a spine
# that splits late), so per-cluster flat hubs -- each paying the shared near-source
# haul on its own -- fall far short.
LADDER = [
    (3, 27, 0.11, (40, 250), 0),
    (4, 32, 0.10, (38, 265), 0),
    (4, 36, 0.10, (36, 275), 0),
    (4, 36, 0.10, (34, 285), 1),
    (5, 40, 0.09, (34, 290), 1),
    (5, 45, 0.09, (32, 295), 1),
    (5, 45, 0.09, (32, 300), 2),
    (6, 48, 0.09, (30, 305), 2),
    (6, 54, 0.08, (30, 315), 2),
    (6, 54, 0.08, (28, 320), 2),
]

Wm = 1.0                                        # Murray material weight
Wd = float(os.environ.get("FSX_WD", "0.03"))    # linear delivery weight (env override for tuning)


def seg_hits_rect(ax, ay, bx, by, rect):
    x0, y0, x1, y1 = rect
    dx = bx - ax; dy = by - ay
    p = [-dx, dx, -dy, dy]
    q = [ax - x0, x1 - ax, ay - y0, y1 - ay]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-15:
            if qi < 0:
                return False
        else:
            r = qi / pi
            if pi < 0:
                if r > t1:
                    return False
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return False
                if r < t1:
                    t1 = r
    return t1 - t0 > 1e-9


def main():
    tid = int(sys.argv[1])
    idx = min(max(tid, 1), len(LADDER)) - 1
    ncl, Ktarget, hw, (Rmin, Rmax), nobs = LADDER[idx]
    rng = random.Random(70581 * 1000 + tid)

    base_ang = rng.uniform(0.0, 2.0 * math.pi)
    sinks = []
    used = set()
    per = max(1, Ktarget // ncl)
    remaining = Ktarget

    for c in range(ncl):
        sz = per if c < ncl - 1 else remaining
        remaining -= sz
        # clusters at DISTINCT radii along the corridor (small angular jitter)
        frac = c / max(1, ncl - 1)
        cr = Rmin + (Rmax - Rmin) * frac
        ca = base_ang + rng.uniform(-hw, hw)
        cx = cr * math.cos(ca); cy = cr * math.sin(ca)
        sigma = rng.uniform(6.0, 13.0)          # tight blob
        # cluster demand level: heterogeneous across clusters
        lvl = rng.choice([1.0, 1.0, 2.0, 3.0])
        placed = 0; tries = 0
        while placed < sz and tries < 400:
            tries += 1
            px = cx + rng.gauss(0.0, sigma)
            py = cy + rng.gauss(0.0, sigma)
            key = (round(px, 3), round(py, 3))
            if key in used:
                continue
            used.add(key)
            d = max(1.0, lvl + rng.choice([0.0, 0.0, 1.0]))
            sinks.append((px, py, d))
            placed += 1

    K = len(sinks)

    # obstacles: rectangles between cluster corridors that never cross a straight
    # source->organ line and hold no organ (so the star stays feasible), placed to
    # obstruct naive shared-trunk shortcuts.
    rects = []
    tries = 0
    while len(rects) < nobs and tries < 600:
        tries += 1
        ang = base_ang + rng.uniform(-1.2, 1.2)
        rr = rng.uniform(60.0, 140.0)
        cx = rr * math.cos(ang); cy = rr * math.sin(ang)
        w = rng.uniform(12.0, 26.0); h = rng.uniform(12.0, 26.0)
        rect = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if rect[0] <= 0.0 <= rect[2] and rect[1] <= 0.0 <= rect[3]:
            continue
        ok = True
        for (x, y, _d) in sinks:
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                ok = False; break
        if not ok:
            continue
        for (x, y, _d) in sinks:
            if seg_hits_rect(0.0, 0.0, x, y, rect):
                ok = False; break
        if ok:
            rects.append(rect)

    M = len(rects)
    out = ["%d %d %.6f %.6f" % (K, M, Wm, Wd), "%.6f %.6f" % (0.0, 0.0)]
    for (x, y, d) in sinks:
        out.append("%.6f %.6f %.6f" % (x, y, d))
    for r in rects:
        out.append("%.6f %.6f %.6f %.6f" % r)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
