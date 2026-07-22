#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE symmetry-generator-discovery instance to stdout.
Deterministic: everything is seeded from testId only. Prints VISIBLE data rows only
(never the hidden center/direction/coefficients -- those live only inside verify.py's
own independent copy of this construction)."""
import sys, math, random

TYPES = ["rotation", "scaling", "translation", "rotation", "scaling",
         "rotation", "scaling", "translation", "rotation", "scaling"]
GCOUNT = [14, 13, 12, 11, 10, 9, 9, 8, 7, 6]
ARCW   = [1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.75, 0.65, 0.55]
QW     = [1.3, 1.2, 1.1, 1.0, 0.9, 0.85, 0.75, 0.7, 0.6, 0.5]
EPS = 0.04
NHELD = 6


def _make_group(ax, ay, truelabel):
    pts = [(ax, ay), (ax + EPS, ay), (ax - EPS, ay), (ax, ay + EPS), (ax, ay - EPS)]
    return [(x, y, truelabel(x, y)) for (x, y) in pts]


def build_instance(tid):
    tid = int(tid)
    idx = (tid - 1) % 10
    typ = TYPES[idx]
    rnd = random.Random(90000 + 131 * tid)
    groups = []
    heldout = []

    if typ == "rotation":
        hx = rnd.uniform(-2.0, 2.0); hy = rnd.uniform(-2.0, 2.0)
        b0 = rnd.uniform(-1.0, 1.0); b1 = rnd.uniform(0.8, 1.6)
        b2 = rnd.uniform(-0.15, 0.15); b3 = rnd.uniform(-0.6, 0.6)

        def hfun(r):
            return b0 + b1 * r + b2 * r * r + b3 * math.sin(2.0 * r)

        def truelabel(x, y):
            r = math.hypot(x - hx, y - hy)
            return hfun(r)

        arc = ARCW[idx]
        for _ in range(GCOUNT[idx]):
            r = rnd.uniform(3.0, 4.0); th = rnd.uniform(0.0, arc)
            ax = hx + r * math.cos(th); ay = hy + r * math.sin(th)
            groups.append(_make_group(ax, ay, truelabel))
        for _ in range(NHELD):
            # SAME radius band as training (the invariant coordinate must stay
            # reachable), but an angle far OUTSIDE the training arc: this point
            # is unreachable by interpolation yet on a training-covered orbit.
            r = rnd.uniform(2.9, 4.1)
            th = rnd.uniform(arc + 0.5, 2.0 * math.pi - 0.15)
            x = hx + r * math.cos(th); y = hy + r * math.sin(th)
            heldout.append((x, y, truelabel(x, y)))

    elif typ == "scaling":
        hx = rnd.uniform(-2.0, 2.0); hy = rnd.uniform(-2.0, 2.0)
        b0 = rnd.uniform(-1.0, 1.0); b1 = rnd.uniform(-1.3, 1.3)
        b2 = rnd.uniform(-1.3, 1.3); b3 = rnd.uniform(-0.5, 0.5)

        def hfun(th):
            return b0 + b1 * math.cos(th) + b2 * math.sin(th) + b3 * math.cos(2.0 * th)

        def truelabel(x, y):
            th = math.atan2(y - hy, x - hx)
            return hfun(th)

        arc = ARCW[idx]
        for _ in range(GCOUNT[idx]):
            r = rnd.uniform(1.0, 2.0); th = rnd.uniform(0.0, arc)
            ax = hx + r * math.cos(th); ay = hy + r * math.sin(th)
            groups.append(_make_group(ax, ay, truelabel))
        for _ in range(NHELD):
            # SAME angular sector as training (the invariant coordinate stays
            # reachable), but a radius far OUTSIDE the training band.
            r = rnd.uniform(4.0, 8.0)
            th = rnd.uniform(0.0, arc)
            x = hx + r * math.cos(th); y = hy + r * math.sin(th)
            heldout.append((x, y, truelabel(x, y)))

    else:  # translation
        phi = rnd.uniform(0.0, 2.0 * math.pi)
        ux, uy = math.cos(phi), math.sin(phi)
        vx, vy = -math.sin(phi), math.cos(phi)
        b0 = rnd.uniform(-1.0, 1.0)
        sgn = rnd.choice([-1.0, 1.0]); b1 = sgn * rnd.uniform(0.5, 1.3)
        b2 = rnd.uniform(-0.1, 0.1); b3 = rnd.uniform(-0.5, 0.5)

        def hfun(q):
            return b0 + b1 * q + b2 * q * q + b3 * math.sin(0.6 * q)

        def truelabel(x, y):
            q = x * vx + y * vy
            return hfun(q)

        qw = QW[idx]
        for _ in range(GCOUNT[idx]):
            p = rnd.uniform(-1.5, 1.5); q = rnd.uniform(0.3, 0.3 + qw)
            ax = p * ux + q * vx; ay = p * uy + q * vy
            groups.append(_make_group(ax, ay, truelabel))
        for _ in range(NHELD):
            # SAME q-band as training (the invariant coordinate stays
            # reachable), but a p_par far OUTSIDE the training range.
            p = rnd.choice([-1.0, 1.0]) * rnd.uniform(4.0, 9.0)
            q = rnd.uniform(0.3, 0.3 + qw)
            x = p * ux + q * vx; y = p * uy + q * vy
            heldout.append((x, y, truelabel(x, y)))

    return {"type": typ, "groups": groups, "heldout": heldout, "truelabel": truelabel}


def main():
    tid = int(sys.argv[1])
    inst = build_instance(tid)
    groups = inst["groups"]
    out = []
    out.append(str(tid))
    out.append("%d %.6f" % (len(groups), EPS))
    for g in groups:
        for (x, y, lab) in g:
            out.append("%.9g %.9g %.9g" % (x, y, lab))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
