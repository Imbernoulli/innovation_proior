# TIER: strong
# Discovers the angular momentum L = x*vy - y*vx by searching a small family of
# bilinear forms in the state and picking the one that is most constant within
# training orbits (lowest within-orbit / total variance).  L needs no fitted
# constant and, unlike the energy, carries no noise-amplifying 1/r term, so it is
# the cleanest first integral -> the highest, but noise still bounds it below 1.
import sys, math

# candidate bilinear/quadratic monomials in the state (x1,x2,x3,x4)
def feats(s):
    x1, x2, x3, x4 = s
    return [x1 * x4, x2 * x3, x1 * x3, x2 * x4,
            x1 * x1 + x2 * x2, x3 * x3 + x4 * x4, x1 * x2, x3 * x4]

# candidate expressions: (label emitted, function over state)
CANDS = [
    ("1.0 * (x1*x4 - x2*x3)", lambda s: s[0] * s[3] - s[1] * s[2]),
    ("1.0 * (x2*x3 - x1*x4)", lambda s: s[1] * s[2] - s[0] * s[3]),
    ("1.0 * (x1*x3 + x2*x4)", lambda s: s[0] * s[2] + s[1] * s[3]),
    ("1.0 * (x1*x1 + x2*x2)", lambda s: s[0] ** 2 + s[1] ** 2),
    ("1.0 * (x3*x3 + x4*x4)", lambda s: s[2] ** 2 + s[3] ** 2),
]


def load():
    data = sys.stdin.read().split("\n")
    h = data[0].split()
    J = int(h[0])
    trajs = [[] for _ in range(J)]
    for ln in data[1:]:
        p = ln.split()
        if len(p) >= 5:
            j = int(p[0])
            trajs[j].append(tuple(map(float, p[1:5])))
    return [t for t in trajs if t]


def within_over_total(trajs, f):
    vals = []
    groups = []
    for tr in trajs:
        gv = [f(s) for s in tr]
        groups.append(gv)
        vals.extend(gv)
    gm = sum(vals) / len(vals)
    sst = sum((v - gm) ** 2 for v in vals)
    if sst < 1e-12:
        return 1.0
    ssw = 0.0
    for gv in groups:
        m = sum(gv) / len(gv)
        ssw += sum((v - m) ** 2 for v in gv)
    return ssw / sst


def main():
    trajs = load()
    best = None
    for label, f in CANDS:
        w = within_over_total(trajs, f)
        if best is None or w < best[0]:
            best = (w, label)
    sys.stdout.write(best[1] + "\n")


if __name__ == "__main__":
    main()
