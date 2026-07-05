# TIER: greedy
# Energy-like invariant with a FITTED constant.  Guess the specific-energy form
#     E(mu') = 0.5*(vx^2+vy^2) - mu'/r,   r = sqrt(x^2+y^2)
# and 1-D search mu' to minimise the within-orbit variance of E on the training
# orbits (normalised by its global spread).  Recovers the shape of the conserved
# energy, but the 1/r term is very noise-sensitive near perihelion, so it is a
# noisier invariant than the clean angular momentum -> medium score.
import sys, math


def load():
    data = sys.stdin.read().split("\n")
    h = data[0].split()
    J, K = int(h[0]), int(h[1])
    trajs = [[] for _ in range(J)]
    for ln in data[1:]:
        p = ln.split()
        if len(p) >= 5:
            j = int(p[0])
            x1, x2, x3, x4 = map(float, p[1:5])
            trajs[j].append((x1, x2, x3, x4))
    return [t for t in trajs if t]


def within_over_total(trajs, mu):
    vals = []
    groups = []
    for tr in trajs:
        gv = []
        for (x1, x2, x3, x4) in tr:
            r = math.sqrt(x1 * x1 + x2 * x2)
            gv.append(0.5 * (x3 * x3 + x4 * x4) - mu / r)
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
    for i in range(1, 400):
        mu = 0.2 + i * 0.005          # scan mu' in (0.2, 2.2]
        w = within_over_total(trajs, mu)
        if best is None or w < best[0]:
            best = (w, mu)
    mu = best[1]
    sys.stdout.write("0.5 * (x3**2 + x4**2) - %r / sqrt(x1**2 + x2**2)\n" % mu)


if __name__ == "__main__":
    main()
