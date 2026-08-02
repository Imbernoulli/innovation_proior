import sys, math, random

# ---------------------------------------------------------------------------
# structural-damage-modal (format C, MAXIMIZE a quality metric).
#   `python3 gen.py <testId>` prints ONE instance to stdout. Deterministic in
#   testId only.
#
# Physical model (self-contained, defined precisely in statement.md too):
#   A 1-D structure of length L has mode shapes phi_m(x) = cos(m*pi*x/L),
#   m = 1,2,3,... (a free-free-type family). Mode m has m interior NODES
#   (zeros of phi_m) at x = (2j-1)*L/(2m), j=1..m -- points where mode m is
#   perfectly insensitive to local damage.
#
#   A single crack of unknown location x* in (0,L) and severity s* (small
#   fractional stiffness loss) perturbs:
#     - frequency of every measured mode m:
#         f_m = f0_m * (1 - s* * phi_m(x*)^2) + measurement noise
#       (zero shift exactly at a node of mode m -- "node-line blindness")
#     - the mode SHAPE itself, sampled coarsely at G gauge points:
#         psi_m(x_g) = phi_m(x_g) * (1 - s* * bump(x_g - x*)) + noise
#       bump(d) = exp(-(d/w)^2), w = 0.10*L (a local dent around the crack).
#
# The generator hides x* and s* -- they are NEVER printed. verify.py
# recomputes them by re-running this exact deterministic construction from
# testId (the harness sandboxes solutions so they cannot read this file).
#
# TRAP (>=3 of 10 cases, here 5): x* is placed EXACTLY at a node of the
# FIRST (lowest-index) measured mode m1. A single-mode method built around
# m1 alone (the "obvious" first instinct: trust the lowest available mode)
# sees ~zero frequency shift AND a locally-vanishing shape signal for m1
# specifically -- it is genuinely blind there. Other measured modes, whose
# node lines land elsewhere, are not blind at x* -- fusing them is the
# insight a strong solver needs.
# ---------------------------------------------------------------------------

L_LADDER = [10, 14, 18, 22, 28, 34, 42, 52, 65, 80]
K_LADDER = [3, 3, 4, 4, 4, 5, 5, 5, 6, 6]
G_LADDER = [7, 8, 8, 9, 9, 10, 10, 11, 12, 13]
TRAP_IDS = {3, 4, 5, 7, 9}

KMAX = 14
S_MIN, S_MAX = 0.08, 0.30
SIGMA_F = 0.0035
SIGMA_SHAPE = 0.012
W_BUMP_FRAC = 0.10


def _phi(m, x, L):
    return math.cos(m * math.pi * x / L)


def _nearest_node_dist(m, x, L):
    jf = (x * 2 * m / L + 1) / 2.0
    j = max(1, min(m, round(jf)))
    node = (2 * j - 1) * L / (2 * m)
    return abs(x - node)


def build_instance(t):
    t = max(1, min(10, t))
    idx = t - 1
    L, K, G = L_LADDER[idx], K_LADDER[idx], G_LADDER[idx]
    rng = random.Random(31337 + 101 * t)

    modes = sorted(rng.sample(range(2, KMAX + 1), K))
    m1 = modes[0]
    sstar = round(rng.uniform(S_MIN, S_MAX), 4)

    if t in TRAP_IDS:
        js = list(range(1, m1 + 1))
        rng.shuffle(js)
        xstar = None
        for j in js:
            cand = (2 * j - 1) * L / (2 * m1)
            if abs(cand - L / 2) > 0.08 * L and 0.03 * L < cand < 0.97 * L:
                xstar = cand
                break
        if xstar is None:
            xstar = 0.31 * L
    else:
        xstar = None
        for _ in range(300):
            cand = rng.uniform(0.05 * L, 0.95 * L)
            if abs(cand - L / 2) < 0.08 * L:
                continue
            if all(_nearest_node_dist(m, cand, L) > 0.12 * (L / m) for m in modes):
                xstar = cand
                break
        if xstar is None:
            xstar = 0.4 * L

    f0, fdam = [], []
    for m in modes:
        base = 3.0 + 0.7 * m + 0.15 * (m % 3)
        shift = sstar * _phi(m, xstar, L) ** 2
        noise = rng.gauss(0.0, SIGMA_F)
        f0.append(base)
        fdam.append(base * (1.0 - shift + noise))

    gpts = [L * g / (G + 1) for g in range(1, G + 1)]
    w = W_BUMP_FRAC * L
    shape_u, shape_d = [], []
    for m in modes:
        urow = [_phi(m, xg, L) for xg in gpts]
        drow = []
        for xg in gpts:
            bump = math.exp(-((xg - xstar) / w) ** 2)
            val = _phi(m, xg, L) * (1.0 - sstar * bump) + rng.gauss(0.0, SIGMA_SHAPE)
            drow.append(val)
        shape_u.append(urow)
        shape_d.append(drow)

    return dict(t=t, L=L, K=K, G=G, modes=modes, sstar=sstar, xstar=xstar,
                f0=f0, fdam=fdam, gpts=gpts, shape_u=shape_u, shape_d=shape_d)


def main():
    t = int(sys.argv[1])
    inst = build_instance(t)
    L, K, G = inst["L"], inst["K"], inst["G"]

    out = []
    out.append("%d %d %d %d" % (inst["t"], L, G, K))
    out.append(" ".join(str(m) for m in inst["modes"]))
    out.append(" ".join("%.6f" % v for v in inst["f0"]))
    out.append(" ".join("%.6f" % v for v in inst["fdam"]))
    out.append(" ".join("%.6f" % v for v in inst["gpts"]))
    for row in inst["shape_u"]:
        out.append(" ".join("%.6f" % v for v in row))
    for row in inst["shape_d"]:
        out.append(" ".join("%.6f" % v for v in row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
