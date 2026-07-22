import sys, random, math

T = 260
K = 1_000_000.0
A_FRAC = 0.25
COLLAPSE_FRAC = 0.06
GAMMA = 0.9985
P_BASE = 10.0
QCAP = 0.05 * K
N_FISHERS = 300
CLOSED_MOD = list(range(34, 39))  # 5-week legal spawning closure, every 52-week year


def price_season():
    return [1.0 + 0.28 * math.sin(2 * math.pi * w / 52.0 + 0.7)
            + 0.05 * math.sin(2 * math.pi * 3 * w / 52.0) for w in range(52)]


def main():
    i = int(sys.argv[1])
    rng = random.Random(9001 + 17 * i)

    A = A_FRAC * K
    Acol = COLLAPSE_FRAC * K
    S0 = rng.uniform(0.55, 0.72) * K
    r_base = rng.uniform(0.055, 0.085)
    dstart = rng.randint(70, 170)
    dlen = rng.randint(30, 60)
    dend = min(T, dstart + dlen)
    drought_mult = rng.uniform(0.12, 0.35)

    costs = [rng.uniform(0.5, 9.5) for _ in range(N_FISHERS)]
    caps = [rng.uniform(500.0, 3000.0) for _ in range(N_FISHERS)]

    ps = price_season()

    out = []
    out.append(str(T))
    out.append("%.6f %.6f %.6f %.6f %.6f %.6f" % (K, A, Acol, GAMMA, P_BASE, QCAP))
    out.append("%.6f %.6f" % (S0, r_base))
    out.append("%d %d %.6f" % (dstart, dend, drought_mult))
    out.append(str(len(CLOSED_MOD)))
    out.append(" ".join(str(w) for w in CLOSED_MOD))
    out.append(str(len(ps)))
    out.append(" ".join("%.6f" % v for v in ps))
    out.append(str(N_FISHERS))
    for c, cap in zip(costs, caps):
        out.append("%.6f %.6f" % (c, cap))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
