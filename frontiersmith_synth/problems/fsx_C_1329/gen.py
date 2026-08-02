#!/usr/bin/env python3
"""gen.py <testId> -- zeolite-template-choose instance generator.
Deterministic: all randomness seeded from testId (+ a fixed problem salt).
Prints one instance to stdout. No answer file is produced (format C).
"""
import sys
import random

T_NORM = 100.0
PH_NORM = 3.0
Q_NORM = 2.0
F_NORM = 1.0
SALT = 1329


def f_ideal(c):
    return 0.15 * c + 0.1


# Per-case plan: (K, trap_type)
# trap_type in {"none", "sweetspot_outside", "window_disjoint", "high_removal_cost"}
PLAN = {
    1: (6, "none"),
    2: (8, "none"),
    3: (10, "sweetspot_outside"),
    4: (12, "window_disjoint"),
    5: (15, "high_removal_cost"),
    6: (20, "sweetspot_outside"),
    7: (25, "none"),
    8: (40, "window_disjoint"),
    9: (60, "high_removal_cost"),
    10: (100, "sweetspot_outside"),
}


def fmt(x):
    return "%.6f" % x


def gen(test_id):
    K, trap = PLAN[test_id]
    rng = random.Random(test_id * 1000003 + SALT)

    c = [1, 2, 3][(test_id - 1) % 3]
    D_target = round(rng.uniform(4.0, 9.0), 4)
    q_target = round(rng.uniform(-1.2, 1.2), 4)

    Tf_lo = round(rng.uniform(120.0, 170.0), 3)
    Tf_width = round(rng.uniform(35.0, 65.0), 3)
    Tf_hi = Tf_lo + Tf_width
    pHf_lo = round(rng.uniform(8.5, 11.0), 3)
    pHf_width = round(rng.uniform(1.8, 3.0), 3)
    pHf_hi = pHf_lo + pHf_width

    Tf_mid = (Tf_lo + Tf_hi) / 2.0
    pHf_mid = (pHf_lo + pHf_hi) / 2.0

    # weights (positive, sum to 1)
    a, b, d = rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0)
    tot = a + b + d
    w1, w2, w3 = a / tot, b / tot, d / tot

    templates = []

    # --- template 0: baseline / always-safe reference (window fully contains
    # the framework window; sweet spot at the framework center) ---
    fid = f_ideal(c)
    s0 = D_target * 1.75
    q0 = q_target + 1.6
    f0 = fid + 0.85
    t0 = dict(
        s=s0, q=q0, f=f0,
        Tlo=Tf_lo - 5.0, Thi=Tf_hi + 5.0,
        pHlo=pHf_lo - 0.3, pHhi=pHf_hi + 0.3,
        Topt=Tf_mid, pHopt=pHf_mid,
        R=5.0, r=0.10,
    )
    templates.append(t0)

    # --- template 1: the "decoy" -- near-perfect geometric fit, but its
    # reachability / cost depends on trap_type ---
    s1 = D_target * (1.0 + rng.uniform(-0.05, 0.05))
    q1 = q_target + rng.uniform(-0.2, 0.2)
    f1 = fid + rng.uniform(-0.1, 0.1)

    if trap == "none":
        decoy = dict(
            s=s1, q=q1, f=f1,
            Tlo=Tf_lo - 5.0, Thi=Tf_hi + 5.0,
            pHlo=pHf_lo - 0.3, pHhi=pHf_hi + 0.3,
            Topt=Tf_mid, pHopt=pHf_mid,
            R=1.0, r=0.15,
        )
    elif trap == "sweetspot_outside":
        # windows overlap (share exactly the framework's own T-range) but the
        # decoy's kinetic optimum sits well outside that range.
        overshoot = rng.uniform(18.0, 30.0)
        decoy = dict(
            s=s1, q=q1, f=f1,
            Tlo=Tf_lo - 10.0, Thi=Tf_hi + 40.0,
            pHlo=pHf_lo - 0.3, pHhi=pHf_hi + 0.3,
            Topt=Tf_hi + overshoot, pHopt=pHf_mid,
            R=0.6, r=0.60,
        )
    elif trap == "window_disjoint":
        # decoy's stability window does not overlap the framework window at all
        gap = rng.uniform(20.0, 40.0)
        decoy = dict(
            s=s1, q=q1, f=f1,
            Tlo=Tf_hi + gap, Thi=Tf_hi + gap + 50.0,
            pHlo=pHf_lo - 0.3, pHhi=pHf_hi + 0.3,
            Topt=Tf_hi + gap + 20.0, pHopt=pHf_mid,
            R=1.0, r=0.30,
        )
    elif trap == "high_removal_cost":
        decoy = dict(
            s=s1, q=q1, f=f1,
            Tlo=Tf_lo - 5.0, Thi=Tf_hi + 5.0,
            pHlo=pHf_lo - 0.3, pHhi=pHf_hi + 0.3,
            Topt=Tf_mid, pHopt=pHf_mid,
            R=1.0, r=0.85,
        )
    else:
        raise ValueError(trap)
    templates.append(decoy)

    # --- template 2: the "good alternative" -- moderate fit, robust window,
    # cheap to remove ---
    s2 = D_target * (1.0 + rng.uniform(-0.22, 0.22))
    q2 = q_target + rng.uniform(-0.55, 0.55)
    f2 = fid + rng.uniform(-0.32, 0.32)
    Toff = rng.uniform(-0.12, 0.12) * Tf_width
    pHoff = rng.uniform(-0.12, 0.12) * pHf_width
    alt = dict(
        s=s2, q=q2, f=f2,
        Tlo=Tf_lo - 5.0, Thi=Tf_hi + 5.0,
        pHlo=pHf_lo - 0.3, pHhi=pHf_hi + 0.3,
        Topt=Tf_mid + Toff, pHopt=pHf_mid + pHoff,
        R=1.2, r=round(rng.uniform(0.10, 0.25), 4),
    )
    templates.append(alt)

    # --- filler templates: random noise, bounded well below the decoy's SDI ---
    for _ in range(K - 3):
        sf = D_target * (1.0 + rng.uniform(-0.9, 0.9))
        qf = q_target + rng.uniform(-2.0, 2.0)
        ff = fid + rng.uniform(-1.5, 1.5)
        window_ok = rng.random() < 0.5
        if window_ok:
            Tlo = Tf_lo - rng.uniform(0, 10)
            Thi = Tf_hi + rng.uniform(0, 10)
            pHlo = pHf_lo - rng.uniform(0, 0.5)
            pHhi = pHf_hi + rng.uniform(0, 0.5)
        else:
            gap = rng.uniform(5.0, 40.0)
            Tlo = Tf_hi + gap
            Thi = Tlo + rng.uniform(10, 40)
            pHlo = pHf_lo - rng.uniform(0, 0.5)
            pHhi = pHf_hi + rng.uniform(0, 0.5)
        Topt_f = rng.uniform(Tlo, Thi)
        pHopt_f = rng.uniform(pHlo, pHhi)
        filler = dict(
            s=sf, q=qf, f=ff,
            Tlo=Tlo, Thi=Thi, pHlo=pHlo, pHhi=pHhi,
            Topt=Topt_f, pHopt=pHopt_f,
            R=round(rng.uniform(0.3, 1.5), 4),
            r=round(rng.uniform(0.1, 0.7), 4),
        )
        templates.append(filler)

    # defensive check: decoy (index 1) must strictly be the SDI argmax. If a
    # filler randomly beat it, nudge the decoy's match terms upward.
    def sdi_of(t):
        size_m = max(0.0, 1.0 - abs(t["s"] - D_target) / D_target)
        charge_m = max(0.0, 1.0 - abs(t["q"] - q_target) / Q_NORM)
        shape_m = max(0.0, 1.0 - abs(t["f"] - fid) / F_NORM)
        return w1 * size_m + w2 * charge_m + w3 * shape_m

    guard = 0
    while guard < 50 and any(sdi_of(templates[j]) >= sdi_of(templates[1]) for j in range(len(templates)) if j != 1):
        templates[1]["s"] = D_target
        templates[1]["q"] = q_target
        templates[1]["f"] = fid
        guard += 1

    lines = []
    lines.append("%d %d" % (K, c))
    lines.append("%s %s" % (fmt(D_target), fmt(q_target)))
    lines.append("%s %s %s %s" % (fmt(Tf_lo), fmt(Tf_hi), fmt(pHf_lo), fmt(pHf_hi)))
    lines.append("%s %s %s" % (fmt(w1), fmt(w2), fmt(w3)))
    for t in templates:
        lines.append(" ".join(fmt(v) for v in [
            t["s"], t["q"], t["f"], t["Tlo"], t["Thi"], t["pHlo"], t["pHhi"],
            t["Topt"], t["pHopt"], t["R"], t["r"],
        ]))
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    sys.stdout.write(gen(test_id))


if __name__ == "__main__":
    main()
