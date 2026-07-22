# TIER: trivial
# Reproduces the checker's own internal baseline: reheat immediately before EVERY
# single operation (never trusts natural cooling at all), processing pieces one
# at a time. Always feasible, but pays the reheat penalty three times per piece,
# so it lands at Ratio ~= 0.1 by construction.
import sys, math

Tamb = 20.0


def cool_time_to(T_from, T_to, k):
    if T_from <= T_to:
        return 0.0
    return math.log((T_from - Tamb) / (T_to - Tamb)) / k


def main():
    global Tamb
    toks = sys.stdin.read().split()
    it = iter(toks)
    W = int(next(it))
    Tamb = float(next(it))
    H_lo = float(next(it)); H_hi = float(next(it))
    M_lo = float(next(it)); M_hi = float(next(it))
    L_lo = float(next(it)); L_hi = float(next(it))
    REHEAT_TEMP = float(next(it))
    REHEAT_DUR = float(next(it))
    OP_DUR = float(next(it))
    PENALTY = float(next(it))
    HORIZON = float(next(it))
    bands = [(H_lo, H_hi), (M_lo, M_hi), (L_lo, L_hi)]
    pieces = []
    for _ in range(W):
        T0 = float(next(it)); k = float(next(it))
        v1 = int(next(it)); v2 = int(next(it)); v3 = int(next(it))
        pieces.append((T0, k, [v1, v2, v3]))

    t = 0.0
    actions = []
    for pid, (T0, k, vs) in enumerate(pieces, start=1):
        for (lo, hi) in bands:
            actions.append(("RH", pid, t))
            t += REHEAT_DUR
            target = (lo + hi) / 2.0
            t += cool_time_to(REHEAT_TEMP, target, k)
            actions.append(("OP", pid, t))
            t += OP_DUR

    out = [str(len(actions))]
    for (typ, pid, st) in actions:
        out.append("%s %d %.9f" % (typ, pid, st))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
