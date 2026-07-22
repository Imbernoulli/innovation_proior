# TIER: greedy
"""
The "obvious recipe": at every decision point, chase whichever reaction is
locally most productive right now. Concretely: compute each group's pure
DIRECT yield-per-raw-material rate (dv/(dx*px)), pick the single best group,
and dump the ENTIRE raw-material budget into that group's direct chain.

This never builds any Y_g (which has zero value on its own), never touches
the catalyst Z, and never fires a single Combo reaction -- it is exactly
the kind of myopic, single-reaction-at-a-time optimization a solver writes
when it only tracks "what gives me target NOW" and ignores that a scarce
shared precursor (Z) plus a deliberately-idle co-reagent (Y) could be
reserved for a much better multi-reactant combination downstream.
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    m = int(data[idx]); idx += 1
    N_R = int(data[idx]); idx += 1
    N_Z = int(data[idx]); idx += 1
    groups = []
    for g in range(m):
        vals = [int(data[idx + k]) for k in range(7)]
        idx += 7
        groups.append(vals)

    best_g, best_rate = 0, -1.0
    for g, (px, py, dx, dv, cx, cy, cw) in enumerate(groups):
        denom = dx * px
        rate = dv / denom if denom > 0 else 0.0
        if rate > best_rate:
            best_rate = rate
            best_g = g

    px, py, dx, dv, cx, cy, cw = groups[best_g]
    denom = dx * px
    f = N_R // denom if denom > 0 else 0

    tokens = []
    tokens.extend(["%dX" % best_g] * (f * dx))
    tokens.extend(["%dD" % best_g] * f)

    out = []
    out.append(str(len(tokens)))
    out.append(" ".join(tokens))
    print("\n".join(out))


if __name__ == "__main__":
    main()
