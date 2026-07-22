#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic scorer for shared-grade-allocation.

Feasibility (checked strictly against EVERY chain):
  PRIMARY : tol(g[p1]) + tol(g[p2]) + tol(g[h])  <= spec_primary
  BACKUP  : tol(g[p1]) + tol(g[p2])              <= spec_backup   (worst-case fallback:
             must hold even when the shared hub h is unavailable / fully degraded)

Objective (minimize): F = sum over all m features of cost(g[f])

Internal baseline B (independent of the submission): a "scaled fair-share" construction --
for every check, assume EVERY member of that check must, on its own, carry a 1/(3x the
member count) share of the check's budget (a naive per-requirement sizing rule that never
recognizes that a feature shared by many chains is worth extra investment). This is always
feasible (member_tol <= spec/(count*SCALE) <= spec/count for every check, with SCALE>=1,
so the true per-check sum is <= count*(spec/count) = spec... with extra slack from SCALE).
"""
import sys

TOL = [64, 32, 16, 8, 4, 2, 1]
COST = [0, 1, 3, 7, 15, 31, 63]
GMAX = 6
SCALE = 3  # integer scale factor baked into the baseline (see solutions/trivial.py)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    idx = 0
    m = int(toks[idx]); idx += 1
    C = int(toks[idx]); idx += 1
    chains = []
    for _ in range(C):
        p1 = int(toks[idx]); idx += 1
        p2 = int(toks[idx]); idx += 1
        h = int(toks[idx]); idx += 1
        sp = int(toks[idx]); idx += 1
        sb = int(toks[idx]); idx += 1
        chains.append((p1, p2, h, sp, sb))
    return m, C, chains


def baseline_cost(m, chains):
    """SCALE-inflated fair-share baseline: grade(f) = min g s.t. for every check touching
    f, tol(g) * count * SCALE <= spec.  Always feasible; used both as B here and as the
    'trivial' reference construction (solutions/trivial.py implements the identical rule)."""
    constraints = [[] for _ in range(m)]
    for (p1, p2, h, sp, sb) in chains:
        constraints[p1].append((sp, 3))
        constraints[p1].append((sb, 2))
        constraints[p2].append((sp, 3))
        constraints[p2].append((sb, 2))
        constraints[h].append((sp, 3))
    total = 0
    for f in range(m):
        g = 0
        for spec, cnt in constraints[f]:
            while g < GMAX and TOL[g] * cnt * SCALE > spec:
                g += 1
        total += COST[g]
    return total


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    m, C, chains = read_instance(in_path)

    try:
        with open(out_path) as f:
            raw = f.read().split()
    except Exception:
        print("bad output file. Ratio: 0.0")
        return 0

    if len(raw) != m:
        print(f"expected exactly {m} grade tokens, got {len(raw)}. Ratio: 0.0")
        return 0

    grade = []
    for tok in raw:
        try:
            g = int(tok)
        except (ValueError, OverflowError):
            print(f"non-integer token {tok!r}. Ratio: 0.0")
            return 0
        if g < 0 or g > GMAX:
            print(f"grade {g} out of range [0,{GMAX}]. Ratio: 0.0")
            return 0
        grade.append(g)

    for (p1, p2, h, sp, sb) in chains:
        primary_sum = TOL[grade[p1]] + TOL[grade[p2]] + TOL[grade[h]]
        if primary_sum > sp:
            print(f"PRIMARY violated: {primary_sum} > {sp}. Ratio: 0.0")
            return 0
        backup_sum = TOL[grade[p1]] + TOL[grade[p2]]
        if backup_sum > sb:
            print(f"BACKUP violated: {backup_sum} > {sb}. Ratio: 0.0")
            return 0

    F = sum(COST[g] for g in grade)
    B = baseline_cost(m, chains)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
