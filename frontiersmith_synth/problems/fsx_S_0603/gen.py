#!/usr/bin/env python3
# Generator for fsx_S_0603 -- "A vein network that survives a cut".
# `python3 gen.py <testId>` prints ONE instance to stdout (testId 1..10).
#
# Each instance = an R x C grid mesh, one source S, K sink nodes, a reinforcement
# exponent range, and a MENU of demand scenarios the solver may weight:
#   scenario 0     : the aggregate load (every sink drawn simultaneously)
#   scenarios 1..K : each single sink drawn alone (the fluctuating regime)
#   two half-group scenarios.
# TRAP layouts spread the sinks far apart around an interior source, so the
# aggregate-load remodelling prunes a fragile trunk; CONTROL layouts cluster the
# sinks (aggregate remodelling is fine there).  Difficulty grows with testId.
import sys

GMIN, GMAX = 0.0, 0.85

# (R, C, source, sinks)  -- deterministic ladder, ~small -> large.
LAYOUTS = {
    1:  (5, 5, 12, [0, 4, 20, 24]),                       # spread corners (trap)
    2:  (5, 6, 15, [0, 1, 6, 7]),                         # clustered top-left (control)
    3:  (6, 6, 21, [5, 11, 29, 35, 0, 30]),               # far spread (trap)
    4:  (6, 6, 14, [0, 5, 30, 35, 2, 33]),                # corners+edges (trap)
    5:  (7, 7, 24, [0, 3, 6, 21, 27, 42, 45, 48]),        # ring / boundary (trap)
    6:  (6, 8, 27, [0, 1, 8, 9, 6, 7, 14, 15]),           # two clusters (control)
    7:  (7, 7, 24, [0, 6, 42, 48, 3, 45, 21, 27]),        # 8-way spread (trap)
    8:  (5, 9, 22, [0, 4, 8, 36, 40, 44, 18, 26]),        # wide spread (trap)
    9:  (7, 8, 27, [24, 25, 32, 33, 26, 34, 40, 41]),     # clustered mid-left (control)
    10: (8, 8, 27, [0, 7, 56, 63, 3, 60, 24, 39, 28, 35]),# large spread (trap)
}

def main():
    tid = int(sys.argv[1])
    if tid not in LAYOUTS:
        tid = ((tid - 1) % 10) + 1
    R, C, S, sinks = LAYOUTS[tid]
    N = R * C
    # sanity: valid, distinct, not the source
    seen = set()
    clean = []
    for t in sinks:
        if 0 <= t < N and t != S and t not in seen:
            seen.add(t); clean.append(t)
    sinks = clean
    K = len(sinks)

    # menu of demand scenarios
    menu = [list(sinks)]                      # 0: aggregate
    for t in sinks:                           # 1..K: single sinks (fluctuating)
        menu.append([t])
    half = K // 2
    menu.append(list(sinks[:half]))           # left half group
    menu.append(list(sinks[half:]))           # right half group
    M = len(menu)

    lines = []
    lines.append("%d %d" % (R, C))
    lines.append("%d" % S)
    lines.append("%d" % K)
    lines.append(" ".join(map(str, sinks)))
    lines.append("%.2f %.2f" % (GMIN, GMAX))
    lines.append("%d" % M)
    for grp in menu:
        lines.append(" ".join([str(len(grp))] + list(map(str, grp))))
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
