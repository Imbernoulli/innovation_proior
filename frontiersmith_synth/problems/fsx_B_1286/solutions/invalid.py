# TIER: invalid
"""Deliberately infeasible: audit a tier-3 supplier directly, without ever
mapping its tier-1/tier-2 ancestors -- it is not visible yet, so the checker
must reject this with Ratio: 0.0."""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    idx += 1  # t1n
    idx += 1  # budget
    idx += 3  # prop, mapmit, auditmit
    target = None
    for _ in range(n):
        nid = int(toks[idx]); idx += 1
        t = int(toks[idx]); idx += 1
        idx += 1  # parent
        idx += 1  # risk
        idx += 1  # mapcost
        idx += 1  # auditcost
        if t == 3 and target is None:
            target = nid
    if target is None:
        target = n
    sys.stdout.write(f"A {target}\n")


if __name__ == "__main__":
    main()
