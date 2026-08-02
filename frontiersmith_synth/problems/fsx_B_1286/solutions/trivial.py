# TIER: trivial
"""Audit whichever direct (tier-1) supplier is cheapest to audit, and stop.
No mapping, no consideration of risk or of tiers 2/3 at all -- the laziest
feasible move."""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    t1n = int(toks[idx]); idx += 1
    budget = int(toks[idx]); idx += 1
    idx += 3  # prop, mapmit, auditmit -- unused
    best_id, best_cost = None, None
    for _ in range(n):
        nid = int(toks[idx]); idx += 1
        t = int(toks[idx]); idx += 1
        idx += 1  # parent
        idx += 1  # risk
        idx += 1  # mapcost
        ac = int(toks[idx]); idx += 1
        if t == 1 and (best_cost is None or ac < best_cost):
            best_id, best_cost = nid, ac
    out = []
    if best_id is not None and best_cost is not None and best_cost <= budget:
        out.append(f"A {best_id}")
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
