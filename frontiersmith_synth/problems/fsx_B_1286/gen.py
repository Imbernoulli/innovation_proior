#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE supplier-ESG-audit instance to stdout.

Instance = a 3-tier supplier forest (tier1 = direct suppliers of the focal
firm, tier2 = their suppliers, tier3 = tier2's suppliers). Tier1 nodes are
visible from the start; a tier2/tier3 node only becomes visible once its
parent has been MAPPED. Determinism: all randomness is seeded from testId
only.
"""
import random
import sys

# ---- tunable constants (mirrored in verify.py / solutions) ----
RISK1 = (30, 50)
RISK2 = (10, 20)
RISK3 = (15, 35)
MAPCOST1 = (4, 8)
MAPCOST2 = (2, 5)
AUDITCOST1 = (10, 18)
AUDITCOST2 = (5, 10)
AUDITCOST3 = (3, 7)
PROP = 0.6      # upstream risk-propagation decay factor per tier hop
MAPMIT = 0.2    # fractional risk mitigation from mapping (visibility) alone
AUDITMIT = 0.85  # fractional risk mitigation from a full audit
BUDGET_K = 2.5  # budget = BUDGET_K * sum(tier-1 audit costs)

# (T1, tier2-per-tier1, tier3-per-tier2) ladder, small -> large/adversarial.
PARAMS = {
    1: (2, 1, 1),
    2: (2, 2, 1),
    3: (3, 2, 2),
    4: (3, 2, 2),
    5: (3, 3, 2),
    6: (4, 2, 3),
    7: (4, 3, 3),
    8: (4, 3, 4),
    9: (5, 3, 4),
    10: (5, 3, 5),
}


def build(test_id):
    rnd = random.Random(test_id * 1000003 + 17)
    t1n, m2, m3 = PARAMS[test_id]

    rows = []  # (id, tier, parent, risk, mapcost, auditcost)
    nid = 1
    t1ids = []
    for _ in range(t1n):
        risk = rnd.randint(*RISK1)
        mc = rnd.randint(*MAPCOST1)
        ac = rnd.randint(*AUDITCOST1)
        rows.append([nid, 1, 0, risk, mc, ac])
        t1ids.append(nid)
        nid += 1

    t2ids = []
    for p in t1ids:
        for _ in range(m2):
            risk = rnd.randint(*RISK2)
            mc = rnd.randint(*MAPCOST2)
            ac = rnd.randint(*AUDITCOST2)
            rows.append([nid, 2, p, risk, mc, ac])
            t2ids.append(nid)
            nid += 1

    for p in t2ids:
        for _ in range(m3):
            risk = rnd.randint(*RISK3)
            ac = rnd.randint(*AUDITCOST3)
            rows.append([nid, 3, p, risk, 0, ac])  # tier3 = leaf, no map cost
            nid += 1

    n = nid - 1
    budget = round(BUDGET_K * sum(r[5] for r in rows if r[1] == 1))
    return n, t1n, budget, rows


def main():
    test_id = int(sys.argv[1])
    n, t1n, budget, rows = build(test_id)
    out = [f"{n} {t1n} {budget}", f"{PROP:.4f} {MAPMIT:.4f} {AUDITMIT:.4f}"]
    for r in rows:
        out.append(" ".join(str(x) for x in r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
