import sys, random

# ---------------------------------------------------------------------------
# fsx_B_1275 -- procurement-award-split
#
# Instance layout (all deterministic given testId):
#   T N G
#   V P
#   D_1 ... D_T
#   N supplier lines: group qualified lead qualcost ntiers (th price)*ntiers
#   E
#   E lines: period group   (disruption events: every supplier in <group> delivers
#                            zero units during <period>)
# ---------------------------------------------------------------------------

# Hand-authored ladder: testId 1-3 are disruption-free warm-ups (single-sourcing
# the cheapest supplier is genuinely fine there); testId 4-10 plant correlated
# disruption events that hit the cheapest supplier's correlation group with enough
# lead-time runway that a supplier qualified *in advance* would have dodged them,
# but not enough for a *reactive* qualification started only once the disruption is
# already underway.
SPEC = {
    1:  dict(T=5,  N=3, G=2, groups=[0,1,1],           leads=[0,3,3],             quals=[0,220,220],           disruptions=[]),
    2:  dict(T=6,  N=3, G=2, groups=[0,1,1],           leads=[0,3,2],             quals=[0,220,200],           disruptions=[]),
    3:  dict(T=7,  N=4, G=2, groups=[0,1,1,0],         leads=[0,2,3,2],           quals=[0,200,220,180],       disruptions=[]),
    4:  dict(T=8,  N=4, G=2, groups=[0,1,1,0],         leads=[0,2,3,2],           quals=[0,200,240,180],       disruptions=[(5,0),(6,0),(7,0)]),
    5:  dict(T=10, N=5, G=3, groups=[0,1,2,1,2],       leads=[0,3,2,4,3],         quals=[0,220,200,240,220],   disruptions=[(6,0),(7,0),(8,0),(9,1)]),
    6:  dict(T=10, N=5, G=3, groups=[0,1,2,1,2],       leads=[0,3,2,4,3],         quals=[0,220,200,240,220],   disruptions=[(4,0),(8,0),(9,0)]),
    7:  dict(T=12, N=6, G=3, groups=[0,1,2,1,2,2],     leads=[0,3,2,4,3,3],       quals=[0,220,200,240,220,210],disruptions=[(5,0),(6,0),(7,0),(9,1)]),
    8:  dict(T=14, N=6, G=4, groups=[0,1,2,3,1,2],     leads=[0,3,3,2,4,3],       quals=[0,220,220,200,240,220],disruptions=[(6,0),(7,0),(8,0),(9,0),(11,2)]),
    9:  dict(T=16, N=7, G=4, groups=[0,0,1,2,3,1,2],   leads=[0,2,4,3,3,4,4],     quals=[0,150,240,220,220,240,240],disruptions=[(7,0),(8,0),(9,0),(10,0),(12,1)]),
    10: dict(T=20, N=8, G=4, groups=[0,0,1,2,3,1,2,3], leads=[0,2,4,3,3,4,4,3],   quals=[0,150,240,220,240,240,240,220],disruptions=[(8,0),(9,0),(10,0),(11,0),(14,1),(18,2)]),
}


def build(tid):
    sp = SPEC[tid]
    T, N, G = sp["T"], sp["N"], sp["G"]
    rnd = random.Random(90210 + 733 * tid)

    V = 20
    P = 4

    # demand: mild per-period variation around a scale that grows gently with tid
    base_d = 34 + 2 * min(tid, 8)
    D = [base_d + rnd.randint(-6, 6) for _ in range(T)]

    # supplier 0 is always the primary / initially-qualified / globally cheapest.
    primary_base = 14 + rnd.randint(0, 2)

    prices = [primary_base]
    for i in range(1, N):
        if sp["groups"][i] == 0 and i == 1 and tid >= 9:
            # decoy: "2nd cheapest" supplier that shares the primary's correlation
            # group, so diversifying into it buys nothing when disruption hits.
            prices.append(primary_base + rnd.randint(1, 2))
        else:
            prices.append(primary_base + rnd.randint(4, 10))

    qualified = [1] + [0] * (N - 1)

    avgD = sum(D) / len(D)
    th1 = max(1, round(0.30 * avgD))
    th2 = max(th1 + 1, round(0.60 * avgD))

    suppliers = []
    for i in range(N):
        base = prices[i]
        p1 = max(2, base - rnd.randint(3, 5))
        p2 = max(1, p1 - rnd.randint(4, 7))
        tiers = [(0, base), (th1, p1), (th2, p2)]
        suppliers.append(dict(
            group=sp["groups"][i], qualified=qualified[i],
            lead=sp["leads"][i], qualcost=sp["quals"][i], tiers=tiers,
        ))

    disruptions = sp["disruptions"]

    lines = []
    lines.append(f"{T} {N} {G}")
    lines.append(f"{V} {P}")
    lines.append(" ".join(str(d) for d in D))
    for s in suppliers:
        toks = [s["group"], s["qualified"], s["lead"], s["qualcost"], len(s["tiers"])]
        for th, pr in s["tiers"]:
            toks += [th, pr]
        lines.append(" ".join(str(x) for x in toks))
    lines.append(str(len(disruptions)))
    for per, grp in disruptions:
        lines.append(f"{per} {grp}")
    return "\n".join(lines) + "\n"


def main():
    tid = int(sys.argv[1])
    if tid not in SPEC:
        tid = ((tid - 1) % 10) + 1
    sys.stdout.write(build(tid))


if __name__ == "__main__":
    main()
