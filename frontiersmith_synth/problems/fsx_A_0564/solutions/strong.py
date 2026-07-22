# TIER: strong
# Insight: co-optimise the roster WITH the frozen greedy repairer. Slack is only worth
# what the fixed low-index scan can reach, so we place scarce-skill reserves at exactly the
# (staff,day) states the published repair rule actually visits. We evaluate the TRUE worst-
# case objective by running the same repair simulation, and do local search over rosters.
import sys, random

def parse():
    toks = sys.stdin.read().split()
    it = iter(toks); nxt = lambda: next(it)
    S = int(nxt()); D = int(nxt()); C = int(nxt())
    maxshift = [int(nxt()) for _ in range(S)]
    base = [int(nxt()) for _ in range(S)]
    skills = []
    for _ in range(S):
        cnt = int(nxt()); skills.append(set(int(nxt()) for _ in range(cnt)))
    days = []
    for _ in range(D):
        T = int(nxt()); days.append([(int(nxt()), int(nxt())) for _ in range(T)])
    K = int(nxt()); scen = []
    for _ in range(K):
        A = int(nxt()); scen.append(set((int(nxt()), int(nxt())) for _ in range(A)))
    U = int(nxt()); OT = int(nxt())
    return dict(S=S, D=D, maxshift=maxshift, base=base, skills=skills,
                days=days, scen=scen, U=U, OT=OT)

def simulate(assign, absent, g):
    S, D, days, skills = g["S"], g["D"], g["days"], g["skills"]
    maxshift, base, U, OT = g["maxshift"], g["base"], g["U"], g["OT"]
    busy = [[False] * D for _ in range(S)]
    used = [0] * S
    holes = []
    for d in range(D):
        row = assign[d]
        for t, (k, h) in enumerate(days[d]):
            a = row[t]
            if (a, d) in absent:
                holes.append((d, k, h))
            else:
                busy[a][d] = True; used[a] += 1
    unc = ot = 0
    for (d, k, h) in holes:
        filled = False
        for j in range(S):
            if k in skills[j] and (j, d) not in absent and not busy[j][d] and used[j] < maxshift[j]:
                busy[j][d] = True; used[j] += 1; filled = True
                if used[j] > base[j]:
                    ot += h
                break
        if not filled:
            unc += h
    return U * unc + OT * ot

def worst(assign, g):
    return max(simulate(assign, ab, g) for ab in g["scen"])

def qualified(g, k):
    return [j for j in range(g["S"]) if k in g["skills"][j]]

def build(order_key, g):
    # greedy construction, assigning each slot to the qualified staff chosen by order_key,
    # skipping staff already busy that day / over maxshift
    S, D, days, skills, maxshift = g["S"], g["D"], g["days"], g["skills"], g["maxshift"]
    used = [0] * S
    assign = []
    for d in range(D):
        busy = set()
        row = []
        for (k, h) in days[d]:
            cands = [j for j in range(S) if k in skills[j] and j not in busy and used[j] < maxshift[j]]
            j = min(cands, key=lambda x: order_key(x, d)) if cands else 0
            busy.add(j); used[j] += 1
            row.append(j)
        assign.append(row)
    return assign

def local_search(assign, g, rng, passes=6):
    S, D, days, skills, maxshift = g["S"], g["D"], g["days"], g["skills"], g["maxshift"]
    cur = worst(assign, g)
    for _ in range(passes):
        improved = False
        # recompute per-day busy + used
        for d in range(D):
            for t, (k, h) in enumerate(days[d]):
                a = assign[d][t]
                busy_today = set(assign[d])
                used = [0] * S
                for dd in range(D):
                    for x in assign[dd]:
                        used[x] += 1
                for b in range(S):
                    if b == a or k not in skills[b]:
                        continue
                    if b in busy_today:
                        continue
                    if used[b] - (1 if b == a else 0) >= maxshift[b]:
                        continue
                    old = assign[d][t]
                    assign[d][t] = b
                    nc = worst(assign, g)
                    if nc < cur:
                        cur = nc; improved = True
                        break
                    else:
                        assign[d][t] = old
        if not improved:
            break
    return assign, cur

def main():
    g = parse()
    S, D = g["S"], g["D"]

    seeds = []
    # seed 1: reserve the scarce specialists -> assign common slots to HIGH-index staff,
    # scarce slots to the highest-index qualified scarce holder (keep low-index scarce free)
    seeds.append(build(lambda j, d: -j, g))
    # seed 2: canonical low-index
    seeds.append(build(lambda j, d: j, g))
    # seed 3: rotate scarce coverage by day
    seeds.append(build(lambda j, d: (j + d) % S, g))

    best = None; bestc = None
    rng = random.Random(12345)
    for sd in seeds:
        a, c = local_search([row[:] for row in sd], g, rng)
        if bestc is None or c < bestc:
            bestc = c; best = a

    out = []
    for d in range(D):
        for x in best[d]:
            out.append(str(x))
    sys.stdout.write(" ".join(out) + "\n")

main()
