import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

# ---------------------------------------------------------------- parsing
def parse_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    nxt = lambda: next(it)
    S = int(nxt()); D = int(nxt()); C = int(nxt())
    maxshift = [int(nxt()) for _ in range(S)]
    base = [int(nxt()) for _ in range(S)]
    skills = []
    for _ in range(S):
        cnt = int(nxt())
        skills.append(set(int(nxt()) for _ in range(cnt)))
    days = []
    for _ in range(D):
        T = int(nxt())
        days.append([(int(nxt()), int(nxt())) for _ in range(T)])
    K = int(nxt())
    scen = []
    for _ in range(K):
        A = int(nxt())
        scen.append(set((int(nxt()), int(nxt())) for _ in range(A)))
    U = int(nxt()); OT = int(nxt())
    return dict(S=S, D=D, C=C, maxshift=maxshift, base=base, skills=skills,
               days=days, K=K, scen=scen, U=U, OT=OT)

# ---------------------------------------------------------------- fixed repair simulation
def simulate(assign, absent, g):
    S, D = g["S"], g["D"]
    days, skills = g["days"], g["skills"]
    maxshift, base, U, OT = g["maxshift"], g["base"], g["U"], g["OT"]
    busy = [[False] * D for _ in range(S)]
    used = [0] * S
    holes = []                      # (d, k, h) in day-major slot order
    for d in range(D):
        row = assign[d]
        for t, (k, h) in enumerate(days[d]):
            a = row[t]
            if (a, d) in absent:
                holes.append((d, k, h))
            else:
                busy[a][d] = True
                used[a] += 1
    unc = 0
    ot = 0
    for (d, k, h) in holes:         # already day-major slot order
        filled = False
        for j in range(S):
            if k in skills[j] and (j, d) not in absent and not busy[j][d] and used[j] < maxshift[j]:
                busy[j][d] = True
                used[j] += 1
                filled = True
                if used[j] > base[j]:
                    ot += h
                break
        if not filled:
            unc += h
    return U * unc + OT * ot

def worst_cost(assign, g):
    return max(simulate(assign, ab, g) for ab in g["scen"])

# ---------------------------------------------------------------- canonical baseline roster
def canonical(g):
    S, D, days, skills, maxshift = g["S"], g["D"], g["days"], g["skills"], g["maxshift"]
    used = [0] * S
    assign = []
    for d in range(D):
        row = []
        busy_today = set()
        for (k, h) in days[d]:
            pick = -1
            for j in range(S):
                if k in skills[j] and j not in busy_today and used[j] < maxshift[j]:
                    pick = j
                    break
            if pick < 0:            # construction guarantees this never happens
                pick = 0
            row.append(pick)
            busy_today.add(pick)
            used[pick] += 1
        assign.append(row)
    return assign

# ---------------------------------------------------------------- main
def main():
    g = None
    try:
        g = parse_instance(sys.argv[1])
    except Exception:
        fail("bad input")

    S, D, days, skills, maxshift = g["S"], g["D"], g["days"], g["skills"], g["maxshift"]
    total_slots = sum(len(days[d]) for d in range(D))

    # ---- parse participant roster: exactly total_slots integers, day-major slot order ----
    try:
        otoks = open(sys.argv[2]).read().split()
        vals = [int(x) for x in otoks]
    except Exception:
        fail("output not integer tokens")
    if len(vals) != total_slots:
        fail("expected %d assignments, got %d" % (total_slots, len(vals)))

    assign = []
    used = [0] * S
    p = 0
    for d in range(D):
        row = []
        busy_today = set()
        for (k, h) in days[d]:
            a = vals[p]; p += 1
            if a < 0 or a >= S:
                fail("staff %d out of range" % a)
            if k not in skills[a]:
                fail("staff %d lacks skill %d" % (a, k))
            if a in busy_today:
                fail("staff %d double-booked on day %d" % (a, d))
            busy_today.add(a)
            used[a] += 1
            row.append(a)
        assign.append(row)
    for a in range(S):
        if used[a] > maxshift[a]:
            fail("staff %d exceeds maxshift" % a)

    F = worst_cost(assign, g)
    B = worst_cost(canonical(g), g)
    B = max(1, B)

    if F <= 0:
        sc = 1000.0
    else:
        sc = min(1000.0, 100.0 * B / F)
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
