import sys

CLASSES = "HML"


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        outraw = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    idx = [0]

    def nxt():
        v = inp[idx[0]]
        idx[0] += 1
        return v

    # ---- parse instance ----
    try:
        N = int(nxt())
        D = int(nxt())
        FIXSEP = int(nxt())
        S = [[0, 0, 0] for _ in range(3)]
        for a in range(3):
            for b in range(3):
                S[a][b] = int(nxt())
        fw = [int(nxt()) for _ in range(3)]
        r = []
        cls = []
        for i in range(N):
            r.append(int(nxt()))
            c = nxt()
            if c not in CLASSES:
                fail("bad class token in input")
            cls.append(CLASSES.index(c))
    except Exception:
        fail("malformed input file (harness bug, not participant's fault)")

    # ---- parse participant artifact: N triples (runway, fix_time, landing_time) ----
    try:
        vals = [float(x) for x in outraw]
    except Exception:
        fail("output is not purely numeric")
    for v in vals:
        if v != v or v in (float("inf"), float("-inf")):
            fail("non-finite value in output")
    if len(vals) != 3 * N:
        fail("expected %d tokens (3 per aircraft, in input order), got %d" % (3 * N, len(vals)))

    triples = []
    for i in range(N):
        rw_f, ft_f, lt_f = vals[3 * i], vals[3 * i + 1], vals[3 * i + 2]
        if (abs(rw_f - round(rw_f)) > 1e-6 or abs(ft_f - round(ft_f)) > 1e-6
                or abs(lt_f - round(lt_f)) > 1e-6):
            fail("non-integer field at aircraft %d" % (i + 1))
        rw, ft, lt = int(round(rw_f)), int(round(ft_f)), int(round(lt_f))
        if rw not in (1, 2):
            fail("aircraft %d: runway %d not in {1,2}" % (i + 1, rw))
        if ft < -1 or lt < -1 or ft > 10 ** 12 or lt > 10 ** 12:
            fail("aircraft %d: time field out of sane range" % (i + 1))
        triples.append((rw, ft, lt))

    # ---- feasibility: readiness + transit ----
    for i in range(N):
        rw, ft, lt = triples[i]
        if ft < r[i]:
            fail("aircraft %d crosses the fix at %d before its ready time %d" % (i + 1, ft, r[i]))
        if lt < ft + D:
            fail("aircraft %d lands at %d, less than fix_time+D=%d" % (i + 1, lt, ft + D))

    # ---- feasibility: shared-fix separation (global, order-independent of runway) ----
    order = sorted(range(N), key=lambda i: (triples[i][1], i))
    for k in range(1, N):
        p, c = order[k - 1], order[k]
        gap = triples[c][1] - triples[p][1]
        if gap < FIXSEP:
            fail("shared-fix separation violated between aircraft %d and %d (gap %d < %d)"
                 % (p + 1, c + 1, gap, FIXSEP))

    # ---- feasibility: per-runway sequence-dependent separation ----
    for rw_id in (1, 2):
        members = [i for i in range(N) if triples[i][0] == rw_id]
        members.sort(key=lambda i: (triples[i][2], i))
        for k in range(1, len(members)):
            p, c = members[k - 1], members[k]
            need = S[cls[p]][cls[c]]
            gap = triples[c][2] - triples[p][2]
            if gap < need:
                fail("runway %d separation violated: aircraft %d (%s) -> aircraft %d (%s), gap %d < %d"
                     % (rw_id, p + 1, CLASSES[cls[p]], c + 1, CLASSES[cls[c]], gap, need))

    # ---- objective: total fuel burned while holding (ready -> landing), class-weighted ----
    F = sum(fw[cls[i]] * (triples[i][2] - r[i]) for i in range(N))
    if F < 0:
        fail("negative total cost (impossible)")

    # ---- internal baseline B: two-runway ROUND-ROBIN FCFS (feasible, deliberately
    # unoptimized -- alternates runways blindly in ready-time order, never reorders
    # by class and never picks the "better" runway) ----
    order0 = sorted(range(N), key=lambda i: (r[i], i))
    fix_prev = None
    fixt = {}
    for i in order0:
        ft = r[i] if fix_prev is None else max(r[i], fix_prev + FIXSEP)
        fixt[i] = ft
        fix_prev = ft
    land_prev = [None, None]
    prevcls = [None, None]
    B = 0
    for k, i in enumerate(order0):
        rw = k % 2
        lt = fixt[i] + D
        if land_prev[rw] is not None:
            lt = max(lt, land_prev[rw] + S[prevcls[rw]][cls[i]])
        B += fw[cls[i]] * (lt - r[i])
        land_prev[rw] = lt
        prevcls[rw] = cls[i]
    B = max(1, B)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
