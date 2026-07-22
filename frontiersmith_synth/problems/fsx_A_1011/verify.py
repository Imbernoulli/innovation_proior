import sys, math

Tamb = 20.0
EPS = 1e-6


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def cool_time_to(T_from, T_to, k):
    if T_from <= T_to:
        return 0.0
    return math.log((T_from - Tamb) / (T_to - Tamb)) / k


def read_instance(path):
    try:
        toks = open(path).read().split()
        it = iter(toks)
        W = int(next(it))
        Tamb_in = float(next(it))
        H_lo = float(next(it)); H_hi = float(next(it))
        M_lo = float(next(it)); M_hi = float(next(it))
        L_lo = float(next(it)); L_hi = float(next(it))
        REHEAT_TEMP = float(next(it))
        REHEAT_DUR = float(next(it))
        OP_DUR = float(next(it))
        PENALTY = float(next(it))
        HORIZON = float(next(it))
        pieces = []
        for _ in range(W):
            T0 = float(next(it)); k = float(next(it))
            v1 = int(next(it)); v2 = int(next(it)); v3 = int(next(it))
            pieces.append(dict(T0=T0, k=k, v=[v1, v2, v3]))
    except StopIteration:
        raise ValueError("truncated instance")
    global Tamb
    Tamb = Tamb_in
    bands = [(H_lo, H_hi), (M_lo, M_hi), (L_lo, L_hi)]
    return W, bands, REHEAT_TEMP, REHEAT_DUR, OP_DUR, PENALTY, HORIZON, pieces


def baseline_value(pieces, bands, REHEAT_TEMP, PENALTY):
    """Checker's own internal baseline B: reheat before every single operation
    (always safe, never exploits cooling), applied to the SAME pieces."""
    B = 0.0
    for p in pieces:
        B += sum(p["v"]) * (PENALTY ** 3)
    return B


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    try:
        W, bands, REHEAT_TEMP, REHEAT_DUR, OP_DUR, PENALTY, HORIZON, pieces = read_instance(sys.argv[1])
    except Exception as e:
        fail("bad input: %s" % e)

    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    if not raw:
        fail("empty output")

    try:
        it = iter(raw)
        M = int(next(it))
    except Exception:
        fail("bad action count")
    if M < 0 or M > 200000:
        fail("action count out of range")

    actions = []
    try:
        for _ in range(M):
            typ = next(it)
            pid_s = next(it)
            st_s = next(it)
            pid = int(pid_s)
            st = float(st_s)
            if not math.isfinite(st):
                fail("non-finite start time")
            if typ not in ("OP", "RH"):
                fail("unknown action type %r" % typ)
            if pid < 1 or pid > W:
                fail("piece id out of range: %r" % pid)
            actions.append((typ, pid, st))
    except (StopIteration, ValueError):
        fail("truncated / malformed action list")
    # reject trailing garbage
    try:
        next(it)
        fail("trailing tokens after declared action count")
    except StopIteration:
        pass

    # per-piece simulation state
    idx = [0] * (W + 1)          # next pending op (0,1,2; 3 = done), 1-indexed
    t_r = [0.0] * (W + 1)
    T_r = [0.0] * (W + 1)
    reheats = [0] * (W + 1)
    for i, p in enumerate(pieces, start=1):
        t_r[i] = 0.0
        T_r[i] = p["T0"]

    cur_forge_time = 0.0
    for (typ, pid, st) in actions:
        if st < cur_forge_time - EPS:
            fail("action at t=%.6f overlaps previous action ending %.6f" % (st, cur_forge_time))
        if st < -EPS:
            fail("negative start time")
        k = pieces[pid - 1]["k"]
        if typ == "RH":
            end = st + REHEAT_DUR
            if end > HORIZON + EPS:
                fail("reheat exceeds horizon")
            t_r[pid] = end
            T_r[pid] = REHEAT_TEMP
            reheats[pid] += 1
            cur_forge_time = end
        else:  # OP
            j = idx[pid]
            if j >= 3:
                fail("operation on already-completed piece %d" % pid)
            lo, hi = bands[j]
            end = st + OP_DUR
            if end > HORIZON + EPS:
                fail("operation exceeds horizon")
            cur_temp = Tamb + (T_r[pid] - Tamb) * math.exp(-k * (st - t_r[pid]))
            if not (lo - EPS <= cur_temp <= hi + EPS):
                fail("piece %d op %d at t=%.6f temp=%.4f outside window [%.2f,%.2f]"
                     % (pid, j + 1, st, cur_temp, lo, hi))
            idx[pid] = j + 1
            cur_forge_time = end

    F = 0.0
    completed = 0
    for i, p in enumerate(pieces, start=1):
        if idx[i] == 3:
            completed += 1
            F += sum(p["v"]) * (PENALTY ** reheats[i])

    B = baseline_value(pieces, bands, REHEAT_TEMP, PENALTY)
    B = max(1e-9, B)
    sc = min(1000.0, 100.0 * F / B)
    print("W=%d completed=%d F=%.6f B=%.6f Ratio: %.6f" % (W, completed, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
