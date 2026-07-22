import sys, math


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def read_instance(path):
    try:
        toks = open(path).read().split()
        it = iter(toks)
        U = int(next(it)); T = int(next(it))
        PEN = float(next(it)); BUDGET = float(next(it))
        C = []; R = []; r = []; D = []
        for _ in range(U):
            C.append(float(next(it))); R.append(float(next(it)))
            r.append(float(next(it))); D.append(int(next(it)))
        demand = [float(next(it)) for _ in range(T)]
    except Exception:
        fail("bad input")
    if U < 2 or T < 3:
        fail("bad input dims")
    return U, T, PEN, BUDGET, C, R, r, D, demand


def capacity_and_rate(t, s_i, C_i, R_i, r_i, D_i):
    """t is 1-indexed. s_i=0 means the unit never retrofits."""
    if s_i == 0:
        return C_i, R_i
    off_start, off_end = s_i, s_i + D_i - 1
    if off_start <= t <= off_end:
        return 0.0, 0.0
    if t < off_start:
        return C_i, R_i
    return C_i, r_i


def evaluate(schedule, dispatch, U, T, PEN, BUDGET, C, R, r, D):
    """Assumes schedule/dispatch already validated for shape/range/coverage."""
    total_served = 0.0
    cum_em = 0.0
    for t in range(1, T + 1):
        row = dispatch[t - 1]
        for i in range(U):
            _, rate = capacity_and_rate(t, schedule[i], C[i], R[i], r[i], D[i])
            d = row[i]
            total_served += d
            cum_em += d * rate
    overage = max(0.0, cum_em - BUDGET)
    raw = total_served - PEN * overage
    return max(0.0, raw), total_served, cum_em, overage


def naive_baseline(U, T, C, R, r, D, demand):
    """Never retrofit; fixed INDEX-order fill to exactly meet demand each
    step (no bonus dispatch). This is the checker's own internal baseline."""
    schedule = [0] * U
    dispatch = []
    for t in range(1, T + 1):
        need = demand[t - 1]
        row = [0.0] * U
        for i in range(U):
            if need <= 1e-9:
                break
            cap, _ = capacity_and_rate(t, 0, C[i], R[i], r[i], D[i])
            use = min(cap, need)
            row[i] = use
            need -= use
        dispatch.append(row)
    return schedule, dispatch


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    U, T, PEN, BUDGET, C, R, r, D, demand = read_instance(inf)

    # ---- internal baseline B: naive never-retrofit, index-order exact fill ----
    base_sched, base_disp = naive_baseline(U, T, C, R, r, D, demand)
    B_raw, _, _, _ = evaluate(base_sched, base_disp, U, T, PEN, BUDGET, C, R, r, D)
    B = max(B_raw, 1e-6)

    # ---- parse & strictly validate participant output ----
    try:
        toks = open(outf).read().split()
    except Exception:
        fail("cannot read output")
    if len(toks) < U:
        fail("output too short (missing retrofit schedule)")
    it = iter(toks)

    schedule = []
    for k in range(U):
        tok = next(it)
        try:
            v = int(tok)
        except ValueError:
            fail("non-integer retrofit start at unit %d" % (k + 1))
        schedule.append(v)
    for i in range(U):
        s = schedule[i]
        if s != 0 and not (1 <= s <= T - D[i] + 1):
            fail("retrofit start %d out of range for unit %d (dur %d, T %d)" % (s, i + 1, D[i], T))

    dispatch = []
    for t in range(T):
        row = []
        for i in range(U):
            try:
                tok = next(it)
            except StopIteration:
                fail("missing dispatch tokens (need %d rows of %d)" % (T, U))
            try:
                v = float(tok)
            except ValueError:
                fail("non-numeric dispatch token at t=%d unit=%d" % (t + 1, i + 1))
            if not math.isfinite(v):
                fail("non-finite dispatch token at t=%d unit=%d" % (t + 1, i + 1))
            if v < -1e-6:
                fail("negative dispatch at t=%d unit=%d" % (t + 1, i + 1))
            row.append(max(0.0, v))
        dispatch.append(row)
    extra = list(it)
    if extra:
        fail("extra trailing tokens in output (%d)" % len(extra))

    # ---- capacity bound check ----
    for t in range(1, T + 1):
        row = dispatch[t - 1]
        for i in range(U):
            cap, _ = capacity_and_rate(t, schedule[i], C[i], R[i], r[i], D[i])
            if row[i] > cap + 1e-6:
                fail("dispatch %.4f exceeds capacity %.4f at t=%d unit=%d" % (row[i], cap, t, i + 1))

    # ---- coverage check: demand must be met every step ----
    for t in range(1, T + 1):
        tot = sum(dispatch[t - 1])
        if tot < demand[t - 1] - 1e-6:
            fail("coverage violated at step %d (%.4f < %.4f)" % (t, tot, demand[t - 1]))

    F, total_served, cum_em, overage = evaluate(schedule, dispatch, U, T, PEN, BUDGET, C, R, r, D)

    sc = min(1000.0, 100.0 * F / B)
    print("U=%d T=%d F=%.4f B=%.4f served=%.2f em=%.2f overage=%.2f Ratio: %.6f"
          % (U, T, F, B, total_served, cum_em, overage, sc / 1000.0))


if __name__ == "__main__":
    main()
