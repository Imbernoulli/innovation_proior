import sys, math
from sensorfield import build_instance

EPS = 0.40              # RMSE smoothing floor (quality = 1/(RMSE+EPS))
W0 = 0.45               # base weight on reconstruction quality; (1-W0) scales with F1
MAX_CORR = 1.0e4        # sanity bound on a submitted correction magnitude


def fail(msg):
    print(msg)
    print("Ratio: 0.0")
    sys.exit(0)


def parse_in(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    test_id = int(next(it))
    n = int(next(it)); T = int(next(it)); F_max = int(next(it))
    m = int(next(it))
    edges = []
    for _ in range(m):
        u = int(next(it)); v = int(next(it))
        edges.append((u, v))
    R = [[0.0] * T for _ in range(n)]
    for i in range(n):
        for t in range(T):
            R[i][t] = float(next(it))
    return test_id, n, T, F_max, R


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    test_id, n, T, F_max, R_in = parse_in(inf)

    inst = build_instance(test_id)
    if inst["n"] != n or inst["T"] != T or inst["F_max"] != F_max:
        fail("Corrupt input (does not match generator for this testId)")
    Y = inst["Y"]
    R = inst["R"]
    # sanity: the input file's reading matrix must match the regenerated one (bit-tolerant)
    for i in range(n):
        for t in range(T):
            if abs(R_in[i][t] - R[i][t]) > 1e-3:
                fail("Corrupt input (reading mismatch)")
    fault_true = inst["fault"]
    S_true = set(fault_true.keys())

    try:
        with open(outf) as f:
            toks = f.read().split()
    except Exception:
        fail("Cannot read output")
    if not toks:
        fail("Empty output")
    it = iter(toks)

    def nxt():
        try:
            return next(it)
        except StopIteration:
            return None

    dtok = nxt()
    if dtok is None:
        fail("Missing declaration count")
    try:
        D = int(dtok)
    except ValueError:
        fail("Declaration count not an integer")
    if D < 0 or D > F_max:
        fail(f"Declared fault count {D} outside budget [0,{F_max}]")

    declared = {}
    order = []
    for _ in range(D):
        sid_t, a_t, b_t = nxt(), nxt(), nxt()
        if sid_t is None or a_t is None or b_t is None:
            fail("Truncated declaration line")
        try:
            sid = int(sid_t)
            a = float(a_t)
            b = float(b_t)
        except ValueError:
            fail("Non-numeric declaration fields")
        if not (math.isfinite(a) and math.isfinite(b)):
            fail("Non-finite correction (nan/inf rejected)")
        if sid < 0 or sid >= n:
            fail(f"Sensor id {sid} out of range")
        if sid in declared:
            fail(f"Duplicate sensor id {sid} declared")
        if abs(a) > MAX_CORR or abs(b) * max(1, T) > MAX_CORR:
            fail("Correction magnitude out of sane bounds")
        declared[sid] = (a, b)
        order.append(sid)
    extra = nxt()
    if extra is not None:
        fail("Trailing garbage after declared corrections")

    # ---- reconstruction --------------------------------------------
    def rmse(decl):
        se = 0.0
        cnt = 0
        for i in range(n):
            a_hat, b_hat = decl.get(i, (0.0, 0.0))
            for t in range(T):
                xh = R[i][t] - a_hat - b_hat * t
                d = xh - Y[i][t]
                se += d * d
                cnt += 1
        return math.sqrt(se / cnt)

    def f1(decl_ids):
        S_hat = set(decl_ids)
        tp = len(S_hat & S_true)
        prec = tp / len(S_hat) if S_hat else 0.0
        rec = tp / len(S_true) if S_true else (1.0 if not S_hat else 0.0)
        if prec + rec == 0.0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    def quality(rm, f1v):
        return (1.0 / (rm + EPS)) * (W0 + (1.0 - W0) * f1v)

    rmse_sub = rmse(declared)
    f1_sub = f1(declared.keys())
    Q_sub = quality(rmse_sub, f1_sub)

    rmse_base = rmse({})
    Q_base = quality(rmse_base, 0.0)

    sc = min(1000.0, 100.0 * Q_sub / max(1e-9, Q_base))
    ratio = sc / 1000.0
    ratio = max(0.0, min(1.0, ratio))

    print(f"test_id={test_id} n={n} T={T} F_max={F_max} D={D} "
          f"rmse_sub={rmse_sub:.4f} rmse_base={rmse_base:.4f} f1={f1_sub:.4f} "
          f"Q_sub={Q_sub:.4f} Q_base={Q_base:.4f}")
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
