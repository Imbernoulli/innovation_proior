import sys, math
from _model import build


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    inp_path, out_path = sys.argv[1], sys.argv[2]
    try:
        toks = open(inp_path).read().split()
        it = iter(toks)
        n = int(next(it)); m = int(next(it)); K = int(next(it)); L = int(next(it)); seed = int(next(it))
        p = int(next(it))
        observed = []
        obs_val = {}
        for _ in range(p):
            i = int(next(it)); j = int(next(it)); v = float(next(it))
            observed.append((i, j, v))
            obs_val[(i, j)] = v
        re_ = int(next(it))
        for _ in range(re_):
            next(it); next(it)
        ce_ = int(next(it))
        for _ in range(ce_):
            next(it); next(it)
        q = int(next(it))
        query = []
        for _ in range(q):
            qi = int(next(it)); qj = int(next(it))
            query.append((qi, qj))
    except Exception:
        fail("bad input")

    if n <= 0 or m <= 0 or q <= 0:
        fail("degenerate instance")

    # ---- reconstruct ground truth deterministically from the embedded seed/testId ----
    # find test_id from seed: seed == 20260 + 97*test_id
    if (seed - 20260) % 97 != 0:
        fail("bad seed")
    test_id = (seed - 20260) // 97
    try:
        inst = build(test_id)
    except Exception:
        fail("cannot rebuild instance")
    if inst['n'] != n or inst['m'] != m or inst['seed'] != seed:
        fail("instance mismatch")
    true = inst['true']

    # ---- internal baseline B: row-mean-of-observed fallback global-mean predictor ----
    row_sum = [0.0] * n
    row_cnt = [0] * n
    tot_sum, tot_cnt = 0.0, 0
    for (i, j, v) in observed:
        row_sum[i] += v
        row_cnt[i] += 1
        tot_sum += v
        tot_cnt += 1
    global_mean = tot_sum / tot_cnt if tot_cnt > 0 else 0.0
    row_mean = [(row_sum[i] / row_cnt[i]) if row_cnt[i] > 0 else global_mean for i in range(n)]

    se_base = 0.0
    for (i, j) in query:
        pred = row_mean[i]
        d = pred - true[i][j]
        se_base += d * d
    rmse_base = math.sqrt(se_base / len(query))
    B = 1.0 / (1.0 + rmse_base)

    # ---- parse participant output: exactly q finite floats, in query order ----
    try:
        out_toks = open(out_path).read().split()
    except Exception:
        fail("cannot read output")
    if len(out_toks) != q:
        fail("expected %d predictions, got %d" % (q, len(out_toks)))
    preds = []
    for tok in out_toks:
        try:
            v = float(tok)
        except Exception:
            fail("non-numeric token %r" % tok)
        if not math.isfinite(v):
            fail("non-finite prediction %r" % tok)
        if abs(v) > 1e6:
            fail("prediction out of range %r" % tok)
        preds.append(v)

    se_part = 0.0
    for k, (i, j) in enumerate(query):
        d = preds[k] - true[i][j]
        se_part += d * d
    rmse_part = math.sqrt(se_part / len(query))
    F = 1.0 / (1.0 + rmse_part)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f rmse=%.6f rmse_base=%.6f Ratio: %.6f" % (F, B, rmse_part, rmse_base, sc / 1000.0))


if __name__ == "__main__":
    main()
