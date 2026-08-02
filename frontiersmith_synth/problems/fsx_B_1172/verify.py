#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_1172 (seismic-layer-recover).

Ground truth is NOT read from <in>; it is regenerated deterministically from the
test_id (first token of <in>) via the SAME true_model() used by gen.py. Score =
0.5*(depth-to-interface fit) + 0.5*(held-out travel-time fit), both against the
hidden truth, normalized by an internal trivial baseline B.
"""
import sys, math

V_MIN, V_MAX = 800.0, 6000.0
H_MIN, H_MAX = 20.0, 400.0
NLAYERS = {1: 3, 2: 4, 3: 5, 4: 5, 5: 5, 6: 6, 7: 6, 8: 6, 9: 6, 10: 6}
DIP     = {4: 3, 5: 2, 6: 3, 7: 4, 8: 2, 9: 5, 10: 3}

F_HELD = [0.17, 0.45, 0.73]

TAU_D = 0.10
TAU_T = 0.10


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def branch_line(i, h, v, N):
    if i == 0:
        return 1.0 / v[1], 0.0
    vi1 = v[i + 1]
    c = 0.0
    for j in range(1, i + 1):
        vj = v[j]
        c += h[j] * math.sqrt(vi1 * vi1 - vj * vj) / (vj * vi1)
    return 1.0 / vi1, 2.0 * c


def valid_branches(v, N):
    running_max = v[1]
    vis = [0]
    for i in range(1, N):
        vi1 = v[i + 1]
        if vi1 > running_max:
            vis.append(i)
        running_max = max(running_max, vi1)
    return vis


def lower_envelope(lines):
    stack = []
    for m, b, tag in lines:
        while len(stack) >= 2:
            m1, b1, _ = stack[-2]
            m2, b2, _ = stack[-1]
            x12 = (b2 - b1) / (m1 - m2)
            x2new = (b - b2) / (m2 - m)
            if x2new <= x12:
                stack.pop()
            else:
                break
        stack.append((m, b, tag))
    windows = []
    for idx in range(len(stack)):
        m, b, tag = stack[idx]
        xs = 0.0 if idx == 0 else (b - stack[idx - 1][1]) / (stack[idx - 1][0] - m)
        xe = math.inf if idx == len(stack) - 1 else (stack[idx + 1][1] - b) / (m - stack[idx + 1][0])
        windows.append((xs, xe, tag, m, b))
    return windows


def gen_v_h(rng, N, m):
    v = [0.0] * (N + 1)
    v[1] = rng.uniform(900.0, 1400.0)
    for i in range(2, N + 1):
        v[i] = v[i - 1] + rng.uniform(350.0, 700.0)
    if m is not None:
        dip_factor = rng.uniform(0.6, 0.8)
        v[m] = v[m - 1] * dip_factor
        v[m + 1] = v[m - 1] + rng.uniform(400.0, 750.0)
        for i in range(m + 2, N + 1):
            v[i] = v[i - 1] + rng.uniform(350.0, 700.0)
    h = [0.0] * N
    for i in range(1, N):
        h[i] = rng.uniform(H_MIN, H_MAX)
    return v, h


def true_model(test_id, max_attempts=20000):
    N = NLAYERS[test_id]
    m = DIP.get(test_id)
    expected_vis = set(range(N)) - ({m - 1} if m is not None else set())
    for attempt in range(max_attempts):
        import random
        rng = random.Random((1000 * test_id + 7) * 100003 + attempt)
        v, h = gen_v_h(rng, N, m)
        vis = valid_branches(v, N)
        if set(vis) != expected_vis:
            continue
        lines = [(*branch_line(i, h, v, N), i) for i in vis]
        lines.sort(key=lambda t: -t[0])
        windows = lower_envelope(lines)
        if set(w[2] for w in windows) != expected_vis:
            continue
        if any(windows[k][1] - windows[k][0] < 1e-6 for k in range(len(windows) - 1)):
            continue
        return N, m, v, h, windows
    raise RuntimeError("true_model: no convergence for test_id=%d" % test_id)


def first_arrival(x, h, v, N):
    best = x / v[1]
    running_max = v[1]
    for i in range(1, N):
        vi1 = v[i + 1]
        if vi1 > running_max:
            c = 0.0
            for j in range(1, i + 1):
                vj = v[j]
                c += h[j] * math.sqrt(vi1 * vi1 - vj * vj) / (vj * vi1)
            t = x / vi1 + 2.0 * c
            if t < best:
                best = t
        running_max = max(running_max, vi1)
    return best


def tau_of(h, v, N):
    taus = []
    s = 0.0
    for k in range(1, N):
        s += h[k] / v[k]
        taus.append(2.0 * s)
    return taus


def held_offsets(windows, last_extra):
    xs = []
    for (xstart, xend, tag, mm, bb) in windows:
        xend_eff = xstart + last_extra if xend == math.inf else xend
        width = xend_eff - xstart
        for f in F_HELD:
            xs.append(xstart + f * width)
    return xs


def depth_score(h_sub, h_true, N):
    scores = []
    Dk = 0.0
    Dt = 0.0
    for k in range(1, N):
        Dk += h_sub[k]
        Dt += h_true[k]
        rel = abs(Dk - Dt) / Dt
        scores.append(math.exp(-rel / TAU_D))
    return sum(scores) / len(scores)


def time_score(h_sub, v_sub, held_x, h_true, v_true, N):
    scores = []
    for x in held_x:
        tt = first_arrival(x, h_true, v_true, N)
        tp = first_arrival(x, h_sub, v_sub, N)
        if not math.isfinite(tp):
            scores.append(0.0)
            continue
        rel = abs(tp - tt) / tt
        scores.append(math.exp(-rel / TAU_T))
    return sum(scores) / len(scores)


def quality(h_sub, v_sub, h_true, v_true, held_x, N):
    ds = depth_score(h_sub, h_true, N)
    ts = time_score(h_sub, v_sub, held_x, h_true, v_true, N)
    return 0.5 * ds + 0.5 * ts


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        in_toks = open(in_path).read().split()
        it = iter(in_toks)
        test_id = int(next(it))
        N_in = int(next(it))
    except Exception:
        fail("bad input file")

    if test_id not in NLAYERS:
        fail("unknown test_id")

    try:
        N, m, v_true, h_true, windows = true_model(test_id)
    except Exception:
        fail("ground truth reconstruction failed")

    if N != N_in:
        fail("input/ground-truth N mismatch")

    last_extra = max(3000.0, 2.2 * (windows[-1][0] - (windows[-2][0] if len(windows) > 1 else 0.0)))
    held_x = held_offsets(windows, last_extra)

    # ---- parse participant output: N-1 lines "h_i v_i", then 1 line "v_N" ----
    try:
        out_toks = open(out_path).read().split()
    except Exception:
        fail("cannot read output")

    if len(out_toks) != 2 * (N - 1) + 1:
        fail("expected %d tokens (N-1 h/v pairs + v_N), got %d" % (2 * (N - 1) + 1, len(out_toks)))

    try:
        vals = [float(tok) for tok in out_toks]
    except Exception:
        fail("non-numeric token")

    if not all(math.isfinite(x) for x in vals):
        fail("nan/inf in output")

    h_sub = [0.0] * N
    v_sub = [0.0] * (N + 1)
    for k in range(1, N):
        h_sub[k] = vals[2 * (k - 1)]
        v_sub[k] = vals[2 * (k - 1) + 1]
    v_sub[N] = vals[2 * (N - 1)]

    H_LO, H_HI = 0.1, 8.0 * H_MAX
    V_LO, V_HI = 1.0, 8.0 * V_MAX
    for k in range(1, N):
        if not (H_LO <= h_sub[k] <= H_HI):
            fail("h_%d=%g out of feasible range" % (k, h_sub[k]))
        if not (V_LO <= v_sub[k] <= V_HI):
            fail("v_%d=%g out of feasible range" % (k, v_sub[k]))
    if not (V_LO <= v_sub[N] <= V_HI):
        fail("v_%d=%g out of feasible range" % (N, v_sub[N]))

    F = quality(h_sub, v_sub, h_true, v_true, held_x, N)

    # ---- internal trivial baseline B: constant velocity = v_1, evenly split depth ----
    v1 = v_true[1]
    Dtot = (tau_of(h_true, v_true, N)[-1]) * v1 / 2.0
    h_base = [0.0] * N
    v_base = [0.0] * (N + 1)
    for i in range(1, N):
        h_base[i] = Dtot / (N - 1)
        v_base[i] = v1
    v_base[N] = v1
    B = quality(h_base, v_base, h_true, v_true, held_x, N)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    ratio = sc / 1000.0
    print("depth+heldout-time quality F=%.6f baseline B=%.6f Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
