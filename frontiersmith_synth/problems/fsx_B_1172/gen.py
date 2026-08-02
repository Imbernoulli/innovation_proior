#!/usr/bin/env python3
"""gen.py <testId> -- one seismic-refraction/reflection instance for fsx_B_1172.
Deterministic: everything is seeded from testId only. Prints the instance to stdout.
"""
import sys, random, math

V_MIN, V_MAX = 800.0, 6000.0
H_MIN, H_MAX = 20.0, 400.0
NLAYERS = {1: 3, 2: 4, 3: 5, 4: 5, 5: 5, 6: 6, 7: 6, 8: 6, 9: 6, 10: 6}
DIP     = {4: 3, 5: 2, 6: 3, 7: 4, 8: 2, 9: 5, 10: 3}   # test_id -> hidden low-velocity layer index (or absent)

F_GIVEN = [0.10, 0.24, 0.38, 0.52, 0.66, 0.80, 0.94]     # offsets shown to the solver, as a fraction of each branch's window
F_HELD  = [0.17, 0.45, 0.73]                             # held-out fractions used ONLY by the checker


def branch_line(i, h, v, N):
    """(slope, intercept) of first-arrival branch i (0=direct wave, i=1..N-1 = head wave
    critically refracted along the top of layer i+1)."""
    if i == 0:
        return 1.0 / v[1], 0.0
    vi1 = v[i + 1]
    c = 0.0
    for j in range(1, i + 1):
        vj = v[j]
        c += h[j] * math.sqrt(vi1 * vi1 - vj * vj) / (vj * vi1)
    return 1.0 / vi1, 2.0 * c


def valid_branches(v, N):
    """Branch i is a real head wave iff layer i+1 is faster than every layer above it
    (Snell's law needs a real critical angle at each overlying interface)."""
    running_max = v[1]
    vis = [0]
    for i in range(1, N):
        vi1 = v[i + 1]
        if vi1 > running_max:
            vis.append(i)
        running_max = max(running_max, vi1)
    return vis


def lower_envelope(lines):
    """Lower envelope (min) of lines given in DECREASING-slope order. Returns
    [(x_start, x_end, tag, slope, intercept), ...] left to right; last x_end = inf."""
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
        v[m] = v[m - 1] * dip_factor          # low-velocity zone: v_m < v_{m-1}
        v[m + 1] = v[m - 1] + rng.uniform(400.0, 750.0)   # velocity recovers past the pre-dip max
        for i in range(m + 2, N + 1):
            v[i] = v[i - 1] + rng.uniform(350.0, 700.0)
    h = [0.0] * N
    for i in range(1, N):
        h[i] = rng.uniform(H_MIN, H_MAX)
    return v, h


def true_model(test_id, max_attempts=20000):
    """Deterministic function of test_id ONLY. Retries (still deterministically, via an
    attempt counter folded into the seed) until the planted layer structure actually
    produces the intended visible/invisible branch pattern with well-formed windows."""
    N = NLAYERS[test_id]
    m = DIP.get(test_id)
    expected_vis = set(range(N)) - ({m - 1} if m is not None else set())
    for attempt in range(max_attempts):
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


def make_offsets(windows, fracs, last_extra):
    xs = []
    for (xstart, xend, tag, mm, bb) in windows:
        xend_eff = xstart + last_extra if xend == math.inf else xend
        width = xend_eff - xstart
        for f in fracs:
            xs.append(xstart + f * width)
    return xs


def main():
    test_id = int(sys.argv[1])
    N, m, v, h, windows = true_model(test_id)
    last_extra = max(3000.0, 2.2 * (windows[-1][0] - (windows[-2][0] if len(windows) > 1 else 0.0)))
    given_x = make_offsets(windows, F_GIVEN, last_extra)
    given_t = [first_arrival(x, h, v, N) for x in given_x]
    taus = tau_of(h, v, N)

    out = []
    out.append(str(test_id))
    out.append(str(N))
    out.append(str(len(given_x)))
    for x, t in zip(given_x, given_t):
        out.append("%.6f %.6f" % (x, t))
    out.append(" ".join("%.6f" % t for t in taus))
    out.append("%.2f %.2f %.2f %.2f" % (V_MIN, V_MAX, H_MIN, H_MAX))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
