#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the glassblower annealing-hysteresis problem.

- Reads T_MIN, T_MAX, N, testId and the N noisy training rows from <in>.
- Regenerates the hidden first-order law (branch coefficients + the two
  spinodal temperatures T_down < T_up) entirely from testId -- the law lives
  ONLY here and in gen.py (never printed to the participant).
- Regenerates a HELD-OUT set of protocols -- a mix of extrapolated monotone
  cooling AND, dominantly, reheating protocols that visit the bistable
  window [T_down, T_up] from both directions -- entirely inside this file.
- Parses the participant's branch-diagram model:
      A1 B1        (m_hot(T)  = A1 + B1*(T-600))
      A2 B2        (m_cold(T) = A2 + B2*(T-600))
      Tdown Tup    (submitted spinodal temperatures, Tdown <= Tup required)
- Rolls the SAME simulate() state machine forward on every held-out protocol
  using the SUBMITTED model, and scores held-out RMSE against the TRUE
  m_final (minimisation):
      F = RMSE(prediction, truth)
      B = RMSE(constant = mean(training m), truth)     # internal baseline
      Ratio = min(1000, 100*B/F) / 1000
  A model that reproduces the mean training measurement (no T-dependence at
  all) reproduces the baseline (~0.1).  A model that fits BOTH branches but
  assumes zero hysteresis width (Tdown==Tup, the single-threshold trap) beats
  the baseline handily on unambiguous cases but is repeatedly wrong inside
  the bistable window.  Only a model that POSITS a nonzero hysteresis loop
  (branch memory) tracks those cases correctly -- but measurement noise plus
  the fact that Tup is never directly exercised by training data keep even a
  good model below the ceiling, leaving headroom.
"""
import sys, math, random

T_CENTER = 600.0
MAX_OUT_BYTES = 20000


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


# ---------- hidden law (identical to gen.py) ----------
def true_law(t):
    rng = random.Random(900001 + t * 7919)
    T_up = rng.uniform(540.0, 610.0)
    gap = rng.uniform(70.0, 220.0)
    T_down = T_up - gap
    A1 = rng.uniform(55.0, 65.0)
    B1 = rng.uniform(-0.02, 0.02)
    GAP0 = rng.uniform(22.0, 32.0)
    A2 = A1 - GAP0
    B2 = rng.uniform(-0.02, 0.02)
    return A1, B1, A2, B2, T_down, T_up


def branch_val(T, A, B):
    return A + B * (T - T_CENTER)


def simulate(breakpoints, A1, B1, A2, B2, T_down, T_up):
    branch = "hot"
    for i in range(len(breakpoints) - 1):
        Ti, Tj = breakpoints[i], breakpoints[i + 1]
        if branch == "hot" and Tj < Ti:
            if Ti > T_down >= Tj:
                branch = "cold"
        elif branch == "cold" and Tj > Ti:
            if Ti < T_up <= Tj:
                branch = "hot"
    Tend = breakpoints[-1]
    return branch_val(Tend, A1, B1) if branch == "hot" else branch_val(Tend, A2, B2)


# ---------- held-out protocol generation (never shown to the participant) ----------
def make_heldout(t, T_down, T_up):
    rng = random.Random(31337 + t * 15485863)
    protos = []

    # (1) extrapolated monotone cooling -- sanity / curve-fit check, low signal
    for _ in range(18):
        T0 = rng.uniform(860.0, 910.0)
        Tend = rng.uniform(250.0, T0 - 30.0)
        mid = min(T0 - 1.0, max(Tend + 1.0, (T0 + Tend) / 2.0 + rng.uniform(-20.0, 20.0)))
        protos.append([T0, mid, Tend])

    # (2) shallow reheat INTO the bistable window (stays below T_up): true=cold.
    # A single-threshold ("no hysteresis") model predicts hot here -> main trap.
    for _ in range(10):
        T0 = rng.uniform(860.0, 910.0)
        low = rng.uniform(250.0, T_down - 20.0)
        Tend = rng.uniform(T_down + 5.0, T_up - 12.0)
        protos.append([T0, low, Tend])

    # (3) full reheat PAST T_up then re-cool into/through the window: true=hot.
    for _ in range(18):
        T0 = rng.uniform(860.0, 910.0)
        low = rng.uniform(250.0, T_down - 20.0)
        peak = rng.uniform(T_up + 20.0, 830.0)
        Tend = rng.uniform(max(250.0, T_down - 80.0), T_up - 10.0)
        protos.append([T0, low, peak, Tend])

    # (4) multi-cycle cool/reheat -- stress full path simulation
    for _ in range(8):
        T0 = rng.uniform(860.0, 910.0)
        pts = [T0]
        cur = T0
        for _c in range(rng.choice([2, 3])):
            low = rng.uniform(250.0, T_down - 10.0)
            pts.append(low)
            high = rng.uniform(T_down - 10.0, 850.0)
            pts.append(high)
            cur = high
        Tend = rng.uniform(250.0, cur - 5.0)
        pts.append(Tend)
        protos.append(pts)

    # (5) winding path ending deep cold -- unambiguous, sanity check
    for _ in range(16):
        T0 = rng.uniform(860.0, 910.0)
        pts = [T0]
        cur = T0
        for _s in range(rng.choice([2, 3])):
            nxt = rng.uniform(250.0, max(255.0, cur - 15.0))
            pts.append(nxt)
            cur = nxt
        protos.append(pts)

    return protos


# ---------- parse instance ----------
def read_instance(path):
    with open(path) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    head = lines[0].split()
    T_MIN, T_MAX, N, t = float(head[0]), float(head[1]), int(head[2]), int(head[3])
    m_sum = 0.0
    m_count = 0
    for ln in lines[1 : 1 + N]:
        toks = ln.split()
        m_sum += float(toks[-1])
        m_count += 1
    return t, m_sum / max(1, m_count)


# ---------- parse participant model ----------
def read_model(path):
    try:
        with open(path, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) != 3:
        fail("expected exactly 3 non-empty output lines, got %d" % len(lines))
    vals = []
    for ln in lines:
        toks = ln.split()
        if len(toks) != 2:
            fail("each line must have exactly 2 numbers")
        for tok in toks:
            try:
                v = float(tok)
            except Exception:
                fail("non-numeric value '%s'" % tok)
            if not math.isfinite(v):
                fail("non-finite value")
            vals.append(v)
    A1, B1, A2, B2, T_down, T_up = vals
    if abs(A1) > 1000.0 or abs(A2) > 1000.0:
        fail("branch intercept out of range")
    if abs(B1) > 5.0 or abs(B2) > 5.0:
        fail("branch slope out of range")
    if T_down > T_up:
        fail("T_down must be <= T_up")
    if T_down < -500.0 or T_up > 1500.0:
        fail("threshold out of range")
    return A1, B1, A2, B2, T_down, T_up


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        t, mean_train_m = read_instance(inf)
    except Exception:
        fail("bad instance file")

    A1, B1, A2, B2, T_down, T_up = read_model(outf)

    tA1, tB1, tA2, tB2, tT_down, tT_up = true_law(t)
    protos = make_heldout(t, tT_down, tT_up)

    se_model = 0.0
    se_base = 0.0
    for proto in protos:
        truth = simulate(proto, tA1, tB1, tA2, tB2, tT_down, tT_up)
        pred = simulate(proto, A1, B1, A2, B2, T_down, T_up)
        if not math.isfinite(pred):
            fail("non-finite prediction during rollout")
        se_model += (pred - truth) ** 2
        se_base += (mean_train_m - truth) ** 2

    n = len(protos)
    F = math.sqrt(se_model / n)
    B = math.sqrt(se_base / n)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("heldout_RMSE=%.6f baseline_RMSE=%.6f n_heldout=%d  Ratio: %.6f"
          % (F, B, n, sc / 1000.0))


if __name__ == "__main__":
    main()
