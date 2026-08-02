#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the battery-ECM
identification problem (family battery-ecm-identify).

Feasibility (any violation -> Ratio: 0.0):
  - output is exactly "R0 K M" then K lines of "R_i C_i"; no extra/missing
    tokens; every value finite (rejects nan/inf)
  - R0 in [R0_lo,R0_hi]; K integer in [0,Kmax]; each R_i in [R_lo,R_hi],
    C_i in [C_lo,C_hi]; tau_i=R_i*C_i in [tau_lo,tau_hi]; M in [M_lo,M_hi]

Objective (maximize): the submitted ECM is simulated FORWARD (identical
recursion to gen.py, identical tau_h/capacity/soc_init/OCV) on a HELD-OUT
drive cycle from the SAME cell, regenerated deterministically from the test
id (never printed in <in>), whose current profile mixes a genuinely
different set of pulse durations than the training cycle. F = RMSE(predicted
terminal voltage, true terminal voltage) there. B = RMSE of the checker's own
naive single-generic-branch reference fit (trained on the SAME visible data)
evaluated on the same held-out cycle.
    Ratio = min(1000, 100*B / max(1e-9, F)) / 1000
"""
import sys
import math
import random

DT = 1.0
TAU_H = 90.0
CAPACITY_AH = 5.0
SOC_INIT = 0.85
T_GENERIC = 60.0  # the checker's own naive reference branch time constant


def fail(reason):
    sys.stdout.write("INFEASIBLE: %s\n" % reason)
    sys.stdout.write("Ratio: 0.0\n")
    sys.exit(0)


def ocv_true(soc):
    return 3.00 + 1.20 * soc - 0.30 * soc * soc


# ---- HIDDEN ground truth (identical formulas duplicated from gen.py) ----
def gt_params(tid):
    trap = tid in (4, 5, 6, 9, 10)
    if not trap:
        taus = [15.0, 250.0]
        Rs = [0.028, 0.020]
    else:
        taus = [15.0, 180.0, 230.0]
        Rs = [0.028, 0.014, 0.014]
    R0 = 0.045 + 0.003 * (tid % 5)
    M_true = 0.012 + 0.0015 * tid
    sigma = 0.003
    return taus, Rs, R0, M_true, sigma, trap


PROFILES = {
    "resolved_train": [(10, 20), (220, 280)],
    "resolved_held": [(5, 10), (60, 100), (300, 400)],
    "trap_train": [(10, 20), (190, 210)],
    "trap_held": [(5, 15), (120, 160), (350, 450)],
}


def gen_profile(n, seed, clusters):
    rnd = random.Random(seed)
    I = []
    while len(I) < n:
        lo, hi = rnd.choice(clusters)
        dur = rnd.randint(lo, hi)
        amp = rnd.uniform(1.0, 4.0) * rnd.choice([-1, 1])
        I += [amp] * dur
    return I[:n]


def simulate(I, R0, taus, Rs, M, tau_h=TAU_H, cap_ah=CAPACITY_AH, soc_init=SOC_INIT,
             noise_sigma=0.0, noise_seed=0):
    n = len(I)
    K = len(taus)
    a = [math.exp(-DT / t) for t in taus]
    ah = math.exp(-DT / tau_h)
    vb = [0.0] * K
    h = 0.0
    soc = soc_init
    V = [0.0] * n
    rnd = random.Random(noise_seed) if noise_sigma > 0 else None
    for k in range(n):
        Ik = I[k]
        vt = ocv_true(soc) - R0 * Ik - sum(vb) - h
        if rnd is not None:
            vt += rnd.gauss(0.0, noise_sigma)
        V[k] = vt
        for i in range(K):
            vb[i] = a[i] * vb[i] + Rs[i] * (1 - a[i]) * Ik
        sgn = 0.0 if Ik == 0 else (1.0 if Ik > 0 else -1.0)
        h = ah * h + M * (1 - ah) * sgn
        soc -= Ik * DT / (3600.0 * cap_ah)
    return V


def rmse(a, b):
    n = len(a)
    s = 0.0
    for i in range(n):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s / n)


# ---- linear-in-parameters OLS helper (used ONLY for the checker's own
# naive reference baseline fit -- not part of the participant's model) ----
def build_branch_col(I, tau):
    n = len(I)
    a = math.exp(-DT / tau)
    out = [0.0] * n
    state = 0.0
    for k in range(n):
        out[k] = state
        state = a * state + (1 - a) * I[k]
    return out


def build_hyst_col(I, tau_h):
    n = len(I)
    ah = math.exp(-DT / tau_h)
    out = [0.0] * n
    h = 0.0
    for k in range(n):
        out[k] = h
        sgn = 0.0 if I[k] == 0 else (1.0 if I[k] > 0 else -1.0)
        h = ah * h + (1 - ah) * sgn
    return out


def socs_of(I, soc_init, cap_ah):
    soc = soc_init
    out = []
    for k in range(len(I)):
        out.append(soc)
        soc -= I[k] * DT / (3600.0 * cap_ah)
    return out


def solve3(ATA, ATb):
    """Solve a small (3x3) linear system via Gaussian elimination with
    partial pivoting -- no external deps, deterministic."""
    n = len(ATb)
    M = [row[:] + [ATb[i]] for i, row in enumerate(ATA)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                M[r] = [M[r][c] - f * M[col][c] for c in range(n + 1)]
    return [M[i][n] for i in range(n)]


def naive_baseline_fit(I_train, V_train):
    """Single generic RC branch (fixed tau=T_GENERIC, not adapted to the
    data) + hysteresis, ordinary least squares. This is the checker's own
    positive feasible reference construction."""
    n = len(I_train)
    socs = socs_of(I_train, SOC_INIT, CAPACITY_AH)
    target = [ocv_true(socs[k]) - V_train[k] for k in range(n)]
    Zb = build_branch_col(I_train, T_GENERIC)
    Zh = build_hyst_col(I_train, TAU_H)
    cols = [I_train, Zb, Zh]
    ATA = [[sum(cols[i][k] * cols[j][k] for k in range(n)) for j in range(3)] for i in range(3)]
    ATb = [sum(cols[i][k] * target[k] for k in range(n)) for i in range(3)]
    sol = solve3(ATA, ATb)
    if sol is None:
        return 0.05, T_GENERIC, 0.02, 0.0
    R0, Rb, M = sol
    Rb = max(Rb, 0.0)
    return R0, T_GENERIC, Rb, M


def read_tokens(path, cap_bytes=30_000_000):
    try:
        with open(path, "r") as f:
            data = f.read(cap_bytes + 1)
    except Exception as e:
        fail(f"cannot read file: {e}")
    if len(data) > cap_bytes:
        fail("output far too large")
    return data.split()


def parse_float(tok):
    try:
        v = float(tok)
    except (ValueError, TypeError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    itoks = read_tokens(in_path)
    it = iter(itoks)

    def nxt():
        try:
            return next(it)
        except StopIteration:
            fail("truncated input (bug in generator)")

    n_train = int(nxt()); tid = int(nxt()); dt = float(nxt()); kmax = int(nxt())
    r0_lo = float(nxt()); r0_hi = float(nxt())
    r_lo = float(nxt()); r_hi = float(nxt())
    c_lo = float(nxt()); c_hi = float(nxt())
    tau_lo = float(nxt()); tau_hi = float(nxt())
    m_lo = float(nxt()); m_hi = float(nxt())
    tau_h = float(nxt()); cap_ah = float(nxt()); soc_init = float(nxt())
    n_ocv = int(nxt())
    for _ in range(n_ocv):
        nxt(); nxt()  # OCV table not needed by the checker (fixed formula known)
    I_train = [float(nxt()) for _ in range(n_train)]
    V_train = [float(nxt()) for _ in range(n_train)]

    taus_t, Rs_t, R0_t, M_t, sigma, trap = gt_params(tid)
    n_held = 2600
    hkey = "trap_held" if trap else "resolved_held"
    I_held = gen_profile(n_held, seed=30_000 + tid, clusters=PROFILES[hkey])
    V_held_true = simulate(I_held, R0_t, taus_t, Rs_t, M_t,
                            noise_sigma=sigma, noise_seed=40_000 + tid)

    # ---------------- parse participant output ----------------
    otoks = read_tokens(out_path, cap_bytes=5_000_000)
    if len(otoks) < 3:
        fail("output too short")
    oit = iter(otoks)

    def og():
        try:
            return next(oit)
        except StopIteration:
            fail("truncated output")

    R0v = parse_float(og())
    if R0v is None:
        fail("R0 not finite")
    Ktok = og()
    try:
        K = int(Ktok)
    except ValueError:
        fail("K not an integer")
    Mv = parse_float(og())
    if Mv is None:
        fail("M not finite")

    if not (r0_lo - 1e-9 <= R0v <= r0_hi + 1e-9):
        fail("R0 out of bounds")
    if not (0 <= K <= kmax):
        fail("K out of bounds")
    if not (m_lo - 1e-9 <= Mv <= m_hi + 1e-9):
        fail("M out of bounds")

    taus_v = []
    Rs_v = []
    for _ in range(K):
        Rv = parse_float(og())
        Cv = parse_float(og())
        if Rv is None or Cv is None:
            fail("R_i/C_i not finite")
        if not (r_lo - 1e-12 <= Rv <= r_hi + 1e-9):
            fail("R_i out of bounds")
        if not (c_lo - 1e-9 <= Cv <= c_hi + 1e-9):
            fail("C_i out of bounds")
        tau = Rv * Cv
        if not (tau_lo - 1e-6 <= tau <= tau_hi + 1e-6):
            fail("tau_i=R_i*C_i out of bounds")
        taus_v.append(tau)
        Rs_v.append(Rv)

    # no trailing tokens allowed
    leftover = list(oit)
    if leftover:
        fail("extra trailing tokens in output")

    V_pred = simulate(I_held, R0v, taus_v, Rs_v, Mv, tau_h=tau_h, cap_ah=cap_ah, soc_init=soc_init)
    for v in V_pred:
        if v != v or v in (float("inf"), float("-inf")):
            fail("simulated prediction non-finite")

    F = rmse(V_pred, V_held_true)

    R0b, taub, Rb, Mb = naive_baseline_fit(I_train, V_train)
    V_base = simulate(I_held, R0b, [taub], [Rb], Mb, tau_h=tau_h, cap_ah=cap_ah, soc_init=soc_init)
    B = rmse(V_base, V_held_true)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    sys.stdout.write("F=%.6f B=%.6f\n" % (F, B))
    sys.stdout.write("Ratio: %.6f\n" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
