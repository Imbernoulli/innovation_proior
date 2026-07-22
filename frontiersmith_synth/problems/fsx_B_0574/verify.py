#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the drop-away rocket-staging task.

Instance (from <in>):
    line 1:  S P M_total kappa
    line 2:  m_e T v_e g E_max
    line 3:  L_1 ... L_S

Participant artifact (from <out>): S lines "n_i p_i"  (engine count, propellant
mass) for stage i = 1..S, stage 1 = first to burn / first jettisoned.

Ascent model (exact, deterministic).  Each stage i is built from n_i identical
engine modules (dry mass m_e each) plus a propellant tank whose structural mass
is kappa*p_i.  So the stage's wet mass is  n_i*m_e + (1+kappa)*p_i  and its dry
mass (kept until jettison) is  n_i*m_e + kappa*p_i.  With the mass still on the
vehicle when stage i ignites,
    m_start_i = P + sum_{j>=i} ( n_j*m_e + (1+kappa)*p_j )
    m_end_i   = m_start_i - p_i          # only propellant leaves; tank stays
    dv_i      = v_e * ln(m_start_i / m_end_i)                (Tsiolkovsky)
    burn_i    = p_i * v_e / (n_i * T)     # propellant / mass-flow (= thrust/v_e)
    loss_i    = L_i * g * burn_i          # altitude/stage-indexed gravity+drag loss
    net_i     = dv_i - loss_i
Then the whole stage ( n_i*m_e + kappa*p_i ) is jettisoned and stage i+1 ignites.
Final payload velocity  V = sum_i net_i.

Scoring (maximisation).  Internal baseline B = velocity of the one-engine,
equal-propellant-split vehicle the grader builds itself.
    sc    = min(1000, 100 * V / max(1e-9, B))
    Ratio = max(0, sc) / 1000
so a one-engine equal-split entry reproduces B (~0.1) and a ~10x-better vehicle
would cap at 1.0.  Any feasibility violation prints Ratio: 0.0.
"""
import sys, math

TOL = 1e-6


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as fh:
        toks = fh.read().split()
    it = iter(toks)
    S = int(float(next(it)))
    P = float(next(it))
    M_total = float(next(it))
    kappa = float(next(it))
    m_e = float(next(it))
    T = float(next(it))
    v_e = float(next(it))
    g = float(next(it))
    E_max = int(float(next(it)))
    L = [float(next(it)) for _ in range(S)]
    return dict(S=S, P=P, M_total=M_total, kappa=kappa, m_e=m_e, T=T, v_e=v_e,
                g=g, E_max=E_max, L=L)


def velocity(q, n, p):
    """Exact final payload velocity for engine counts n[] and propellant p[]."""
    S, P, m_e, kappa = q["S"], q["P"], q["m_e"], q["kappa"]
    T, v_e, g, L = q["T"], q["v_e"], q["g"], q["L"]
    # wet mass carried at/above stage i
    suffix = [0.0] * (S + 1)
    for i in range(S - 1, -1, -1):
        suffix[i] = suffix[i + 1] + n[i] * m_e + (1.0 + kappa) * p[i]
    V = 0.0
    for i in range(S):
        m_start = P + suffix[i]
        m_end = m_start - p[i]
        if m_end <= 0.0 or m_start <= 0.0:
            return float("-inf")
        dv = v_e * math.log(m_start / m_end)
        burn = p[i] * v_e / (n[i] * T)
        loss = L[i] * g * burn
        V += dv - loss
    return V


BASE_N = 1  # engines per stage in the internal baseline (must match trivial.py)


def baseline_velocity(q):
    """BASE_N engines per stage, equal propellant split -> the internal baseline B."""
    S, P, M_total, m_e, kappa = q["S"], q["P"], q["M_total"], q["m_e"], q["kappa"]
    n = [BASE_N] * S
    total_prop = (M_total - P - S * BASE_N * m_e) / (1.0 + kappa)
    if total_prop <= 0:
        total_prop = 0.0
    p = [total_prop / S] * S
    return velocity(q, n, p)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    q = read_instance(sys.argv[1])
    S, P, M_total, m_e, kappa, E_max = (q["S"], q["P"], q["M_total"], q["m_e"],
                                        q["kappa"], q["E_max"])

    # ---- parse participant output (bounded read) ----
    try:
        with open(sys.argv[2], "rb") as fh:
            raw = fh.read(1 << 20)
    except Exception:
        fail("cannot read output")
    text = raw.decode("utf-8", "replace")
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    if len(rows) != S:
        fail("need exactly %d stage lines, got %d" % (S, len(rows)))

    n = [0] * S
    p = [0.0] * S
    for i, r in enumerate(rows):
        if len(r) != 2:
            fail("stage %d: expected 'n p'" % (i + 1))
        try:
            ni = int(r[0])
            pi = float(r[1])
        except Exception:
            fail("stage %d: unparseable" % (i + 1))
        if not math.isfinite(pi):
            fail("stage %d: non-finite propellant" % (i + 1))
        if ni < 1 or ni > E_max:
            fail("stage %d: engine count out of [1,%d]" % (i + 1, E_max))
        if pi < 0.0:
            fail("stage %d: negative propellant" % (i + 1))
        n[i] = ni
        p[i] = pi

    total_mass = P + sum(n[i] * m_e + (1.0 + kappa) * p[i] for i in range(S))
    if total_mass > M_total * (1.0 + 1e-9) + TOL:
        fail("mass budget exceeded: %.6f > %.6f" % (total_mass, M_total))

    V = velocity(q, n, p)
    if not math.isfinite(V):
        fail("degenerate vehicle")

    B = baseline_velocity(q)
    if not math.isfinite(B) or B <= 0.0:
        B = 1e-6

    sc = min(1000.0, 100.0 * V / max(1e-9, B))
    if sc < 0.0:
        sc = 0.0
    print("V=%.4f B=%.4f  Ratio: %.6f" % (V, B, sc / 1000.0))


if __name__ == "__main__":
    main()
