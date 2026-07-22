import sys, math, random

# ---------------------------------------------------------------------------
# coproduct-envelope-riding (format C, MINIMIZE fuel+dump cost)
#   `python3 gen.py <testId>`  prints ONE instance to stdout.
#   Deterministic in testId only (seeded via random.Random).
#
# Theme: a combined heat-and-power (CHP) plant serving a campus. Each period t
# the plant must produce EXACTLY the demanded electrical power P_t = DP[t]
# (power cannot be buffered -- "one-sided buffering": only heat may be stored).
# At that fixed power level, the plant's extraction-condensing turbine allows
# heat output Q_t anywhere in a power-coupled interval [L(t), U(t)] (the
# vertical slice, at P = DP[t], of a seeded 2D feasible polygon in (P,Q)
# space). A tank buffers HEAT ONLY across periods; heat may also be dumped
# (wasted) for a small per-unit fee ("dump-option-pricing").
#
# Instance layout (stdin):
#   line 1:  T Cap S_init
#   line 2:  Pmin Pmax Qcap alpha beta Gamma delta Epsilon
#   line 3:  b kappa dumpfee
#   next T lines: DP[t] DQ[t]      (t = 1..T)
#
# Envelope (computed identically by gen/checker from line 2's params):
#   U(P) = clip( min(Qcap, (Gamma - alpha*P)/beta), 0, None )
#   L(P) = clip( delta*P - Epsilon, 0, U(P) )
# The generator always places DQ[t] inside [L(t), U(t)] (heat demand is
# always technically satisfiable by SOME single operating point) so every
# instance is guaranteed feasible; the difficulty is in the FUEL COST, not
# in raw feasibility.
#
# Fuel cost at period t for a chosen Q_t in [L(t),U(t)]:
#   fuel(t,Q) = b*Q + kappa*(Q-L(t))*(U(t)-Q)
# The last term is a concave "part-load inefficiency" bump: zero exactly at
# the two envelope endpoints Q=L(t) and Q=U(t), positive (worst) at the
# midpoint. So, at any single period, only the ENVELOPE VERTICES are fuel-
# efficient; operating strictly inside the interval is provably worse. This
# is the trap: pointwise demand-tracking (Q_t = DQ[t]) sits at an interior
# point almost every period and pays the bump every time. Riding the two
# vertices instead, and using the heat tank (+ paying occasional dump fees)
# to reconcile the mismatch against actual demand, avoids the bump almost
# entirely -- inverting the usual role of storage (it exists to make
# vertex-hopping feasible, not to track demand).
# ---------------------------------------------------------------------------

TRAP_IDS = {4, 7, 9, 10}     # >=3 cases where naive demand-tracking is punished hard
TIGHT_TANK_IDS = {7, 10}     # trap cases that also force real dump decisions


def envelope(P, Pmin, Qcap, alpha, beta, Gamma, delta, Epsilon):
    U = min(Qcap, (Gamma - alpha * P) / beta)
    if U < 0.0:
        U = 0.0
    L = delta * P - Epsilon
    if L < 0.0:
        L = 0.0
    if L > U:
        L = U
    return L, U


def main():
    t_id = int(sys.argv[1])
    t_id = max(1, min(10, t_id))
    rng = random.Random(20260000 + 97 * t_id)

    # ---- difficulty ladder: period count -----------------------------------
    T_by_id = {1: 8, 2: 12, 3: 16, 4: 24, 5: 30, 6: 40,
               7: 60, 8: 90, 9: 140, 10: 220}
    T = T_by_id[t_id]

    # ---- seeded extraction-condensing polygon (fixed for the whole horizon)
    Pmin = rng.randint(15, 25)
    span_p = rng.randint(35, 70)
    Pmax = Pmin + span_p

    Qcap = round(rng.uniform(28, 46), 2)
    alpha = round(rng.uniform(0.6, 1.3), 3)
    beta = 1.0
    U_at_Pmax = round(rng.uniform(6, 15), 2)          # heat ceiling shrinks at rated power
    Gamma = round(alpha * Pmax + U_at_Pmax, 3)

    L_at_Pmax = round(rng.uniform(0.0, 0.45) * U_at_Pmax, 2)  # min-extraction floor at high power
    delta = round(L_at_Pmax / span_p, 4)
    Epsilon = round(delta * Pmin, 4)

    # ---- fuel / dump economics -----------------------------------------
    # (Power's own fuel draw a*DP[t] is identical for every feasible plan --
    # DP[t] is exogenous demand, not a choice -- so it is left OUT of the
    # graded objective entirely; only the heat-dependent terms below, which
    # actually differ between strategies, are scored.)
    b = round(rng.uniform(0.8, 1.6), 3)
    kfactor = round(rng.uniform(1.1, 2.2), 3)
    kappa = round(kfactor * 4.0 * b / max(1.0, Qcap), 5)
    dumpfee = round(rng.uniform(0.15, 0.45) * b, 4)

    # ---- tank ------------------------------------------------------------
    if t_id in TIGHT_TANK_IDS:
        Cap = round(rng.uniform(0.5, 0.85) * Qcap, 2)
    else:
        Cap = round(rng.uniform(1.6, 4.0) * Qcap, 2)
    S_init = round(rng.uniform(0.25, 0.75) * Cap, 2)

    # ---- demand series -----------------------------------------------------
    DP = []
    DQ = []
    trap = t_id in TRAP_IDS
    phase = rng.uniform(0, 2 * math.pi)
    for t in range(1, T + 1):
        # power demand: a smooth day-load curve + seeded jitter, clipped to [Pmin,Pmax]
        frac = t / max(1, T)
        wave = 0.5 + 0.5 * math.sin(2 * math.pi * frac + phase)
        p = Pmin + wave * span_p + rng.uniform(-0.06, 0.06) * span_p
        p = max(Pmin, min(Pmax, p))
        p = round(p, 3)
        L, U = envelope(p, Pmin, Qcap, alpha, beta, Gamma, delta, Epsilon)

        if trap:
            # heat demand pinned near the MIDPOINT of the envelope slice for
            # most periods -> pointwise tracking pays the concave bump nearly
            # every step; occasional excursions keep the tank genuinely needed.
            if rng.random() < 0.85:
                frac_mid = rng.uniform(0.35, 0.65)
            else:
                frac_mid = rng.choice([rng.uniform(0.0, 0.1), rng.uniform(0.9, 1.0)])
            q = L + frac_mid * (U - L)
        else:
            # non-trap periods lean toward the envelope's own vertices (so
            # pointwise tracking is a genuinely reasonable idea here -- the
            # ladder must stay sane), with a minority interior/random mix
            # for variety and to keep the instance genuinely open-ended.
            mode = rng.random()
            if mode < 0.47:
                q = L + rng.uniform(0.0, 0.05) * (U - L)       # near-L pattern
            elif mode < 0.94:
                q = L + rng.uniform(0.95, 1.0) * (U - L)       # near-U pattern
            elif mode < 0.97:
                q = L + rng.uniform(0.0, 1.0) * (U - L)        # uniform-random
            else:
                q = L + (0.5 + 0.5 * math.sin(2 * math.pi * frac * 3)) * (U - L)  # smooth wobble
        q = max(L, min(U, q))
        q = round(q, 3)
        DP.append(p)
        DQ.append(q)

    out = []
    out.append(f"{T} {Cap} {S_init}")
    out.append(f"{Pmin} {Pmax} {Qcap} {alpha} {beta} {Gamma} {delta} {Epsilon}")
    out.append(f"{b} {kappa} {dumpfee}")
    for t in range(T):
        out.append(f"{DP[t]} {DQ[t]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
