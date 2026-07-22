#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE pressure-vessel telemetry log to stdout.

A sealed vessel leaks obeying dL/dt = -c * L^alpha for hidden positive
constants c and alpha (alpha roughly in [1.2, 1.8]). Most days it is also
topped off from a tanker truck. The facility log records only the day's
TOTAL delivered volume Q_d (which may be the sum of one or two unlogged
deliveries that day, at unlogged times) -- never individual delivery times,
counts, or amounts, and the vessel's level sensor is read once per day (end
of day) with small measurement noise.

Each testId fixes a DIFFERENT hidden (c, alpha) and a different telemetry
run, with a GUARANTEED minimum fraction of delivery days (so no instance is
accidentally "easy" for a naive fit). All telemetry stays in a LOW-level
regime; the grader's held-out check (never printed here) lives in a
genuinely higher-level, extrapolated regime, regenerated deterministically
inside the checker from the SAME testId.

STDOUT prints ONLY:
  line 1:  "<D> <t>"           D = number of logged days, t = test id
  line 2:  "<L0>"              level reading at day 0 (before any logged day)
  lines 3..D+2:  "<Q_d> <E_d>" total delivered volume during day d, level
                                reading at the END of day d  (d = 1..D)

The hidden (c, alpha), the per-day delivery TIMING/COUNT, and the RNG seed
are NEVER printed -- only the noisy telemetry rows.
"""
import sys, math, random

SEED_BASE = 900001
T_DAY = 1.0

C_LO, C_HI = 0.006, 0.020
A_LO, A_HI = 1.2, 1.8
L0_LO, L0_HI = 8.0, 18.0
Q_LO, Q_HI = 0.05, 0.30
SIGMA_E = 0.03           # sensor measurement noise (level units)

CP_BASE, CP_SLOPE = 0.72, 0.02   # per-day delivery probability, grows with t
MIN_CHARGE_FRAC = 0.68           # guaranteed minimum fraction of delivery days
SPLIT_PROB = 0.20                # probability a delivery day's total is
                                  # actually two unlogged sub-deliveries


def base_rng(t):
    return random.Random(SEED_BASE + t * 7919)


def hidden_params(t):
    """First two draws of the per-t stream: the hidden leak law (c, alpha).
    verify.py reproduces EXACTLY these two draws (same rng, same order) to
    recover the hidden law without needing anything else gen.py drew."""
    rng = base_rng(t)
    c = rng.uniform(C_LO, C_HI)
    alpha = rng.uniform(A_LO, A_HI)
    return c, alpha, rng


def days_for(t):
    return 10 + t


def charge_prob(t):
    return CP_BASE + CP_SLOPE * t


def decay_step(L, dt, c, alpha):
    """Closed-form solution of dL/dt = -c*L^alpha for a duration dt with no
    delivery events in between (exact, no ODE integrator needed)."""
    if dt <= 0:
        return L
    base = L ** (1 - alpha) + c * (alpha - 1) * dt
    if base < 1e-12:
        base = 1e-12
    return base ** (1.0 / (1 - alpha))


def simulate_day_events(Estart, events, dt_day, c, alpha):
    """events: chronologically sorted list of (time_fraction in (0,1),
    amount>0). Decays between events, applies each jump instantly. Works
    for zero, one, or two events -- the mass-balance identity used by the
    strong reference does not care how the day's total is split."""
    if not events:
        return decay_step(Estart, dt_day, c, alpha)
    L = Estart
    t_prev = 0.0
    for tf, amt in events:
        t_abs = tf * dt_day
        L = decay_step(L, t_abs - t_prev, c, alpha)
        L = L + amt
        t_prev = t_abs
    return decay_step(L, dt_day - t_prev, c, alpha)


def gen_instance(t):
    c, alpha, rng = hidden_params(t)
    D = days_for(t)
    L0_clean = rng.uniform(L0_LO, L0_HI)
    L0 = L0_clean + rng.gauss(0.0, SIGMA_E)
    cp = charge_prob(t)

    has_charge = [rng.random() < cp for _ in range(D)]
    min_required = max(3, int(math.ceil(D * MIN_CHARGE_FRAC)))
    need = min_required - sum(has_charge)
    for d in range(D):
        if need <= 0:
            break
        if not has_charge[d]:
            has_charge[d] = True
            need -= 1

    day_events = []
    for d in range(D):
        if not has_charge[d]:
            day_events.append([])
            continue
        Q = rng.uniform(Q_LO, Q_HI)
        if rng.random() < SPLIT_PROB:
            f = rng.uniform(0.2, 0.8)
            u1 = rng.uniform(0.05, 0.45)
            u2 = rng.uniform(0.55, 0.95)
            day_events.append([(u1, Q * f), (u2, Q * (1.0 - f))])
        else:
            u = rng.uniform(0.1, 0.9)
            day_events.append([(u, Q)])

    rows = []
    E_prev_clean = L0_clean
    for d in range(D):
        events = day_events[d]
        Qd = sum(amt for _, amt in events)
        Eend_clean = simulate_day_events(E_prev_clean, events, T_DAY, c, alpha)
        Eend_noisy = max(0.05, Eend_clean + rng.gauss(0.0, SIGMA_E))
        rows.append((Qd, Eend_noisy))
        E_prev_clean = Eend_clean
    return D, L0, rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    D, L0, rows = gen_instance(t)
    out = ["%d %d" % (D, t), "%.6f" % L0]
    for Q, E in rows:
        out.append("%.6f %.6f" % (Q, E))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
