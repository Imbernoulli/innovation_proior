#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

A product A is being adopted into a market whose TOTAL organic ceiling M0 is
finite but unknown to the solver. Adoption follows the discrete logistic
("Bass-style" diffusion) recursion

    A(t+1) = A(t) + k * A(t) * (1 - A(t) / cap(t))

where `cap(t)` is the CURRENT effective market ceiling. Early on, A(t) is a
small fraction of M0, so the recursion is numerically almost indistinguishable
from pure exponential growth k*A(t) -- the saturation term is invisible in a
short early window even though it is always mathematically present.

At a PUBLICLY ANNOUNCED future step t_B (already known -- competitors must
file launch dates with the exchange), a substitute product B goes on sale and
starts drawing from the SAME addressable market. From t_B onward the ceiling
for A drops:

    cap(t) = M0                              for t <  t_B
    cap(t) = A(t_B) + (1-s) * (M0 - A(t_B))   for t >= t_B

`s` in (0,1) is the fraction of A's remaining headroom that migrates to B
(the cannibalization severity). `s` and `M0` are never observable from the
pre-launch training window, so the solver is given NOISY analyst hints for
both instead of the exact values -- exactly the kind of imperfect market
intelligence a forecaster actually has.

Each testId fixes a DIFFERENT hidden (A1, k, M0, t_B, s). The solver only
ever sees the first N_TRAIN steps (all strictly before t_B). The held-out
grading window (regenerated only inside the checker) reaches past t_B, where
the ceiling drop actually bites. STDOUT here never prints the hidden law,
the seed, or any true parameter -- only the noisy training rows and the
noisy analyst hints.
"""
import sys, math, random

N_TRAIN = 13
H = 11
NOISE_SIGMA = 0.015     # small multiplicative log-noise on training counts

# Fixed per-testId scenario plan: a MIX of "mild" scenarios (late launch,
# gentle growth -- the visible curvature of the true logistic is faint) and
# "sharp" scenarios (early launch, fast growth, high severity -- the true
# curve is already bending hard by the time the substitute launches). Both
# kinds are needed: mild cases keep a naive exponential fit looking
# defensible; sharp cases are where it blows up worst.
PLAN = {
    1: dict(k=0.22, M0=7200.0, tB_off=9, s=0.40),
    2: dict(k=0.24, M0=6600.0, tB_off=8, s=0.44),
    3: dict(k=0.40, M0=5200.0, tB_off=3, s=0.72),
    4: dict(k=0.26, M0=7800.0, tB_off=9, s=0.36),
    5: dict(k=0.42, M0=4800.0, tB_off=3, s=0.75),
    6: dict(k=0.24, M0=8200.0, tB_off=8, s=0.48),
    7: dict(k=0.25, M0=6000.0, tB_off=7, s=0.40),
    8: dict(k=0.38, M0=5500.0, tB_off=4, s=0.67),
    9: dict(k=0.20, M0=7000.0, tB_off=9, s=0.40),
    10: dict(k=0.26, M0=6800.0, tB_off=7, s=0.49),
}


def hidden_law(t):
    """Hidden scenario for this test id. Lives in gen AND checker; never printed."""
    rng = random.Random(119600 + t * 7919)
    plan = PLAN[t]
    A1 = rng.uniform(90.0, 140.0)
    k = plan['k'] + rng.uniform(-0.01, 0.01)
    M0 = plan['M0'] + rng.uniform(-150.0, 150.0)
    tB = N_TRAIN + plan['tB_off']
    s_true = plan['s'] + rng.uniform(-0.02, 0.02)
    s_hint = s_true * (1.0 + rng.uniform(-0.15, 0.15)) + rng.uniform(-0.03, 0.03)
    s_hint = max(0.05, min(0.95, s_hint))
    M_hint = M0 * (1.0 + rng.uniform(-0.12, 0.12))
    return A1, k, M0, tB, s_true, s_hint, M_hint


def simulate(A1, k, M0, tB, s_true, T_end):
    """Exact discrete recursion, A(1)..A(T_end), no noise."""
    A = {1: A1}
    M1 = None
    for t in range(1, T_end):
        cur_cap = M0 if (t < tB) else M1
        nxt = A[t] + k * A[t] * (1.0 - A[t] / cur_cap)
        A[t + 1] = nxt
        if (t + 1) == tB:
            M1 = A[tB] + (1.0 - s_true) * (M0 - A[tB])
    return A


def train_rows(t):
    A1, k, M0, tB, s_true, s_hint, M_hint = hidden_law(t)
    A = simulate(A1, k, M0, tB, s_true, N_TRAIN)
    rng = random.Random(220300 + t * 13)
    rows = []
    for ti in range(1, N_TRAIN + 1):
        noisy = A[ti] * math.exp(rng.gauss(0.0, NOISE_SIGMA))
        rows.append((ti, noisy))
    return rows, tB, s_hint, M_hint


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows, tB, s_hint, M_hint = train_rows(t)
    out = ["%d %d" % (t, len(rows)), "%d %.8f %.8f" % (tB, s_hint, M_hint)]
    for ti, A in rows:
        out.append("%d %.8f" % (ti, A))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
