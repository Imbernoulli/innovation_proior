#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy SUB-CRITICAL percolation census to stdout.

Family: percolation-scaling-extrapolation.  A hidden finite random-graph ensemble
undergoes a bond-percolation phase transition: as the bond-occupation probability
p rises past a HIDDEN critical threshold p_c, the relative size of the giant
component S(p) obeys the near-critical scaling law

    S(p)  ~  A * (p - p_c)^beta        for p somewhat above p_c

with a HIDDEN critical exponent beta and amplitude A.  Below p_c, finite-size
rounding (governed by a KNOWN crossover width W, printed in the header -- think
of it as a calibrated property of your apparatus/graph family) lets S(p) stay
small but nonzero, rising smoothly toward p_c.  Both regimes are two asymptotics
of ONE smooth scaling function:

    z(p)   = (p - p_c) / W
    g(z)   = 0.5 * ( z + sqrt(z*z + 4) )          # smooth "hinge" kernel, g(z)>0
    S(p)   = clip( A * (W * g(z))^beta , 0, 1 )

As z -> +inf,  W*g(z) -> (p-p_c), so S -> A*(p-p_c)^beta  (the stated scaling law).
As z -> -inf,  S decays smoothly toward 0 (finite-size rounding below threshold).

Each testId fixes a DIFFERENT hidden ecosystem (p_c, beta, A all hidden; W known).
The solver ONLY ever sees a SUB-CRITICAL census: p values that stay strictly BELOW
p_c (from deep sub-critical up to just below the transition), with small
measurement noise.  The HELD-OUT grading points sit AT and ABOVE the transition
(genuine extrapolation into a region never shown here) and are regenerated only
inside the checker -- never printed by this file.

STDOUT prints ONLY: a header "<testId> <N> <W>" then N lines "p S_meas".  The
hidden p_c, beta, A and the RNG seeds are NEVER printed.
"""
import sys, random, math

PC_BASE = [0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60, 0.65, 0.70]
BETA_BASE = [1.00, 1.20, 0.90, 1.40, 1.10, 1.60, 1.30, 0.85, 1.50, 1.05]
W_BASE = [0.015, 0.012, 0.020, 0.010, 0.018, 0.008, 0.016, 0.022, 0.009, 0.014]

N_TRAIN = 220
ZLO_TRAIN = -9.0     # deep sub-critical edge of the training band (in z units)
ZHI_TRAIN = -1.2     # near-critical edge of the training band (still z<0, i.e. p<p_c)
TRAIN_ABS_FLOOR = 0.0006
TRAIN_REL = 0.03


def hidden_params(t):
    """Hidden ecosystem for this test id.  Lives in gen AND verify; never printed."""
    rng = random.Random(870000 + t * 7919)
    pc = PC_BASE[t - 1] + rng.uniform(-0.01, 0.01)
    beta = BETA_BASE[t - 1] + rng.uniform(-0.05, 0.05)
    w = W_BASE[t - 1] * rng.uniform(0.9, 1.1)
    far_target = rng.uniform(0.55, 0.80)   # S(p=0.95) roughly this, for headroom
    p_far = 0.95
    amp = far_target / ((p_far - pc) ** beta)
    return pc, beta, w, amp


def gfun(z):
    return 0.5 * (z + math.sqrt(z * z + 4.0))


def S_true(p, pc, beta, w, amp):
    z = (p - pc) / w
    val = amp * (w * gfun(z)) ** beta
    return max(0.0, min(1.0, val))


def train_census(t, pc, beta, w, amp):
    lo = max(0.01, pc + ZLO_TRAIN * w)
    hi = pc + ZHI_TRAIN * w
    rng = random.Random(1000 + t)
    rows = []
    for i in range(N_TRAIN):
        p = lo + (hi - lo) * i / (N_TRAIN - 1)
        strue = S_true(p, pc, beta, w, amp)
        sig = max(TRAIN_ABS_FLOOR, TRAIN_REL * abs(strue))
        s = strue + rng.gauss(0.0, sig)
        rows.append((p, s))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    pc, beta, w, amp = hidden_params(t)
    rows = train_census(t, pc, beta, w, amp)
    out = ["%d %d %.6f" % (t, len(rows), w)]
    for p, s in rows:
        out.append("%.6f %.6f" % (p, s))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
