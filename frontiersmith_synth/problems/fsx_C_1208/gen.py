#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE recsys popularity-forecast instance to stdout.

Setting (hidden from the solver -- only the SHAPE is real, not the numbers):
An item has a genuine ORGANIC interest trajectory  o(t) = O0 + O1*t  (a slow,
honest drift -- it may be truly catching on, or not).  A recommender ALSO
decides how much exposure x(t) to give the item each period, and its policy
REACTS to the item's own recent engagement: whenever the item outperforms its
own baseline, the recommender pushes it harder next period (a classic
exposure-feedback loop / popularity-bias amplifier). Measured engagement is

    e(t) = o(t) + ALPHA * x(t) + noise(t)

i.e. organic interest plus an INDUCED contribution from exposure itself.
Because x(t) is generated from the item's own past e(t) (which is itself
trending with t through o(t)), x(t) drifts upward together with t across the
logged period -- x and t are confounded in exactly the window the solver
gets to see. The hidden shape parameters (O0, O1, ALPHA, the feedback gain,
the exogenous-shock gain) are NEVER printed -- only the (t, x, e) log rows.

STDOUT prints, all whitespace-separated, in order:
  t n_train
  t_1 x_1 e_1
  t_2 x_2 e_2
  ...
  t_{n_train} x_{n_train} e_{n_train}

`t` (first token) is the test id (informational only). `n_train` is the
number of logged periods. Each row is an integer period index, the fraction
of impression slots the recommender gave the item that period (x, in
[0,1]), and the item's measured engagement rate that period (e, >= 0).
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
T_TRAIN = 30
X0 = 0.15
XMIN, XMAX = 0.0, 1.0
O0_LO, O0_HI = 0.5, 3.0
O1_LO, O1_HI = 0.010, 0.045
ALPHA_LO, ALPHA_HI = 1.5, 3.6
BETA_LO, BETA_HI = 0.4, 0.85
GAMMA_LO, GAMMA_HI = 0.15, 0.35
NORM = 4.5
NOISE_SIGMA_E = 0.19


def _gauss(rng):
    """Deterministic standard-normal draw via Box-Muller (no numpy)."""
    u1 = max(1e-12, rng.random())
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def params(t):
    """Hidden instance parameters (identical in gen.py and verify.py)."""
    rng = random.Random(4110001 + t * 700111)
    O0 = rng.uniform(O0_LO, O0_HI)          # HIDDEN organic baseline
    O1 = rng.uniform(O1_LO, O1_HI)          # HIDDEN organic per-period drift
    ALPHA = rng.uniform(ALPHA_LO, ALPHA_HI)  # HIDDEN induced-per-exposure lift
    BETA = rng.uniform(BETA_LO, BETA_HI)    # HIDDEN feedback gain (popularity bias)
    GAMMA = rng.uniform(GAMMA_LO, GAMMA_HI)  # HIDDEN exogenous-shock gain on exposure
    return O0, O1, ALPHA, BETA, GAMMA


def gen_train(t, O0, O1, ALPHA, BETA, GAMMA):
    """T_TRAIN logged (period, exposure, engagement) rows under the ACTIVE
    feedback loop: this period's exposure reacts to last period's engagement
    excess over baseline, so x(t) and t are confounded across this window."""
    rng = random.Random(2290007 + t * 60013)
    rows = []
    e_prev = O0
    for tt in range(1, T_TRAIN + 1):
        z = rng.uniform(-1.0, 1.0)
        if tt == 1:
            x = X0 + GAMMA * z
        else:
            x = X0 + BETA * max(0.0, e_prev - O0) / NORM + GAMMA * z
        x = min(XMAX, max(XMIN, x))
        noise = NOISE_SIGMA_E * _gauss(rng)
        e = max(0.0, O0 + O1 * tt + ALPHA * x + noise)
        rows.append((tt, x, e))
        e_prev = e
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    O0, O1, ALPHA, BETA, GAMMA = params(t)
    train = gen_train(t, O0, O1, ALPHA, BETA, GAMMA)

    out = ["%d %d" % (t, T_TRAIN)]
    for tt, x, e in train:
        out.append("%d %.6f %.6f" % (tt, x, e))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
