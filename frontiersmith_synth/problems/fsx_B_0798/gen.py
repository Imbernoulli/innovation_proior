#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance of the "Fiber Reach Tuning" problem to stdout.

Instance shape (deterministic in testId only, seeded via random.Random):
  N candidate node positions along a straight optical line (x_0=0 is the
  transmitter; x_1..x_{N-1} are downstream nodes, strictly increasing).
  Physical-layer constants: alpha (fiber loss, dB/km), c0 (fixed per-amplifier
  noise floor), c_ase (linear ASE-noise rate per km), c_kerr (cubic Kerr
  nonlinear-noise coefficient). An SNR threshold. A list of K allowed
  discrete launch-power levels.

  A handful of testIds plant one or two abnormally long gaps between
  consecutive candidate positions ("trap spans"): any span that must
  absorb such a gap punishes a uniformly-chosen high power far more than
  the many short spans do, because the Kerr penalty grows with span
  length times power-cubed.

Output (stdout):
  N
  x_0 x_1 ... x_{N-1}
  alpha c0 c_ase c_kerr
  thresh
  K
  p_1 p_2 ... p_K
"""
import sys
import random


def scale_for(t):
    # (N, base_step_lo, base_step_hi, trap) ladder: small/sane -> larger/adversarial.
    # trap=True plants 1-2 abnormally long gaps to punish uniform max-power greedy.
    table = {
        1:  (6,  14, 22, False),
        2:  (7,  14, 24, False),
        3:  (8,  13, 23, False),
        4:  (9,  12, 22, True),
        5:  (10, 12, 22, False),
        6:  (11, 11, 21, True),
        7:  (12, 11, 21, False),
        8:  (13, 10, 20, True),
        9:  (14, 10, 20, True),
        10: (14, 10, 20, True),
    }
    return table[t]


def build(t):
    N, lo, hi, trap = scale_for(t)
    rnd = random.Random(1798000 + 131 * t)

    # --- node positions: N-1 steps of irregular length, x_0 = 0 ---------
    steps = [rnd.randint(lo, hi) for _ in range(N - 1)]
    if trap:
        n_traps = 1 if N <= 10 else 2
        trap_positions = rnd.sample(range(N - 1), n_traps)
        for tp in trap_positions:
            mult = rnd.randint(7, 11)
            steps[tp] = steps[tp] * mult
    xs = [0]
    for s in steps:
        xs.append(xs[-1] + s)

    # --- physical constants: fixed across tests (calibrated so the
    #     ladder gives real headroom between tiers -- see problems/.../NOTES) --
    alpha = 0.22            # dB/km fiber loss
    c0 = 0.08                # fixed per-amp noise floor
    c_ase = 0.0004           # linear ASE rate per km
    c_kerr = 0.000003        # cubic Kerr coefficient
    thresh = 2.0             # SNR threshold to "reach" a node

    # --- allowed discrete launch-power levels ---------------------------
    K = 6
    p_min = rnd.randint(2, 3)
    p_max = rnd.randint(16, 20)
    raw = [p_min + round((p_max - p_min) * i / (K - 1)) for i in range(K)]
    seen = []
    for v in raw:
        if not seen or v > seen[-1]:
            seen.append(v)
    while len(seen) < K:
        seen.append(seen[-1] + 1)
    powers = seen[:K]

    return N, xs, alpha, c0, c_ase, c_kerr, thresh, powers


def main():
    t = int(sys.argv[1])
    N, xs, alpha, c0, c_ase, c_kerr, thresh, powers = build(t)
    out = []
    out.append(str(N))
    out.append(" ".join(str(v) for v in xs))
    out.append("%.4f %.5f %.6f %.8f" % (alpha, c0, c_ase, c_kerr))
    out.append("%.4f" % thresh)
    out.append(str(len(powers)))
    out.append(" ".join(str(v) for v in powers))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
