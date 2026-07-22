#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE staged-rocket instance to stdout.

Modular-booster staging: a launch vehicle is built from S drop-away stages;
stage 1 (bottom) burns and is jettisoned first, up to stage S nearest the
payload.  Every stage is assembled from identical engine modules (fixed dry
mass, thrust, exhaust velocity) plus a propellant tank whose structural mass is
a fixed fraction kappa of the propellant it holds.  The solver chooses, per
stage, an INTEGER engine count n_i and a real propellant mass p_i.  The grader
integrates the ascent stage-by-stage (Tsiolkovsky gain minus a stage-indexed
gravity-drag loss that grows with burn duration) and reports payload velocity.

Difficulty / trap ladder (testId 1..10):
  * tests 1..3  -- near-uniform, moderate loss table.  The textbook geometric
    staging is close to optimal here.
  * tests 4..10 -- the loss coefficient is SPIKED on the bottom stage, exactly
    where loss-blind geometric staging piles the most propellant and lets it
    burn long behind a small engine cluster.  The loss-aware optimum shortens
    that burn (more engines / less propellant on the spiked stage) and moves
    delta-v up the stack -- so an even split or a loss-blind geometric split
    lands far from optimal.  Budget tightens and S grows with difficulty.

STDOUT format:
    line 1:  S P M_total kappa
    line 2:  m_e T v_e g E_max
    line 3:  L_1 L_2 ... L_S
"""
import sys, random

# stage-count ladder: a mix of 3- and 4-stage vehicles, growing with difficulty.
S_LADDER = [3, 3, 3, 4, 3, 4, 4, 4, 3, 4]


def params(t):
    """Return an instance dict deterministically from test id t (1..10)."""
    rng = random.Random(20574 + t * 6089)
    S = S_LADDER[(t - 1) % 10]

    P = 1000.0        # payload mass
    m_e = 100.0       # engine dry mass per module
    v_e = 3000.0      # effective exhaust velocity
    g = 9.8
    T = 10000.0       # thrust per engine module (moderate -> burns are long
                      # enough that the loss term bites unless you add engines)
    E_max = 12
    kappa = 0.12      # tank/structure mass as a fraction of loaded propellant

    # overall mass ratio grows 8.0 -> 13.4 across the ladder
    M_total = P * (8.0 + 0.6 * (t - 1))

    # --- loss table L_i (the altitude/stage coupling) ---
    if t <= 3:
        # easy: near-uniform, moderate loss -> geometric staging is close to best
        L = [round(rng.uniform(0.12, 0.17), 4) for _ in range(S)]
    else:
        # trap: low baseline loss everywhere except a hard SPIKE on the bottom
        # stage, where geometric staging wants the most propellant.
        L = [round(rng.uniform(0.06, 0.10), 4) for _ in range(S)]
        L[0] = round(rng.uniform(0.26, 0.34), 4)

    return dict(S=S, P=P, M_total=M_total, kappa=kappa, m_e=m_e, T=T, v_e=v_e,
                g=g, E_max=E_max, L=L)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    q = params(t)
    out = []
    out.append("%d %.6f %.6f %.6f" % (q["S"], q["P"], q["M_total"], q["kappa"]))
    out.append("%.6f %.6f %.6f %.6f %d" %
               (q["m_e"], q["T"], q["v_e"], q["g"], q["E_max"]))
    out.append(" ".join("%.4f" % x for x in q["L"]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
