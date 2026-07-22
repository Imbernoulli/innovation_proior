#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance of "chord-of-masses" to stdout.

Physical setup (see statement.md): n beads on a taut, massless, fixed-tension
string with fixed (Dirichlet) ends, spaced evenly.  Bead i carries mass
m_i = 1 + e_i, where e_i >= 0 is an integer number of "mass units" the
solver assigns (e_i <= CAP, sum(e_i) == BUDGET).  The transverse small
oscillation frequencies omega_1 < omega_2 < ... < omega_n solve the
generalized eigenproblem K v = lambda M v, where K is the standard
tridiagonal(-1,2,-1) discrete-Laplacian stiffness matrix and M=diag(m_i).
omega_k = sqrt(lambda_k).

Instance format (stdout):
  line 1: n B CAP r
  line 2: r pairs "p_k q_k" -- the TARGET ratio omega_k/omega_1 = p_k/q_k
          for k=1..r.  Always p_1=q_1=1 (mode 1 is the reference / root).

Everything is a deterministic function of testId only (the "difficulty
ladder"); no external randomness is used anywhere.
"""
import sys
import math
from fractions import Fraction


def natural_log_ratios(n, r):
    """log(omega_k^uniform / omega_1^uniform) for the UNIFORM-mass string,
    k=1..r.  Closed form: omega_k^uniform proportional to sin(k*pi/(2(n+1)))."""
    out = []
    denom = math.sin(math.pi / (2.0 * (n + 1)))
    for k in range(1, r + 1):
        out.append(math.log(math.sin(k * math.pi / (2.0 * (n + 1))) / denom))
    return out


# Difficulty ladder.  Each row: (n, r, B, CAP, shifts)
#   shifts[j] (j=0..r-2) is the desired log-ratio SHIFT away from the
#   natural (uniform-mass) spectrum for mode k=j+2, i.e. the target is
#   T_k = N_k * exp(shifts[j]) where N_k is the natural ratio omega_k/omega_1.
#   Negative shifts compress the spectrum (the "chord" is voiced tighter
#   than the string's natural near-harmonic overtones -- this is the hard,
#   physically meaningful direction: more mass always slows every mode
#   down, so bringing a high overtone's RELATIVE pitch down means loading
#   it more than the fundamental, which only works if you load the right
#   SHAPE, not just "more mass somewhere").
# Rows marked TRAP have two or more shifts of similar size on adjacent
# modes k, k+1: their antinode positions are close together, so a solver
# that treats each target mode independently (adjust the antinode of mode
# k, then the antinode of mode k+1, ...) fights itself.
TABLE = [
    # id 1
    dict(n=16, r=3, B=40, CAP=5, shifts=[-0.22, -0.30]),
    # id 2
    dict(n=20, r=3, B=55, CAP=5, shifts=[-0.18, -0.28]),
    # id 3  TRAP: shifts close together on adjacent modes 2,3
    dict(n=18, r=3, B=45, CAP=4, shifts=[-0.20, -0.22]),
    # id 4
    dict(n=24, r=4, B=90, CAP=7, shifts=[-0.18, -0.24, -0.30]),
    # id 5  TRAP: shifts 2,3 close together
    dict(n=28, r=4, B=110, CAP=8, shifts=[-0.22, -0.20, -0.34]),
    # id 6
    dict(n=32, r=4, B=130, CAP=8, shifts=[-0.16, -0.28, -0.22]),
    # id 7
    dict(n=36, r=5, B=170, CAP=9, shifts=[-0.15, -0.20, -0.25, -0.30]),
    # id 8  TRAP: shifts 2,3 identical (adjacent close modes)
    dict(n=42, r=5, B=200, CAP=9, shifts=[-0.18, -0.18, -0.30, -0.24]),
    # id 9
    dict(n=50, r=5, B=240, CAP=9, shifts=[-0.20, -0.26, -0.22, -0.32]),
    # id 10 TRAP: tight relative budget forces efficient (non-wasteful) allocation
    dict(n=60, r=5, B=200, CAP=8, shifts=[-0.15, -0.22, -0.28, -0.34]),
]


def build(test_id):
    row = TABLE[(test_id - 1) % len(TABLE)]
    n, r, B, CAP = row["n"], row["r"], row["B"], row["CAP"]
    shifts = row["shifts"]
    L0 = natural_log_ratios(n, r)
    targets = [1.0]
    for j, s in enumerate(shifts):
        k = j + 2
        targets.append(math.exp(L0[k - 1] + s))
    fracs = [Fraction(1, 1)]
    for t in targets[1:]:
        fracs.append(Fraction(t).limit_denominator(360))
    return n, B, CAP, r, fracs


def main():
    test_id = int(sys.argv[1])
    n, B, CAP, r, fracs = build(test_id)
    print(n, B, CAP, r)
    print(" ".join(f"{f.numerator} {f.denominator}" for f in fracs))


if __name__ == "__main__":
    main()
