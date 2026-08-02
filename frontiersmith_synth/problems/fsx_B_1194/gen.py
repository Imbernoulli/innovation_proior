#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy subcritical wake-decay logbook to stdout.

Hidden physics (a supercritical Hopf bifurcation -- vortex shedding onset):
    g(R)  = a * (R - Rc)          linear growth rate of the least-stable mode
                                   near the critical flow parameter Rc (a>0)
    R < Rc  (steady flow): an injected perturbation DECAYS at rate d(R) = -g(R) > 0.
                            The oscillation amplitude itself is (noisy) ZERO.
    R > Rc  (shedding):    the mode grows and saturates on a limit cycle whose
                            equilibrium amplitude obeys the Landau law
                                A(R) = sqrt( g(R) / L ),   L>0 a system constant.

The TRAIN logbook the solver SEES samples R ONLY in the steady (subcritical)
branch, so every measured amplitude in training is noise around zero -- no
training row ever shows shedding. Each row also carries the measured DECAY
RATE of a small artificial perturbation injected at that R; that column is the
only place the approach-to-onset (and hence the post-onset growth rate law)
leaves a signature. The Landau constant L is a fixed, given system property
(second line of the input) -- it is NOT to be fitted, only used correctly.

STDOUT prints ONLY: "<M> <test_id>", then "<L>", then M rows
"<R> <decay_rate> <amplitude>". Rc, a, L's provenance, and all seeds are never
explained -- only the numbers a solver would actually measure in a lab.
"""
import sys, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
RC_LO, RC_HI = 40.0, 140.0
A_LO, A_HI = 0.010, 0.022
L_LO, L_HI = 0.06, 0.15
OFF_LO, OFF_HI_CAP = 4.0, 190.0


def params(t):
    """Hidden bifurcation law for this test id (identical in gen.py and verify.py)."""
    rng = random.Random(8100000 + t * 3141593)
    Rc = rng.uniform(RC_LO, RC_HI)
    a = rng.uniform(A_LO, A_HI)
    L = rng.uniform(L_LO, L_HI)
    return Rc, a, L


def noise_floor(t):
    return 0.05 + (0.10 / 9.0) * (t - 1)


def noise_rel(t):
    return 0.10 + (0.18 / 9.0) * (t - 1)


def train_size(t):
    return max(8, round(22 - (8.0 / 9.0) * (t - 1)))


def gen_train(t):
    Rc, a, L = params(t)
    M = train_size(t)
    sf = noise_floor(t)
    sr = noise_rel(t)
    rng = random.Random(55019 + t * 977)
    rows = []
    max_off = max(OFF_LO + 1.0, 0.9 * Rc)
    off_hi = min(OFF_HI_CAP, max_off)
    for _ in range(M):
        off = rng.uniform(OFF_LO, off_hi)
        R = Rc - off
        true_d = a * off
        sigma = sf + sr * true_d
        decay = true_d + rng.gauss(0.0, sigma)
        amp = abs(rng.gauss(0.0, 0.01 + 0.002 * (t - 1)))
        rows.append((R, decay, amp))
    return rows, L


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows, L = gen_train(t)
    out = ["%d %d" % (len(rows), t), "%.8g" % L]
    for R, decay, amp in rows:
        out.append("%.6f %.6f %.6f" % (R, decay, amp))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
