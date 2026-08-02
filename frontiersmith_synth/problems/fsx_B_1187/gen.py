#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE instance to stdout.

Family: radar-micro-doppler-id.  A dwell observes a rotating scatterer (drone
rotor, helicopter rotor, propeller, turbofan, wind turbine) at K different
aspect angles.  At aspect angle theta the micro-Doppler line sits at

    f_true(theta) = blade_count * rotation_rate * sin(theta)     [Hz]

(forward model; the sin(theta) factor is the aspect-angle dependence -- only
the radial component of blade-tip velocity Doppler-shifts).  The radar's PRF
can only unambiguously represent frequencies in [0, PRF/2]: anything above
that ALIASES, folding back into band.  gen.py bakes this fold in when it
prints the "observed" spectral peak at each angle, so a rotation rate whose
true line exceeds PRF/2 at some angle "masquerades" as a slower one there.

The printed instance never reveals the true class or rate -- only the PRF,
the K (angle, observed folded frequency) pairs, and a small table of
candidate target classes (name, blade/scatterer count, plausible rotation
-rate range).  Everything is a fixed function of testId (a hand-placed
difficulty/trap ladder) -- fully deterministic, no RNG needed.
"""
import sys, math

PRF = 100.0
DF = PRF / 256.0  # simulated FFT bin resolution

# Fixed taxonomy of candidate classes: (name, blade/scatterer count, rate_min, rate_max) in RPS.
CLASSES = [
    ("drone",       2, 3.0, 7.0),
    ("helicopter",  4, 5.0, 9.0),
    ("propeller",   3, 15.0, 35.0),
    ("turbofan",    8, 18.0, 40.0),
    ("windturbine", 3, 0.1, 1.0),
]

# testId -> (true_class_index, true_rotation_rate_RPS, [aspect angles in degrees])
# >=3 of the 10 cases are TRAP cases: the angle with the largest sin(theta) (the
# most sensitive look, the one an obvious "trust the raw peak" solver reaches for)
# has ITS true line above PRF/2, so it aliases; the aliased value happens to match
# a completely different candidate class's plausible rate at that one angle.  The
# remaining angles are kept below the aliasing threshold at the true rate, so a
# solver that reconciles ALL angles (not just the loudest one) can unfold it.
CASES = {
    1:  (0, 4.2,  [15, 35, 55, 78]),
    2:  (4, 0.5,  [20, 40, 60, 80]),
    3:  (2, 28.0, [10, 20, 32, 45, 65]),
    4:  (1, 6.0,  [15, 30, 50, 60, 70]),
    5:  (3, 30.0, [3, 6, 9, 11.5, 16.9]),
    6:  (0, 6.5,  [12, 28, 45, 58, 68, 72]),
    7:  (2, 32.0, [8, 16, 24, 31, 40, 49.3]),
    8:  (3, 35.0, [2, 4, 6, 8, 9.5, 66]),
    9:  (1, 8.5,  [10, 22, 34, 48, 60, 66, 80]),
    10: (2, 29.0, [5, 10, 15, 20, 25, 30, 54.4]),
}


def fold(f, prf=PRF):
    """Canonical PRF fold: true frequency f -> observable in [0, prf/2]."""
    k = math.floor(f / prf + 0.5)
    return abs(f - k * prf)


def quantize(f, df=DF):
    return round(round(f / df) * df, 6)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t not in CASES:
        t = ((t - 1) % len(CASES)) + 1
    true_cid, true_rate, angles = CASES[t]
    name, blade, lo, hi = CLASSES[true_cid]
    assert lo <= true_rate <= hi

    obs = []
    for th in angles:
        s = math.sin(math.radians(th))
        f_true = blade * true_rate * s
        obs.append(quantize(fold(f_true)))

    # Shuffle the printed class order per test (a fixed rotation) so the
    # answer is never "always class 0" / "always the first row" -- solvers
    # must match candidates by their stated blade count and rate range.
    shift = (2 * t + 1) % len(CLASSES)
    order = [(shift + i) % len(CLASSES) for i in range(len(CLASSES))]
    printed = [CLASSES[j] for j in order]

    K = len(angles)
    C = len(CLASSES)
    out = []
    out.append("%.6f %d %d" % (PRF, K, C))
    out.append(" ".join("%.4f" % a for a in angles))
    out.append(" ".join("%.6f" % o for o in obs))
    for nm, b, rlo, rhi in printed:
        out.append("%s %d %.6f %.6f" % (nm, b, rlo, rhi))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
