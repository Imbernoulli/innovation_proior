import sys, math

# Feedback-delay-network (FDN) reverb design instance generator.
# Fixed architecture: N delay lines, feedback matrix M = I - (2/N) * ones(N,N)
# (a Householder reflection of the all-ones vector -- symmetric, orthogonal).
# Free design variables (the solver's output): N integer delay lengths and N gains.
# The target is TWO curves sampled at K checkpoints: an energy-decay curve (dB) built
# as a mix of a fast and a slow exponential mode, and an echo-density growth curve
# that saturates at Dmax. testId selects N, T, the allowed delay-length range, and the
# two curves' shape parameters -- everything the solver needs is printed; the internal
# (f, tau_fast, tau_slow, theta) shape parameters are NOT printed, only the resulting
# K-point target arrays are (so the checker can score directly off the printed numbers,
# no hidden re-derivation needed).

# idx: N, T, Lmin, Lmax, greedy-step-parity-hint(unused by checker, only shapes the
# instance so a naive evenly-stepped delay choice collides on >=5/10 tests), theta-frac
LADDER = [
    # N,  T,    Lmin, Lmax, theta_frac
    (4, 1500, 20, 80, 0.06),   # trap: even Lmin (greedy's fixed step=2 -> gcd=2)
    (4, 1800, 21, 85, 0.18),
    (5, 2100, 30, 100, 0.06),  # trap: even Lmin (greedy's fixed step=2 -> gcd=2)
    (5, 2400, 31, 105, 0.18),
    (6, 2700, 24, 120, 0.06),  # trap: even Lmin (greedy's fixed step=2 -> gcd=2)
    (6, 3000, 25, 125, 0.18),
    (7, 3300, 36, 150, 0.06),  # trap: even Lmin (greedy's fixed step=2 -> gcd=2)
    (7, 3600, 37, 155, 0.18),
    (8, 4000, 40, 180, 0.06),  # trap: even Lmin (greedy's fixed step=2 -> gcd=2)
    (8, 4500, 41, 190, 0.18),
]

FRACS = [0.06, 0.15, 0.35, 0.65, 1.00]
F_MIX = 0.5
TAUFAST_FRAC = 0.05
TAUSLOW_FRAC = 0.50
DMAX = 0.30
W_DECAY = 0.05
W_DENSITY = 1.0


def main():
    i = int(sys.argv[1])
    idx = min(max(i, 1), len(LADDER)) - 1
    N, T, Lmin, Lmax, theta_frac = LADDER[idx]

    tau_fast = TAUFAST_FRAC * T
    tau_slow = TAUSLOW_FRAC * T
    theta = theta_frac * T

    K = len(FRACS)
    ts = []
    prev = 0
    for frac in FRACS:
        t = round(frac * T)
        t = max(t, prev + 1)
        t = min(t, T)
        ts.append(t)
        prev = t
    ts[-1] = T

    # Decay target is expressed as segment-midpoint envelope level, RELATIVE to the
    # first segment's own level (both the target and the checker's measured curve are
    # normalized this way -- see statement -- so an arbitrary overall output-amplitude
    # scale never matters, only the decay SHAPE does).
    raw_db = []
    target_density = []
    prev = 0
    for t in ts:
        mid = 0.5 * (prev + t)
        e = F_MIX * math.exp(-mid / tau_fast) + (1.0 - F_MIX) * math.exp(-mid / tau_slow)
        raw_db.append(10.0 * math.log10(max(e, 1e-300)))
        dens = DMAX * (1.0 - math.exp(-t / theta))
        target_density.append(dens)
        prev = t
    ref = raw_db[0]
    target_db = [v - ref for v in raw_db]

    out = []
    out.append("%d %d %d %d" % (N, T, Lmin, Lmax))
    out.append(str(K))
    out.append(" ".join(str(t) for t in ts))
    out.append(" ".join("%.8f" % v for v in target_db))
    out.append(" ".join("%.8f" % v for v in target_density))
    out.append("%.6f %.6f" % (W_DECAY, W_DENSITY))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
