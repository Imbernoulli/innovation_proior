#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy multi-regime TRAIN sample to stdout.

Six (well, K=5 shown + 1 held out) "lab setups" (regimes) each independently
measure the SAME hidden physical law

    core(x) = sin(w*x + phi) + c*x

but every regime's instrument has its own unknown affine miscalibration
(gain_r, offset_r), and each regime can only see a NARROW window of x (its
apparatus's operating range).  Regimes never share an x window.

STDOUT prints ONLY the visible training rows:
    "K t"
    then for each regime: "r n lo hi" followed by n lines "x y"
No gain/offset/w/phi/c/seed is ever printed -- the hidden law and the held-out
regime live ONLY inside verify.py.
"""
import sys, random, math

AMP = 1.0  # fixed oscillation amplitude of the shared core (identifiability anchor)


def hidden_law(t):
    """Shared law params for this test id (mirrored in verify.py)."""
    rng = random.Random(9001 + t * 7919)
    w = rng.uniform(0.9, 1.7)
    phi = rng.uniform(0.0, 2 * math.pi)
    c = rng.uniform(0.05, 0.09)
    return w, phi, c


def true_core(x, w, phi, c):
    return math.sin(w * x + phi) + c * x


def train_regimes(t, K):
    """Per-regime window + nuisance (gen.py ONLY -- never needed by the checker)."""
    rng = random.Random(31337 + t * 104729)
    centers = [-8.0, -4.0, 0.0, 4.0, 8.0][:K]
    regimes = []
    for r, base in enumerate(centers):
        center = base + rng.uniform(-0.2, 0.2)
        halfwidth = rng.uniform(1.0, 1.4)
        gain = rng.uniform(0.5, 2.2)
        offset = rng.uniform(-4.0, 4.0)
        regimes.append((center, halfwidth, gain, offset))
    return regimes


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    K = 5
    n_per = 34 + (t % 5)
    sigma = 0.22 + 0.020 * (t - 1)
    w, phi, c = hidden_law(t)
    regimes = train_regimes(t, K)

    lines = ["%d %d" % (K, t)]
    for r, (center, hw, gain, offset) in enumerate(regimes):
        lo, hi = center - hw, center + hw
        rng = random.Random(555001 + t * 977 + r * 131)
        xs = sorted(rng.uniform(lo, hi) for _ in range(n_per))
        lines.append("%d %d %.6f %.6f" % (r, n_per, lo, hi))
        for x in xs:
            y = gain * true_core(x, w, phi, c) + offset + rng.gauss(0.0, sigma)
            lines.append("%.6f %.6f" % (x, y))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
