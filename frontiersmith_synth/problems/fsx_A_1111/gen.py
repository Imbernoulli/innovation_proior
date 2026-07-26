#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Spearfishing through a layered lagoon: a spear-fisher's laser rangefinder
sends a ray down through a hidden stack of THREE flat, horizontal water
layers (a thermocline/salinity structure) of unknown thickness and
refractive index, down to a fixed sensor plane at depth D. Each testId
fixes a DIFFERENT hidden stack.

The rangefinder is only ever CALIBRATED at NEAR-VERTICAL aim (the diver
holds it almost straight down to avoid glare) -- so the training data are
all NEAR-NORMAL entry angles. The held-out grading probes the FULL range
of steep aim angles, including angles steep enough that a deep layer
totally internally reflects the ray (it never reaches the sensor at all).
That regime is entirely absent from the training data and is regenerated
only inside the checker -- never printed here.

STDOUT prints ONLY:
  line 1: "<n0> <D>"                      water-entry index, sensor depth
  line 2: "<n_train> <test_id>"
  n_train lines: "<entry_deg> <offset> <time>"   noisy measurements

The hidden layer thicknesses/indices and the seed are NOT printed.
"""
import sys, random, math

N0 = 1.33     # fixed index of the entry medium (open water above the stack)
D = 10.0      # fixed depth of the sensor plane


def true_stack(t):
    """Hidden 3-layer stack for this test id (lives in gen AND checker, never printed).
    Layers 1,2 are always DENSER than N0 (never cause TIR, at any angle).
    Layer 3 (deepest, reaches the sensor) is always LESS dense than N0, so it has a
    genuine, reachable critical angle theta_c = asin(n3/N0) that near-normal data
    cannot expose."""
    rng = random.Random(900001 + t * 7919)
    d1 = rng.uniform(1.4, 3.0)
    d2 = rng.uniform(1.4, 3.0)
    cap = 0.78 * D
    if d1 + d2 > cap:
        sc = cap / (d1 + d2)
        d1 *= sc
        d2 *= sc
    d3 = D - d1 - d2
    n1 = rng.uniform(1.37, 1.50)
    n2 = rng.uniform(1.37, 1.50)
    n3 = rng.uniform(1.08, 1.29)
    return [(d1, n1), (d2, n2), (d3, n3)], rng


def trace(theta0, n0, layers):
    """Ray-parameter (Snell-invariant) composition through flat layers.
    Returns (offset, time) or None if TIR occurs in some layer."""
    s = math.sin(theta0)
    x = 0.0
    tt = 0.0
    for d, n in layers:
        sin_i = (n0 / n) * s
        if sin_i >= 1.0 - 1e-15:
            return None
        cos_i = math.sqrt(max(0.0, 1.0 - sin_i * sin_i))
        theta_i = math.asin(sin_i)
        x += d * math.tan(theta_i)
        tt += d * n / cos_i
    return x, tt


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    layers, rng = true_stack(t)

    n_train = max(45, 90 - 5 * (t - 1))
    noise_x = 0.008 + 0.004 * (t - 1)
    noise_t = 0.004 + 0.002 * (t - 1)
    max_deg = max(3.0, 5.5 - 0.2 * (t - 1))

    rows = []
    tries = 0
    while len(rows) < n_train and tries < n_train * 20:
        tries += 1
        deg = rng.uniform(0.25, max_deg)
        res = trace(math.radians(deg), N0, layers)
        if res is None:
            continue
        x, tm = res
        x_noisy = x + rng.gauss(0.0, noise_x)
        t_noisy = tm + rng.gauss(0.0, noise_t)
        rows.append((deg, x_noisy, t_noisy))

    out = ["%.6f %.6f" % (N0, D), "%d %d" % (len(rows), t)]
    for deg, x, tm in rows:
        out.append("%.6f %.6f %.6f" % (deg, x, tm))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
