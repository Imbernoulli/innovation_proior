#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE calibration burst to stdout.

Family: impulse-response-deconvolve.  A hidden sparse spike train x[] is
blurred by convolving it with a normalised triangular kernel of UNKNOWN
half-width w (the family -- symmetric tent, radius w -- is public; the
per-test w is not) and corrupted with additive Gaussian noise of unknown
sigma.  The solver is shown a CALIBRATION BURST: a paired (x, y) trace
recorded by firing a known test pulse train x_cal through the same unknown
instrument and logging both the pulses and the blurred, noisy output y_cal.
From this the solver must recover a fixed sliding filter that, applied to
window taps of y, predicts x -- then that SAME filter is rolled over a
FRESH, unpaired, held-out trace (new pulses, same instrument) that this
script never prints and the solver never sees.

STDOUT prints ONLY: a header "Ncal R t", then Ncal lines "x_cal[i] y_cal[i]".
The hidden kernel half-width w and noise sigma are NEVER printed (they, plus
the held-out trace, live only inside verify.py).
"""
import sys, random

R = 7                      # public: window radius solvers may reference (ym7..yp7)
W_MAX = 5                  # public: the kernel half-width is known to lie in [1, W_MAX]

# ---- difficulty ladder (public constants; NOT the hidden per-instance law) ----
# Ncal = calibration burst length, w = hidden kernel half-width, sigma = hidden noise std.
# Entries marked TRAP put the number of USABLE calibration rows (Ncal-2R) right at the
# interpolation threshold of the full radius-R=7 window (2R+1=15 free taps): an
# unregularised inverse fit there is maximally unstable (classic double-descent peak) --
# it reproduces the calibration burst near-perfectly in-sample yet the fitted taps blow
# up and amplify fresh noise on the held-out trace far worse than predicting nothing.
_TABLE = {
    1:  dict(Ncal=500, w=5, sigma=0.012),
    2:  dict(Ncal=420, w=4, sigma=0.017),
    3:  dict(Ncal=28,  w=2, sigma=0.055),   # TRAP
    4:  dict(Ncal=380, w=3, sigma=0.017),
    5:  dict(Ncal=28,  w=1, sigma=0.060),   # TRAP
    6:  dict(Ncal=330, w=5, sigma=0.024),
    7:  dict(Ncal=290, w=2, sigma=0.026),
    8:  dict(Ncal=270, w=4, sigma=0.026),
    9:  dict(Ncal=230, w=3, sigma=0.028),
    10: dict(Ncal=28,  w=1, sigma=0.065),   # TRAP
}


def plan_for(t):
    base = ((t - 1) % 10) + 1
    growth = (t - 1) // 10
    d = dict(_TABLE[base])
    d["sigma"] = d["sigma"] * (1.0 + 0.15 * growth)
    d["Ncal"] = max(18, d["Ncal"] - 5 * growth)
    return d


def kernel(w):
    """Normalised triangular (tent) kernel of half-width w, support [-w, w]."""
    z = float((w + 1) ** 2)
    return {j: (w + 1 - abs(j)) / z for j in range(-w, w + 1)}


def make_spikes(rng, n, w, target_k):
    """Sparse, well-separated spike train of length n (list of floats)."""
    min_gap = 2 * w + 3
    positions = []
    tries = 0
    while len(positions) < target_k and tries < 30000:
        tries += 1
        p = rng.randint(2, n - 3)
        if all(abs(p - q) >= min_gap for q in positions):
            positions.append(p)
    x = [0.0] * n
    for p in positions:
        x[p] = rng.uniform(1.0, 3.0)
    return x


def convolve(x, w, sigma, rng):
    n = len(x)
    h = kernel(w)
    y = [0.0] * n
    for nn in range(n):
        s = 0.0
        for j in range(-w, w + 1):
            k = nn - j
            if 0 <= k < n:
                s += h[j] * x[k]
        y[nn] = s + rng.gauss(0.0, sigma)
    return y


def cal_rngs(t):
    """RNG streams for the CALIBRATION side only (held-out streams live in verify.py)."""
    r_spk = random.Random(11_000_003 + t * 9973)
    r_noi = random.Random(22_000_057 + t * 7919)
    return r_spk, r_noi


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    plan = plan_for(t)
    ncal, w, sigma = plan["Ncal"], plan["w"], plan["sigma"]

    r_spk, r_noi = cal_rngs(t)
    target_k = max(3, ncal // 20)
    x_cal = make_spikes(r_spk, ncal, w, target_k)
    y_cal = convolve(x_cal, w, sigma, r_noi)

    out = ["%d %d %d" % (ncal, R, t)]
    for i in range(ncal):
        out.append("%.6f %.6f" % (x_cal[i], y_cal[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
