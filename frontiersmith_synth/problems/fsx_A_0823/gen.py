#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy lock-in notebook to stdout.

Two damped mechanical oscillators sit inside a sealed box, linearly coupled to
each other, and only the FIRST one is driven by an external force at angular
frequency w (unit drive amplitude). Its steady-state complex response

    X1(w) = N(w) / D(w)
    N(w)  = (Om2^2 - w^2) + i*g2*w
    D(w)  = [(Om1^2-w^2)+i*g1*w] * [(Om2^2-w^2)+i*g2*w]  -  kc^2

is a rational function of w fixed by FIVE hidden physical constants: the two
natural frequencies Om1 < Om2, the two damping rates g1, g2, and the coupling
constant kc (kc^2 < Om1^2*Om2^2, so the box is stable and both oscillators are
individually stiff: 0 < Om1^2-kc, 0 < Om2^2-kc).

The notebook records X1 (in-phase / quadrature lock-in channels, i.e. Xre,
Xim) at drive frequencies sampled ONLY well BELOW the lower resonance Om1 --
a smooth, featureless, monotonically-rising tail with no visible peak.  The
hidden Om1, Om2, g1, g2, kc are NEVER printed; only the noisy (w, Xre, Xim)
rows are.

STDOUT: header "<N> <testId>", then N lines "<w> <Xre> <Xim>".
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
OM1_LO, OM1_HI = 4.0, 5.0
RATIO_LO, RATIO_HI = 1.25, 1.70       # Om2 = Om1 * ratio (still close enough for a real
                                       # avoided-crossing splitting, separated enough to be
                                       # numerically identifiable from the tail)
DAMP_FRAC_LO, DAMP_FRAC_HI = 0.06, 0.16
COUPL_FRAC_LO, COUPL_FRAC_HI = 0.04, 0.15
TAIL_FMIN_FRAC = 0.05                # training w in [TAIL_FMIN_FRAC, TAIL_FMAX_FRAC] * Om1
TAIL_FMAX_FRAC = 0.45
N_TRAIN = 44
NOISE_TRAIN = 0.05                   # additive noise, fraction of tail-edge amplitude scale


def hidden_params(t):
    """Hidden physical constants for this test id (identical in gen.py and verify.py)."""
    rng = random.Random(700000 + t * 913171)
    Om1 = rng.uniform(OM1_LO, OM1_HI)
    ratio = rng.uniform(RATIO_LO, RATIO_HI)
    Om2 = Om1 * ratio
    g1 = rng.uniform(DAMP_FRAC_LO, DAMP_FRAC_HI) * Om1
    g2 = rng.uniform(DAMP_FRAC_LO, DAMP_FRAC_HI) * Om2
    cfrac = rng.uniform(COUPL_FRAC_LO, COUPL_FRAC_HI)
    kc = cfrac * (Om1 * Om1)         # kc < Om1^2 < Om1*Om2  -> stable, both k_i > 0
    return Om1, Om2, g1, g2, kc


def response(w, Om1, Om2, g1, g2, kc):
    """Complex steady-state response X1(w) = Xre + i*Xim, drive amplitude F0=1."""
    a1 = Om1 * Om1 - w * w
    a2 = Om2 * Om2 - w * w
    b1 = g1 * w
    b2 = g2 * w
    Dre = a1 * a2 - b1 * b2 - kc * kc
    Dim = a1 * b2 + a2 * b1
    Dmag2 = Dre * Dre + Dim * Dim
    if Dmag2 < 1e-300:
        Dmag2 = 1e-300
    Xre = (a2 * Dre + b2 * Dim) / Dmag2
    Xim = (b2 * Dre - a2 * Dim) / Dmag2
    return Xre, Xim


def amplitude(Xre, Xim):
    return math.sqrt(Xre * Xre + Xim * Xim)


def gen_train(t):
    Om1, Om2, g1, g2, kc = hidden_params(t)
    rng = random.Random(271828 + t * 97)
    fmin = TAIL_FMIN_FRAC * Om1
    fmax = TAIL_FMAX_FRAC * Om1
    Xre0, Xim0 = response(fmax, Om1, Om2, g1, g2, kc)
    scale0 = amplitude(Xre0, Xim0)
    rows = []
    for _ in range(N_TRAIN):
        w = rng.uniform(fmin, fmax)
        Xre, Xim = response(w, Om1, Om2, g1, g2, kc)
        Xre_n = Xre + NOISE_TRAIN * scale0 * rng.gauss(0.0, 1.0)
        Xim_n = Xim + NOISE_TRAIN * scale0 * rng.gauss(0.0, 1.0)
        rows.append((w, Xre_n, Xim_n))
    rows.sort()
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for w, Xre, Xim in rows:
        out.append("%.8g %.8g %.8g" % (w, Xre, Xim))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
