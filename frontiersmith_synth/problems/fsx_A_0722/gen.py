#!/usr/bin/env python3
"""gen.py <testId> -- prints one Ladder Filter Synthesis instance to stdout.

Instance = a fixed alternating L/C ladder topology of order N (odd slots = series
inductor, even slots = shunt capacitor, slot 1 nearest the source), equal source/load
resistance Z0, discrete candidate-value lists per component type (index 0 always means
"not populated"), and a target |H(f)| curve (dB) sampled at M frequencies.

The hidden target curve is generated from a genuine doubly-terminated Butterworth LC
ladder prototype of the SAME order N (classical g_k = 2*sin((2k-1)*pi/(2N)) recursion),
at a cutoff frequency f_c that is NOT reported anywhere in the instance -- the solver
must infer it (or otherwise fit the curve) from the (f, target_dB) samples themselves.
Everything is seeded from testId only (deterministic).
"""
import sys, math, random


def cascade_mag(components, Z0, freqs):
    """components: list of ('L'|'C', value). Returns |H(f)| for each f via ABCD cascade."""
    out = []
    for f in freqs:
        w = 2.0 * math.pi * f
        A, B, C, D = complex(1, 0), complex(0, 0), complex(0, 0), complex(1, 0)
        for typ, val in components:
            if typ == 'L':
                a, b, c, d = complex(1, 0), complex(0, w * val), complex(0, 0), complex(1, 0)
            else:
                a, b, c, d = complex(1, 0), complex(0, 0), complex(0, w * val), complex(1, 0)
            A, B, C, D = A * a + B * c, A * b + B * d, C * a + D * c, C * b + D * d
        denom = A * Z0 + B + C * (Z0 * Z0) + D * Z0
        mag = 0.0 if abs(denom) < 1e-300 else abs(Z0 / denom)
        out.append(mag)
    return out


def make_grid(lo, hi, cnt):
    """cnt positive (nonzero) log-spaced levels from lo to hi inclusive."""
    if cnt <= 1:
        return [lo]
    ratio = (hi / lo) ** (1.0 / (cnt - 1))
    return [lo * (ratio ** j) for j in range(cnt)]


def main():
    t = int(sys.argv[1])
    rng = random.Random(1_000_003 * t + 17)

    N = 2 + t  # t=1..10 -> N=3..12
    Z0_choices = [50.0, 75.0, 50.0, 100.0, 50.0, 75.0, 120.0, 50.0, 100.0, 75.0]
    Z0 = Z0_choices[(t - 1) % len(Z0_choices)]

    # hidden cutoff frequency (never printed)
    f_c = math.exp(rng.uniform(math.log(2000.0), math.log(60000.0)))
    w_c = 2.0 * math.pi * f_c

    # classical Butterworth doubly-terminated ladder prototype g-values
    g = [2.0 * math.sin((2 * k - 1) * math.pi / (2 * N)) for k in range(1, N + 1)]

    components = []
    ideal_L, ideal_C = [], []
    for k in range(1, N + 1):
        gk = g[k - 1]
        if k % 2 == 1:  # series inductor
            val = gk * Z0 / w_c
            components.append(('L', val))
            ideal_L.append(val)
        else:  # shunt capacitor
            val = gk / (Z0 * w_c)
            components.append(('C', val))
            ideal_C.append(val)

    # trap cases: coarser discrete grids -> harder quantization / recovery
    coarse = t in (4, 7, 9, 10)
    levels = 6 if coarse else 16

    Lmin, Lmax = min(ideal_L), max(ideal_L)
    Cmin, Cmax = min(ideal_C), max(ideal_C)
    L_grid = [0.0] + make_grid(Lmin * 0.30, Lmax * 3.0, levels)
    C_grid = [0.0] + make_grid(Cmin * 0.30, Cmax * 3.0, levels)

    # asymmetric frequency window around f_c (NOT necessarily centered -> f_c is not
    # recoverable simply from the geometric mean of the sampled range)
    lo_mult = rng.uniform(15.0, 50.0)
    hi_mult = rng.uniform(15.0, 50.0)
    f_lo = f_c / lo_mult
    f_hi = f_c * hi_mult
    M = 30
    freqs = [f_lo * ((f_hi / f_lo) ** (j / (M - 1))) for j in range(M)]

    mags = cascade_mag(components, Z0, freqs)
    target_db = [20.0 * math.log10(max(m, 1e-15)) for m in mags]

    # bounded ripple/tolerance on top of the ideal prototype curve: real target specs are
    # never PERFECTLY achievable by any discrete component choice; this keeps a genuine
    # residual-error floor so the objective doesn't saturate at zero.
    ripple_amp = 4.0
    ripple_cycles = rng.uniform(2.5, 4.5)
    ripple_phase = rng.uniform(0.0, 2.0 * math.pi)
    for j in range(M):
        target_db[j] += ripple_amp * math.sin(2.0 * math.pi * ripple_cycles * j / (M - 1) + ripple_phase)

    cost_per_component = 0.04

    out = []
    out.append(f"{N} {Z0:.6f} {M} {cost_per_component:.6f}")
    out.append(f"{len(L_grid)} " + " ".join(f"{v:.10g}" for v in L_grid))
    out.append(f"{len(C_grid)} " + " ".join(f"{v:.10g}" for v in C_grid))
    out.append(" ".join(f"{v:.8g}" for v in freqs))
    out.append(" ".join(f"{v:.8g}" for v in target_db))
    print("\n".join(out))


if __name__ == "__main__":
    main()
