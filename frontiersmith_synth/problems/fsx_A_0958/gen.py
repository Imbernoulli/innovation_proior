#!/usr/bin/env python3
"""
gen.py <testId>  -- prints the SMALL-SIGNAL / SUB-RESONANT training log for one
instance of the hidden-response-law-recovery problem.

Each line after the header is a data row:  "<f> <a> <y>"
  f = drive frequency  (probed only well BELOW the hidden resonance)
  a = drive amplitude  (probed only in the small-signal, quasi-linear range)
  y = the device's measured response (multiplicative noise)

The hidden ground-truth law (resonance frequency, resonance width, saturation
scale, overall gain), the random seed, and the held-out LARGE-SIGNAL /
NEAR-AND-PAST-RESONANCE region are NOT emitted here -- data rows only. They
live exclusively inside the checker (verify.py), which contains an identical
copy of the derivation logic below.
"""
import sys


# ---- deterministic PRNG (LCG); identical logic must live in verify.py ----
def _rng(seed):
    state = [(seed * 2654435761 + 12345) & 0x7FFFFFFF]

    def nxt():
        state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
        return state[0] / 0x7FFFFFFF

    return nxt


def derive_params(test_id):
    """Ground-truth law: R(f,a) = G * (As*tanh(a/As)) / (1 + ((f-f0)/w)**2)
    -- a SEPARABLE product of a saturating (compressive) amplitude response
    and a resonant (Lorentzian) frequency response, sharing one overall
    gain G. G, As, f0, w are all hidden."""
    r = _rng(1000 + test_id)
    G = 1.5 + 1.5 * r()     # overall gain in [1.5, 3.0]
    As = 3.0 + 3.0 * r()    # saturation scale in [3.0, 6.0]
    f0 = 10.0 + 6.0 * r()   # hidden resonance frequency in [10.0, 16.0]
    w = 1.2 + 1.2 * r()     # resonance half-width in [1.2, 2.4]
    return G, As, f0, w


def n_train(test_id):
    return 60 + 6 * (test_id - 1)          # difficulty ladder: 60 .. 114 rows


def noise_rel(test_id):
    return 0.05 + 0.01 * (test_id - 1)     # difficulty ladder: more noise later


TRAIN_F_LO = 0.1
TRAIN_F_MARGIN_W = 1.5     # train f upper bound = f0 - 1.5*w  (strictly sub-resonant)
TRAIN_A_LO_FRAC = 0.05
TRAIN_A_HI_FRAC = 0.75     # train a upper bound = 0.75*As     (quasi-linear range)


def clean_R(f, a, params):
    G, As, f0, w = params
    sat = As * _tanh(a / As)
    res = 1.0 / (1.0 + ((f - f0) / w) ** 2)
    return G * sat * res


def _tanh(x):
    # avoid overflow for the (never actually reached in training) extreme case
    if x > 40.0:
        return 1.0
    if x < -40.0:
        return -1.0
    e2x = pow(2.718281828459045, 2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)


def make_train_rows(test_id):
    params = derive_params(test_id)
    G, As, f0, w = params
    n = n_train(test_id)
    nr = noise_rel(test_id)
    f_hi = f0 - TRAIN_F_MARGIN_W * w
    a_lo = TRAIN_A_LO_FRAC * As
    a_hi = TRAIN_A_HI_FRAC * As
    rx = _rng(3000 + test_id)   # train (f,a)-stream
    rn = _rng(5000 + test_id)   # train noise-stream
    rows = []
    for _ in range(n):
        f = TRAIN_F_LO + (f_hi - TRAIN_F_LO) * rx()
        a = a_lo + (a_hi - a_lo) * rx()
        clean = clean_R(f, a, params)
        y = clean * (1.0 + nr * (2.0 * rn() - 1.0))
        rows.append((f, a, y))
    return rows


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    rows = make_train_rows(test_id)
    out = [str(len(rows))]
    for f, a, y in rows:
        out.append("%.10g %.10g %.10g" % (f, a, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
