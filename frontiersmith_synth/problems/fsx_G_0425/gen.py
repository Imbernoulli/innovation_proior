#!/usr/bin/env python3
"""
gen.py <testId>  -- prints the TRAIN sample for one instance of the
"gas-cylinder lab" equation-of-state recovery problem.

Scenario (gas cylinder lab): a fixed amount of a pure real gas is charged into
calibrated cylinders. For a grid of molar volumes V (L/mol) and temperatures
T (K) the lab logs the measured pressure P (bar). Every reading carries a small
relative measurement error. Your job is to recover a closed-form equation of
state P(V, T) that also holds in the HIGH-PRESSURE (small-molar-volume) region,
which is NOT sampled in the training log.

Each printed line is one measurement row:

    <V> <T> <P>

The hidden ground-truth law, its coefficients, the noise seed and the held-out
high-pressure region are NOT emitted here -- data rows only. They live
exclusively inside the checker (verify.py).
"""
import sys


# ---- deterministic PRNG (LCG); identical logic lives in verify.py ----
def _rng(seed):
    state = [(seed * 2654435761 + 12345) & 0x7FFFFFFF]

    def nxt():
        state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
        return state[0] / 0x7FFFFFFF

    return nxt


R = 0.083145  # L*bar/(mol*K)


def derive_params(test_id):
    """Ground-truth coefficients (a = attraction, b = co-volume)."""
    r = _rng(1000 + test_id)
    a = 30.0 + 35.0 * r()     # attraction parameter
    b = 0.022 + 0.016 * r()   # molar co-volume (L/mol)
    return a, b


def noise_rel(test_id):
    # difficulty ladder: later tests carry more irreducible measurement noise
    return 0.025 + 0.005 * (test_id - 1)


def true_P(a, b, V, T):
    """Hidden Redlich-Kwong-style real-gas EOS (NOT revealed to the solver)."""
    return R * T / (V - b) - a / (T ** 0.5 * V * (V + b))


T_GRID = [300.0, 350.0, 400.0, 450.0, 500.0]


def train_V(n=18, lo=0.22, hi=6.0):
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


def make_train(test_id):
    a, b = derive_params(test_id)
    nr = noise_rel(test_id)
    rn = _rng(5000 + test_id)   # train-noise stream
    rows = []
    for T in T_GRID:
        for V in train_V():
            clean = true_P(a, b, V, T)
            y = clean * (1.0 + nr * (2.0 * rn() - 1.0))
            rows.append((V, T, y))
    return rows


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    rows = make_train(test_id)
    out = []
    for V, T, y in rows:
        out.append("%.10g %.10g %.10g" % (V, T, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
