#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0877 -- "Static on the Line: Pre-Distorting a Memory Channel"
(family: nonlinear-channel-precoder; format B, quality-metric).

THEME.  A message must be pushed through a fixed, KNOWN nonlinear channel with
memory: a saturating amplifier whose current output depends not just on the
current input symbol but on a short window of RECENT input symbols too (a
finite-impulse-response "smear" followed by a saturating nonlinearity -- the
textbook model of a power amplifier with memory).  The transmitter gets to
choose the INPUT sequence before it goes in; the goal is to choose an input
whose channel output lands close to a given target message, while not
spending more input energy than necessary (energy costs power / drifts the
amplifier toward saturation on later, unrelated traffic).

This is precoding / channel-inversion (the model plays "precoder"), scored
across a fixed, seeded family of channel+message instances (generalization
across instances), where multiple input sequences can approximately realize
a given output -- so a real search over the preimage (not just a formula
plugged into one symbol at a time) is required to spend the energy budget
well and to correctly cancel inter-symbol interference (ISI) from the
channel's memory.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N (int), "L": memory order (int),
             "h": [h_0, ..., h_L]  (float FIR taps, h_0 != 0),
             "A": saturation level (float > 0),
             "xmax": max allowed |input symbol| (float > 0),
             "lambda": energy weight (float >= 0),
             "target": [y*_0, ..., y*_{N-1}]  (float, each in (-A, A))}
  stdout: ONE JSON object:
            {"x": [x_0, ..., x_{N-1}]}   # N real numbers, the input sequence

  A candidate answer is VALID iff `x` is a list of exactly N finite numbers
  with |x_i| <= xmax + 1e-6 for every i.  Invalid output, wrong length, an
  out-of-range symbol, non-finite values, a crash, a timeout, or non-JSON
  output -> that instance scores 0.0.

CHANNEL MODEL.  For t = 0..N-1:
    v_t = h_0*x_t + h_1*x_{t-1} + ... + h_L*x_{t-L}     (x_{negative index} = 0)
    y_t = A * tanh(v_t / A)                             (saturating nonlinearity)

SCORING (deterministic; no wall-time).  Per instance the evaluator computes,
itself, from the full instance:
    cost(x) = mean_t (y_t - target_t)^2  +  lambda * mean_t (x_t^2)
    q_zero  = cost of the all-zero input (a reachable, easily-computed weak
              reference: y=0 everywhere, energy=0, so q_zero = mean(target^2))
    q_ideal = 0   (zero reconstruction error AND zero energy simultaneously --
              generally UNREACHABLE whenever the target is nonzero, since it
              takes nonzero input energy to reproduce a nonzero target; this
              keeps real headroom above any real solver's score)
  and normalizes with an affine anchor (all-zero input -> 0.1, the
  unreachable ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_zero - cost(x)) / q_zero, 0, 1 )
  A candidate that does no better than transmitting nothing scores ~0.1; a
  candidate that does WORSE than transmitting nothing (e.g. a badly aimed
  precoder that amplifies mismatch through the channel's memory) scores
  below 0.1, clamped to 0.

The reported **Ratio** is the mean of `r` over all instances; **Vector** lists
the per-instance scores.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance fields above.
All references (q_zero) are recomputed by THIS parent process from data the
candidate already has, so a frame-walking / introspecting candidate learns
nothing it doesn't already see.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / (1 << 53)   # in [0, 1)

    return nxt


def _rand_range(f, lo, hi):
    return lo + f() * (hi - lo)


# ----------------------------- channel model --------------------------------
def _forward(x, h, A):
    """FIR memory (taps h) followed by a saturating nonlinearity. Causal:
    y_t depends on x_t, x_{t-1}, ..., x_{t-L} only (x with negative index = 0)."""
    n = len(x)
    y = []
    for t in range(n):
        v = 0.0
        for k in range(len(h)):
            if t - k >= 0:
                v += h[k] * x[t - k]
        y.append(A * math.tanh(v / A))
    return y


# ----------------------------- instance family -------------------------------
# (seed, n, L, h taps, A, xmax, lambda, x_true amplitude fraction of xmax, tag)
_SPECS = [
    (101, 50, 1, [0.90, 0.12],             2.0, 3.0, 0.05, 0.50, "easy"),
    (102, 50, 2, [0.85, 0.10, 0.08],       2.0, 3.0, 0.05, 0.50, "easy"),
    (103, 55, 1, [0.80, 0.15],             2.2, 3.2, 0.06, 0.55, "easy"),
    (205, 40, 2, [0.55, 0.40, 0.22],       2.0, 3.0, 0.05, 0.50, "trap"),
    (206, 45, 2, [0.45, 0.50, 0.30],       2.0, 3.0, 0.05, 0.50, "trap"),
    (207, 42, 3, [0.40, 0.40, 0.35, 0.20], 2.0, 3.0, 0.05, 0.50, "trap"),
    (208, 48, 2, [0.40, 0.55, 0.30],       2.2, 3.0, 0.05, 0.50, "trap"),
    (311, 65, 3, [0.35, 0.40, 0.30, 0.25], 2.0, 3.0, 0.08, 0.85, "hard"),
    (312, 60, 2, [0.50, 0.40, 0.30],       2.2, 3.0, 0.08, 0.80, "hard"),
    (313, 70, 3, [0.40, 0.35, 0.30, 0.25], 2.0, 3.2, 0.08, 0.85, "hard"),
]


def _build_instances():
    out = []
    for seed, n, L, h, A, xmax, lam, xtf, tag in _SPECS:
        f = _rng(seed)
        x_true = [_rand_range(f, -xtf * xmax, xtf * xmax) for _ in range(n)]
        target = _forward(x_true, h, A)
        out.append({"name": f"chan{seed}", "n": n, "L": L, "h": list(h),
                    "A": A, "xmax": xmax, "lambda": lam, "target": target,
                    "tag": tag})
    return out


# ----------------------------- validation + scoring ---------------------------
def _cost(x, target, h, A, lam):
    y = _forward(x, h, A)
    n = len(target)
    mse = sum((y[i] - target[i]) ** 2 for i in range(n)) / n
    energy = sum(xi * xi for xi in x) / n
    return mse + lam * energy


def _extract_x(inst, answer):
    """Validate answer against the instance. Return the x list or None."""
    if not isinstance(answer, dict):
        return None
    x = answer.get("x")
    if not isinstance(x, list):
        return None
    n = inst["n"]
    if len(x) != n:
        return None
    xmax = inst["xmax"]
    out = []
    for v in x:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        if abs(v) > xmax + 1e-6:
            return None
        out.append(v)
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        target = inst["target"]
        h = inst["h"]; A = inst["A"]; lam = inst["lambda"]
        q_zero = sum(t * t for t in target) / len(target)   # cost of the all-zero input
        if q_zero < 1e-9:
            q_zero = 1e-9
        public = {"name": inst["name"], "n": inst["n"], "L": inst["L"],
                  "h": list(h), "A": A, "xmax": inst["xmax"], "lambda": lam,
                  "target": list(target)}
        ans, st = isorun.run_candidate(cand, public, timeout=15)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            x = _extract_x(inst, ans)
        except Exception:
            x = None
        if x is None:
            vec.append(0.0)
            continue
        try:
            c = _cost(x, target, h, A, lam)
        except Exception:
            vec.append(0.0)
            continue
        if not (c == c) or c in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_zero - c) / q_zero
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
