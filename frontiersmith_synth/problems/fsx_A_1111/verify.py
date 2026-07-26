#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the layered-lagoon refraction task.

- Reads (n0, D, test id) from <in>'s header, then regenerates the hidden
  3-layer stack for that id (identical code to gen.py) -- the ground truth
  lives ONLY here and in gen.py, never in the printed data.
- Parses the participant's proposed layer stack from <out>:
      L
      d_1 n_1
      ...
      d_L n_L
  (d_i, n_i floats; the LAST layer's printed thickness is ignored and
  replaced by whatever depth remains to reach the fixed sensor plane D --
  only its refractive index matters).
- Scores by ray-tracing the participant's OWN stack (same Snell-invariant
  physics as gen.py) at a fixed, deterministic grid of HELD-OUT entry
  angles that sweeps from moderate to near-grazing incidence -- a regime
  the near-normal training data never showed, including angles beyond the
  true deep layer's critical angle where the true ray never reaches the
  sensor (total internal reflection).
- For each held-out angle: if truth and prediction agree on whether the
  ray exits, and both exit, score the normalised offset+time error
  (capped); if they disagree on exit/no-exit, charge a fixed miss penalty
  (same cap) -- this is what punishes a model with no critical angle, or
  with the wrong one, at STEEP angles even though it never conflicts with
  the near-normal training data.
- F = mean(per-angle error) * (1 + LAMBDA * L)              (parsimony)
  B = mean(per-angle error of the checker's own trivial,
            unrefracted L=1, n=n0 "straight ray" baseline) * (1 + LAMBDA)
  Ratio = min(1000, 100*B/F) / 1000
"""
import sys, math, random

N0 = 1.33
D = 10.0
MAX_OUT_BYTES = 20000
MAX_LAYERS = 8
PENALTY_MISS = 3.0
SCALE_X = 2.2
SCALE_T = 2.6
LAMBDA_PARSIMONY = 0.03
HELD_DEG = [8, 15, 22, 30, 38, 45, 50, 55, 60, 63, 66, 69, 72, 75, 78, 82, 86, 89]


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def true_stack(t):
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
    return [(d1, n1), (d2, n2), (d3, n3)]


def trace(theta0, n0, layers):
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


def point_error(pred_layers, n0, theta0_deg, true_res):
    theta0 = math.radians(theta0_deg)
    pred_res = trace(theta0, n0, pred_layers)
    if true_res is None and pred_res is None:
        return 0.0
    if true_res is None or pred_res is None:
        return PENALTY_MISS
    x_t, t_t = true_res
    x_p, t_p = pred_res
    err = abs(x_p - x_t) / SCALE_X + abs(t_p - t_t) / SCALE_T
    return min(PENALTY_MISS, err)


def mean_error(layers, n0, truths):
    tot = 0.0
    for deg, tr in zip(HELD_DEG, truths):
        tot += point_error(layers, n0, deg, tr)
    return tot / len(HELD_DEG)


def parse_output(text):
    toks = text.split()
    if not toks:
        fail("empty output")
    try:
        vals = [float(x) for x in toks]
    except Exception:
        fail("non-numeric token")
    for v in vals:
        if not math.isfinite(v):
            fail("non-finite value in output")
    if vals[0] != int(vals[0]):
        fail("layer count L must be an integer")
    L = int(vals[0])
    if L < 1 or L > MAX_LAYERS:
        fail("layer count L out of range [1,%d]" % MAX_LAYERS)
    rest = vals[1:]
    if len(rest) != 2 * L:
        fail("expected %d (d n) pairs, got %d numbers" % (L, len(rest)))
    layers_raw = [(rest[2 * i], rest[2 * i + 1]) for i in range(L)]
    for i, (d, n) in enumerate(layers_raw):
        if n <= 0.05 or n > 4.0:
            fail("refractive index n_%d=%.6g out of sane range (0.05,4.0]" % (i + 1, n))
        if i < L - 1 and d <= 1e-6:
            fail("layer %d thickness must be positive" % (i + 1))
    prefix = sum(d for d, _ in layers_raw[:-1])
    if prefix >= D - 1e-6:
        fail("submitted layer thicknesses already reach/exceed the sensor depth D=%.3f" % D)
    last_d = D - prefix
    layers = [(layers_raw[i][0], layers_raw[i][1]) for i in range(L - 1)]
    layers.append((last_d, layers_raw[-1][1]))
    return L, layers


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        with open(inf) as fh:
            n0, d_in = map(float, fh.readline().split())
            header2 = fh.readline().split()
            t = int(header2[1])
    except Exception:
        fail("bad instance header")
    if t < 1 or t > 100000:
        fail("bad test id")

    try:
        with open(outf, "rb") as fh:
            raw = fh.read(MAX_OUT_BYTES + 1)
    except Exception:
        fail("cannot read output")
    if len(raw) > MAX_OUT_BYTES:
        fail("output too large")
    text = raw.decode("utf-8", "replace")

    L, layers = parse_output(text)

    true_layers = true_stack(t)
    truths = [trace(math.radians(deg), N0, true_layers) for deg in HELD_DEG]

    F_err = mean_error(layers, N0, truths)
    B_err = mean_error([(D, N0)], N0, truths)

    F = F_err * (1.0 + LAMBDA_PARSIMONY * L)
    B = B_err * (1.0 + LAMBDA_PARSIMONY * 1)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F_err=%.6f B_err=%.6f L=%d  Ratio: %.6f" % (F_err, B_err, L, sc / 1000.0))


if __name__ == "__main__":
    main()
