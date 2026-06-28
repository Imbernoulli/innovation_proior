#!/usr/bin/env python3
"""Instance generator for "Parameter Placement for a Simulated Controller" (ALE-Bench).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format (all integers / fixed-point):

    S K Q
    LO_0 HI_0   LO_1 HI_1   ...   LO_{K-1} HI_{K-1}
    T
    ref_0 ref_1 ... ref_{T-1}
    d_0   d_1   ... d_{T-1}

Semantics (see context.md for the full statement):

  We tune the gains of a piecewise controller that drives a deterministic scalar
  plant to track a reference trajectory `ref[0..T-1]`.

    * The horizon T is split into S equal-length SEGMENTS. Each segment s owns its
      own gain vector g[s][0..K-1] (K real gains, here K = 3: a PID-like triple).
      So the decision vector has S*K real entries.
    * Each gain g[s][k] must lie in the closed box [LO_k, HI_k] and is reported on a
      fixed lattice: the solver outputs an INTEGER code in [0, Q] for each gain, and
      the scorer maps code -> value LO_k + (HI_k - LO_k) * code / Q. So every gain is
      automatically in-bounds and quantized; an out-of-range code is INFEASIBLE.
    * The plant. State (x, v) starts at (x0, v0) = (ref[0], 0). At step t the
      controller in the active segment sees the tracking error e = ref[t] - x and
      its discrete derivative, and produces a force
            f = g[k=0]*e + g[k=1]*(e - e_prev) + g[k=2]*v        (PD on error + velocity damping)
      The plant integrates (unit mass, dt = 1, with a small drag and an exogenous
      disturbance d[t]):
            v <- v + f - DRAG*v + d[t]
            x <- x + v
      We accumulate squared tracking error  COST = sum_t (ref[t] - x_after)^2.

  Because the plant is deterministic and each segment owns disjoint gains, changing a
  single segment's gain only perturbs the trajectory from that segment's first step
  onward -- the PREFIX-STATE CACHE that makes incremental re-simulation cheap.

  The reference is a smooth-ish random walk (a few sinusoids + steps) and d[t] is a
  small bounded disturbance, so a non-trivial, segment-dependent gain schedule is
  needed: a single global gain cannot track the changing reference well.

Everything is emitted as integers (reference and disturbance as fixed-point *1000)
so the instance parses identically in C++ and Python.
"""
import sys
import math
import random

SCALE = 1000  # fixed-point scale for ref/disturbance


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x39A7_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    K = 3                       # gains per segment (PD on error + velocity damping)
    Q = 1000                    # quantization levels per gain (codes in [0, Q])
    S = rng.randint(6, 12)      # number of segments
    seg_len = rng.randint(40, 70)
    T = S * seg_len             # total horizon

    # Per-gain boxes. Gain 0 (proportional) and 1 (derivative) positive; gain 2
    # (velocity damping) negative (a restoring drag).
    boxes = [
        (0.0, rng.uniform(1.2, 2.2)),     # Kp in [0, ~2]
        (0.0, rng.uniform(0.8, 1.6)),     # Kd in [0, ~1.5]
        (-1.2, 0.0),                      # Kv (damping) in [-1.2, 0]
    ]

    # Reference trajectory: sum of a few sinusoids + a couple of level steps.
    n_sin = rng.randint(2, 4)
    comps = []
    for _ in range(n_sin):
        amp = rng.uniform(3.0, 12.0)
        period = rng.uniform(T / 6.0, T / 1.5)
        phase = rng.uniform(0, 2 * math.pi)
        comps.append((amp, period, phase))
    base = rng.uniform(-5.0, 5.0)
    # a few step changes
    n_steps = rng.randint(1, 3)
    step_times = sorted(rng.sample(range(5, T - 5), n_steps))
    step_vals = [rng.uniform(-8.0, 8.0) for _ in range(n_steps)]

    ref = []
    for t in range(T):
        y = base
        for amp, period, phase in comps:
            y += amp * math.sin(2 * math.pi * t / period + phase)
        for st, sv in zip(step_times, step_vals):
            if t >= st:
                y += sv
        ref.append(y)

    # Exogenous disturbance: small bounded noise (a low-frequency drift + jitter).
    dist = []
    drift = 0.0
    for t in range(T):
        drift += rng.uniform(-0.05, 0.05)
        drift = max(-1.0, min(1.0, drift))
        d = drift + rng.uniform(-0.15, 0.15)
        dist.append(d)

    out = []
    out.append(f"{S} {K} {Q}")
    out.append("  ".join(f"{int(round(lo*SCALE))} {int(round(hi*SCALE))}" for lo, hi in boxes))
    out.append(str(T))
    out.append(" ".join(str(int(round(r * SCALE))) for r in ref))
    out.append(" ".join(str(int(round(d * SCALE))) for d in dist))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
