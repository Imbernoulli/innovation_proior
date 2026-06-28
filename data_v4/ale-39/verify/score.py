#!/usr/bin/env python3
"""Deterministic local scorer for "Parameter Placement for a Simulated Controller"
(ALE-Bench heuristic optimization).

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. Higher is better.

Scoring rule (must match context.md exactly):

  The instance gives S segments, K gains per segment, quantization Q, per-gain boxes
  [LO_k, HI_k] (fixed-point *1000), a horizon T, a reference ref[0..T-1] and a
  disturbance d[0..T-1] (both fixed-point *1000).

  A solution is S*K integer CODES, one per gain g[s][k]. It is FEASIBLE iff
    (1) it parses as exactly S*K integer tokens, and
    (2) every code is in [0, Q].
  Each code is mapped to a real gain value
        g[s][k] = LO_k + (HI_k - LO_k) * code[s][k] / Q .

  The deterministic plant (unit mass, dt = 1) is simulated with the per-segment gains:
        x = ref[0]; v = 0; e_prev = 0
        for t in 0..T-1:
            s = t // seg_len                     # active segment
            e = ref[t] - x
            f = g[s][0]*e + g[s][1]*(e - e_prev) + g[s][2]*v
            v = v + f - DRAG*v + d[t]
            x = x + v
            COST += (ref[t] - x)^2
            e_prev = e
  with DRAG = 0.02. If at any step |x|, |v| or COST exceeds 1e15 or is NaN/inf, the
  simulation has DIVERGED and the solution is INFEASIBLE (score 0).

  A reference "zero-gain" controller (all gains = 0, i.e. no control: f == 0) is
  simulated the same way to give COST_ZERO (the open-loop tracking cost). Tracking
  performance spans many orders of magnitude, so the score is the open-loop-relative
  cost reduction measured in DECADES (log10), which discriminates across the whole
  controlled regime:

        gain_decades = log10( COST_ZERO / max(COST, EPS) )      EPS = 1e-9
        score = round( 1e5 * max(0.0, gain_decades) )           if FEASIBLE,
        score = 0                                               otherwise.

  COST >= 0. A controller that does no better than open loop (COST >= COST_ZERO)
  scores 0; every decade by which it drives the tracking cost below open loop is worth
  1e5 points, so a controller reducing cost from ~1e10 to ~1e2 scores ~8e5. The score
  rises monotonically and smoothly as COST falls. Any infeasible output (wrong token
  count, an out-of-range code, or a diverging simulation) floors the score to 0.
"""
import sys
import math

SCALE = 1000.0
DRAG = 0.02
DIVERGE = 1e15


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    S = int(next(it)); K = int(next(it)); Q = int(next(it))
    boxes = []
    for _ in range(K):
        lo = int(next(it)) / SCALE
        hi = int(next(it)) / SCALE
        boxes.append((lo, hi))
    T = int(next(it))
    ref = [int(next(it)) / SCALE for _ in range(T)]
    dist = [int(next(it)) / SCALE for _ in range(T)]
    return S, K, Q, boxes, T, ref, dist


def simulate(S, K, Q, boxes, T, ref, dist, gains):
    """Run the plant with the given per-segment gain VALUES. Returns COST or None
    if the simulation diverges (NaN/inf/overflow)."""
    seg_len = T // S
    x = ref[0]
    v = 0.0
    e_prev = 0.0
    cost = 0.0
    for t in range(T):
        s = t // seg_len
        if s >= S:
            s = S - 1
        e = ref[t] - x
        g0, g1, g2 = gains[s][0], gains[s][1], gains[s][2]
        f = g0 * e + g1 * (e - e_prev) + g2 * v
        v = v + f - DRAG * v + dist[t]
        x = x + v
        err = ref[t] - x
        cost += err * err
        e_prev = e
        if not (math.isfinite(x) and math.isfinite(v) and math.isfinite(cost)):
            return None
        if abs(x) > DIVERGE or abs(v) > DIVERGE or cost > DIVERGE:
            return None
    return cost


def read_solution(path, S, K, Q):
    """Parse S*K integer codes. Returns the code grid or None if it does not parse
    as exactly S*K integer tokens, or any code is out of [0, Q]."""
    with open(path) as f:
        toks = f.read().split()
    if len(toks) != S * K:
        return None
    codes = [[0] * K for _ in range(S)]
    k = 0
    for s in range(S):
        for j in range(K):
            tok = toks[k]; k += 1
            try:
                c = int(tok)
            except ValueError:
                return None
            if c < 0 or c > Q:
                return None
            codes[s][j] = c
    return codes


def codes_to_gains(codes, S, K, Q, boxes):
    gains = [[0.0] * K for _ in range(S)]
    for s in range(S):
        for j in range(K):
            lo, hi = boxes[j]
            gains[s][j] = lo + (hi - lo) * codes[s][j] / Q
    return gains


def score(instance_path, solution_path):
    S, K, Q, boxes, T, ref, dist = read_instance(instance_path)
    codes = read_solution(solution_path, S, K, Q)
    if codes is None:
        return 0
    gains = codes_to_gains(codes, S, K, Q, boxes)
    cost = simulate(S, K, Q, boxes, T, ref, dist, gains)
    if cost is None:
        return 0  # diverged -> infeasible
    # zero-gain (open-loop) reference cost
    zero_gains = [[0.0] * K for _ in range(S)]
    cost_zero = simulate(S, K, Q, boxes, T, ref, dist, zero_gains)
    if cost_zero is None or cost_zero <= 0.0:
        # degenerate; fall back to a pure inverse-cost score
        return int(round(1e6 / (1.0 + cost)))
    eps = 1e-9
    gain_decades = math.log10(cost_zero / max(cost, eps))
    if gain_decades <= 0.0:
        return 0
    return int(round(1e5 * gain_decades))


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance_file> <solution_file>\n")
        sys.exit(1)
    print(score(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
