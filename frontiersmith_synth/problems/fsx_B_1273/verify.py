#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_1273 (pension-glidepath-design).

The participant's artifact is a de-risking POLICY GRID: for every year
t = 1..T and every funded-ratio bucket b = 0..4, a risky-asset weight
w(t,b) in [0,1] (safe weight = 1 - w). The checker replays every scenario
block given in the instance under that SAME grid (a policy must generalize
across all blocks -- it cannot be hand-tailored per scenario, since two
different scenarios reaching the same (t,b) state are forced to reuse the
same weight), and scores the resulting funded-ratio outcome at the
horizon, averaged across blocks.

Mechanics per scenario block, year t = 1..T (remaining = T - t + 1):
  bucket b        = which of the 5 funded-ratio buckets FR_prev = A/L falls in
  contribution C   = c_base * flex[b]                      (contribution flexibility)
  liability        L *= clip(1 + g - remaining*dr, 0.5, 1.5) (duration = remaining years:
                                                               a rate shock dr moves L more
                                                               the longer the horizon left)
  assets           A  = (A + C) * (1 + w*r_risky + (1-w)*r_safe)   (w = w(t,b), sequential
                                                                     compounding -> sequence-
                                                                     of-returns risk)
Per-scenario score: fr = A_T / L_T; s = min(1.5, fr) if fr >= 1 else fr**2.
F = mean(s) over all scenario blocks. B = the same simulation run under the
fully-immunized baseline grid w == 0 everywhere (take zero risk, always).
Ratio = min(1000, 100*F/max(1e-9,B)) / 1000.
"""
import math
import sys

MAX_TOKENS = 2_000_000


def read_instance(path):
    toks = open(path).read().split()
    if len(toks) > MAX_TOKENS:
        raise ValueError("instance too large")
    ptr = 0

    def nxt():
        nonlocal ptr
        v = toks[ptr]
        ptr += 1
        return v

    T = int(nxt())
    M = int(nxt())
    A0 = float(nxt())
    L0 = float(nxt())
    c_base = float(nxt())
    boundaries = [float(nxt()) for _ in range(4)]
    flex = [float(nxt()) for _ in range(5)]
    blocks = []
    for _ in range(M):
        block = []
        for _ in range(T):
            r_risky = float(nxt())
            r_safe = float(nxt())
            dr = float(nxt())
            g = float(nxt())
            block.append((r_risky, r_safe, dr, g))
        blocks.append(block)
    return T, M, A0, L0, c_base, boundaries, flex, blocks


def bucket(fr, boundaries):
    for i, b in enumerate(boundaries):
        if fr < b:
            return i
    return len(boundaries)


def read_grid(path, T):
    """Returns grid[t-1][b] or None (+reason) on any structural violation."""
    raw = open(path).read().split()
    if len(raw) != 5 * T:
        return None, f"expected exactly {5*T} tokens, got {len(raw)}"
    vals = []
    for tok in raw:
        try:
            v = float(tok)
        except ValueError:
            return None, f"non-numeric token {tok!r}"
        if not math.isfinite(v):
            return None, f"non-finite value {tok!r}"
        if v < -1e-9 or v > 1.0 + 1e-9:
            return None, f"weight {v} outside [0,1]"
        vals.append(min(1.0, max(0.0, v)))
    grid = [vals[i * 5:(i + 1) * 5] for i in range(T)]
    return grid, None


def simulate(T, A0, L0, c_base, boundaries, flex, blocks, grid):
    scores = []
    for block in blocks:
        A = A0
        L = L0
        for t in range(1, T + 1):
            r_risky, r_safe, dr, g = block[t - 1]
            fr_prev = A / L
            b = bucket(fr_prev, boundaries)
            C = c_base * flex[b]
            w = grid[t - 1][b]
            remaining = T - t + 1
            mult = 1.0 + g - remaining * dr
            mult = min(1.5, max(0.5, mult))
            L = L * mult
            invested_base = A + C
            A = invested_base * (1.0 + w * r_risky + (1.0 - w) * r_safe)
        fr_final = A / L
        if fr_final >= 1.0:
            s = min(1.5, fr_final)
        else:
            s = fr_final * fr_final
        scores.append(s)
    return sum(scores) / len(scores)


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0 (bad invocation)")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    T, M, A0, L0, c_base, boundaries, flex, blocks = read_instance(in_path)

    grid, reason = read_grid(out_path, T)
    if grid is None:
        print(f"Ratio: 0.0 (infeasible output: {reason})")
        return 0

    F = simulate(T, A0, L0, c_base, boundaries, flex, blocks, grid)

    baseline_grid = [[0.0] * 5 for _ in range(T)]
    B = simulate(T, A0, L0, c_base, boundaries, flex, blocks, baseline_grid)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
