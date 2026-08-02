#!/usr/bin/env python3
"""gen.py <testId> -- fsx_B_1170 (thermal-source-backtrack)
Prints ONE instance to stdout: a boundary-only thermocouple trace of a
plate heated by K hidden point sources, for the given testId (1..10).
Deterministic: all randomness is seeded from testId alone.
"""
import sys, math, random

CASES = [
    # (N, T, K, r)
    (8, 10, 1, 0.05),
    (9, 12, 1, 0.06),
    (10, 14, 1, 0.08),
    (11, 15, 2, 0.10),
    (12, 17, 2, 0.12),
    (12, 19, 2, 0.15),
    (14, 20, 2, 0.18),
    (14, 22, 3, 0.20),
    (16, 25, 3, 0.22),
    (18, 28, 3, 0.24),
]

A = 8.0


def ring_cells(N):
    cells = []
    for i in range(N):
        for j in range(N):
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                cells.append((i, j))
    return cells


def laplacian(u, N):
    out = [[0.0] * N for _ in range(N)]
    for i in range(N):
        up_row = u[i - 1] if i - 1 >= 0 else u[i]
        down_row = u[i + 1] if i + 1 < N else u[i]
        row = u[i]
        for j in range(N):
            up = up_row[j]
            down = down_row[j]
            left = row[j - 1] if j - 1 >= 0 else row[j]
            right = row[j + 1] if j + 1 < N else row[j]
            out[i][j] = up + down + left + right - 4.0 * row[j]
    return out


def deposit(u, px, py, amp):
    i0 = int(math.floor(px))
    j0 = int(math.floor(py))
    fx = px - i0
    fy = py - j0
    for di, dj, w in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, (1 - fx) * fy),
                      (1, 0, fx * (1 - fy)), (1, 1, fx * fy)):
        u[i0 + di][j0 + dj] += amp * w


def simulate_boundary_trace(sources, N, T, r, amp, bcells):
    """sources: list of (px, py, t0) with px,py possibly fractional (true
    generation); returns list of T+1 rows, each a list of M floats."""
    u = [[0.0] * N for _ in range(N)]
    srcs_sorted = sorted(sources, key=lambda s: (s[2], s[0], s[1]))
    trace = []
    for t in range(T + 1):
        for (px, py, t0) in srcs_sorted:
            if t0 == t:
                deposit(u, px, py, amp)
        trace.append([u[i][j] for (i, j) in bcells])
        if t < T:
            lap = laplacian(u, N)
            u = [[u[i][j] + r * lap[i][j] for j in range(N)] for i in range(N)]
    return trace


def gen_true_sources(rng, N, K, T):
    lo, hi = 1.6, N - 2.6
    srcs = []
    for _ in range(K):
        px = rng.uniform(lo, hi)
        py = rng.uniform(lo, hi)
        t0 = rng.randint(0, max(1, (2 * T) // 3))
        srcs.append((px, py, t0))
    return srcs


def main():
    testId = int(sys.argv[1])
    N, T, K, r = CASES[testId - 1]
    rng = random.Random(20260726 + 977 * testId)
    true_srcs = gen_true_sources(rng, N, K, T)
    bcells = ring_cells(N)
    trace = simulate_boundary_trace(true_srcs, N, T, r, A, bcells)

    out = []
    out.append(f"{N} {T} {K} {r:.6f} {A:.6f}")
    for row in trace:
        out.append(" ".join(f"{v:.8f}" for v in row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
