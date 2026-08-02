# TIER: strong
"""The insight: forward simulation of the (well-posed, stable) heat
equation is cheap and numerically safe regardless of r, so reformulate
"invert the diffusion" as "search over candidate forward simulations".
Because the equation is linear, a single point source's contribution to
the boundary trace only depends on (position, remaining-time), so we
precompute one forward simulation per candidate interior position (a unit
impulse fired at t0=0) and reuse it for every onset time by reading off
the row at index (T - t0) -- no re-simulation needed per (position, t0)
pair. We then run a greedy matching-pursuit: repeatedly pick the
(position, t0) whose predicted boundary contribution reduces the residual
sum of squares the most, subtract it, and repeat for the K sources. This
never touches the unstable backward recursion at all.
"""
import sys
import numpy as np


def ring_cells(N):
    cells = []
    for i in range(N):
        for j in range(N):
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                cells.append((i, j))
    return cells


def laplacian(u):
    up = np.pad(u, 1, mode="edge")
    return up[:-2, 1:-1] + up[2:, 1:-1] + up[1:-1, :-2] + up[1:-1, 2:] - 4 * u


def read_input():
    toks = sys.stdin.read().split()
    ptr = 0
    N = int(toks[ptr]); ptr += 1
    T = int(toks[ptr]); ptr += 1
    K = int(toks[ptr]); ptr += 1
    r = float(toks[ptr]); ptr += 1
    A = float(toks[ptr]); ptr += 1
    M = 4 * N - 4
    obs = np.zeros((T + 1, M))
    for t in range(T + 1):
        obs[t] = [float(toks[ptr + c]) for c in range(M)]
        ptr += M
    return N, T, K, r, A, obs


def simulate_unit_impulse(i, j, N, T, r, A, bcells):
    """Boundary trace of a single source A at (i,j), onset t0=0."""
    u = np.zeros((N, N))
    u[i, j] += A
    trace = np.zeros((T + 1, len(bcells)))
    for t in range(T + 1):
        for k, (bi, bj) in enumerate(bcells):
            trace[t, k] = u[bi, bj]
        if t < T:
            u = u + r * laplacian(u)
    return trace


def main():
    N, T, K, r, A, obs = read_input()
    bcells = ring_cells(N)

    responses = {}
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            responses[(i, j)] = simulate_unit_impulse(i, j, N, T, r, A, bcells)

    residual = obs.copy()
    picks = []
    for _ in range(K):
        best = None
        for (i, j), R in responses.items():
            for t0 in range(T):
                shifted = np.zeros_like(R)
                shifted[t0:] = R[: T + 1 - t0]
                dot = float(np.sum(residual * shifted))
                energy = float(np.sum(shifted * shifted))
                score = 2 * dot - energy
                if best is None or score > best[0]:
                    best = (score, i, j, t0, shifted)
        _, i, j, t0, shifted = best
        picks.append((i, j, t0))
        residual = residual - shifted

    out = "\n".join(f"{i} {j} {t0}" for (i, j, t0) in picks)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
