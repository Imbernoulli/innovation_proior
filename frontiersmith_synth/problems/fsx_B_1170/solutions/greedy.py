# TIER: greedy
"""The obvious first attempt: literally run the heat equation BACKWARD in
time from the observed boundary trace, forcing the known boundary values
at every reversed step, and pick the K (position,time) cells where the
reconstructed field peaks. A mild spatial smoothing is layered on (a
plausible ad hoc stabilization) but the reversal is still the textbook
anti-diffusion update, so it is only mildly unstable on the easy/low-r
cases and it visibly blows up (exponential amplification per step) on the
higher-r / longer-T cases -- exactly the ill-posedness the problem is
about.
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


def box_blur(u):
    up = np.pad(u, 1, mode="edge")
    return 0.2 * (up[:-2, 1:-1] + up[2:, 1:-1] + up[1:-1, :-2] + up[1:-1, 2:]) + 0.2 * u


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


def main():
    N, T, K, r, A, obs = read_input()
    bcells = ring_cells(N)
    smooth_eps = 0.35

    u = np.zeros((N, N))
    for k, (i, j) in enumerate(bcells):
        u[i, j] = obs[T, k]
    frames = {T: u.copy()}
    for t in range(T - 1, -1, -1):
        u = u - r * laplacian(u)
        blurred = box_blur(u)
        u = (1 - smooth_eps) * u + smooth_eps * blurred
        for k, (i, j) in enumerate(bcells):
            u[i, j] = obs[t, k]
        u = np.clip(u, -1e6, 1e6)
        frames[t] = u.copy()

    cands = []
    for t in range(0, T):
        uu = frames[t]
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                v = uu[i, j]
                if np.isfinite(v):
                    cands.append((abs(v), i, j, t))
    cands.sort(reverse=True, key=lambda c: c[0])

    picks = []
    used = set()
    for val, i, j, t in cands:
        key = (i // 2, j // 2, t // 2)
        if key in used:
            continue
        used.add(key)
        picks.append((i, j, t))
        if len(picks) == K:
            break
    while len(picks) < K:
        picks.append((N // 2, N // 2, 0))

    out = "\n".join(f"{i} {j} {t}" for (i, j, t) in picks)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
