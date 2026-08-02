#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_1170 (thermal-source-backtrack)

Forward-simulates the participant's guessed point sources with the exact
recipe stated in statement.md, compares the resulting boundary trace to the
observed trace, and normalizes against a fixed data-independent reference
guess. Deterministic, O(N^2*T) work, no wall-time / randomness in scoring.
"""
import sys, math

MAX_TOKENS = 4000
ALPHA = 0.20          # regularization margin: keeps a lucky exact fit < 1.0
CORNER_X, CORNER_Y, CORNER_T0 = 1, 1, 0


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


def simulate_boundary_trace(sources_int, N, T, r, amp, bcells):
    """sources_int: list of (x, y, t0) with INTEGER x,y (participant / fixed
    reference guesses land here). Returns list of T+1 rows of M floats."""
    u = [[0.0] * N for _ in range(N)]
    srcs_sorted = sorted(sources_int, key=lambda s: (s[2], s[0], s[1]))
    trace = []
    for t in range(T + 1):
        for (x, y, t0) in srcs_sorted:
            if t0 == t:
                u[x][y] += amp
        trace.append([u[i][j] for (i, j) in bcells])
        if t < T:
            lap = laplacian(u, N)
            u = [[u[i][j] + r * lap[i][j] for j in range(N)] for i in range(N)]
    return trace


def mismatch(a, b):
    s = 0.0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            d = va - vb
            s += d * d
    return s


def read_instance(path):
    toks = open(path).read().split()
    ptr = 0
    N = int(toks[ptr]); ptr += 1
    T = int(toks[ptr]); ptr += 1
    K = int(toks[ptr]); ptr += 1
    r = float(toks[ptr]); ptr += 1
    A = float(toks[ptr]); ptr += 1
    M = 4 * N - 4
    obs = []
    for t in range(T + 1):
        row = [float(toks[ptr + c]) for c in range(M)]
        ptr += M
        obs.append(row)
    return N, T, K, r, A, obs


def parse_guess(text, K, N, T):
    """Return (list of (x,y,t0) ints, reason) or (None, reason) on failure."""
    toks = text.split()
    if len(toks) == 0:
        return None, "empty output"
    if len(toks) > MAX_TOKENS:
        return None, "too many tokens"
    if len(toks) != 3 * K:
        return None, f"expected {3*K} tokens, got {len(toks)}"
    vals = []
    for tok in toks:
        try:
            v = float(tok)
        except ValueError:
            return None, "non-numeric token"
        if not math.isfinite(v):
            return None, "non-finite token (nan/inf)"
        if abs(v - round(v)) > 1e-9:
            return None, "non-integer token"
        vals.append(int(round(v)))
    srcs = []
    for k in range(K):
        x, y, t0 = vals[3 * k], vals[3 * k + 1], vals[3 * k + 2]
        if not (1 <= x <= N - 2 and 1 <= y <= N - 2):
            return None, f"source {k}: (x,y)=({x},{y}) outside interior [1,{N-2}]"
        if not (0 <= t0 <= T - 1):
            return None, f"source {k}: t0={t0} outside [0,{T-1}]"
        srcs.append((x, y, t0))
    return srcs, "ok"


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0 (bad args)")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, T, K, r, A, obs = read_instance(in_path)
    bcells = ring_cells(N)

    try:
        out_text = open(out_path).read()
    except Exception:
        print("Ratio: 0.0 (cannot read output)")
        return 0

    srcs, reason = parse_guess(out_text, K, N, T)
    if srcs is None:
        print(f"Ratio: 0.0 (infeasible: {reason})")
        return 0

    pred = simulate_boundary_trace(srcs, N, T, r, A, bcells)
    F = mismatch(pred, obs)

    base_srcs = [(CORNER_X, CORNER_Y, CORNER_T0)] * K
    base_pred = simulate_boundary_trace(base_srcs, N, T, r, A, bcells)
    B = mismatch(base_pred, obs)
    if B <= 0.0:
        B = 1e-9

    F_eff = F + ALPHA * B
    sc = min(1000.0, 100.0 * B / max(1e-9, F_eff))
    ratio = sc / 1000.0
    print(f"F={F:.6f} B={B:.6f} Ratio: {ratio:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
