#!/usr/bin/env python3
"""Deterministic checker for "Two-Plane Phase Plate" (format C, minimize).
CLI: python3 verify.py <in> <out> <ans>   (ans is an empty placeholder, ignored).
Prints "... Ratio: <r>" with r in [0,1] on its own final line; any feasibility
violation prints "Ratio: 0.0" and exits 0.
"""
import sys
import math
import numpy as np

MAX_PHASE = 1.0e6


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it))
    aA = float(next(it))
    aB = float(next(it))
    lam = float(next(it))
    TA = np.array([float(next(it)) for _ in range(N * N)], dtype=np.float64).reshape(N, N)
    TB = np.array([float(next(it)) for _ in range(N * N)], dtype=np.float64).reshape(N, N)
    return N, aA, aB, lam, TA, TB


def _chirp(N, alpha, sign):
    c = (N - 1) / 2.0
    ii, jj = np.indices((N, N))
    return np.exp(sign * 1j * alpha * ((ii - c) ** 2 + (jj - c) ** 2))


def _forward(theta, alpha):
    N = theta.shape[0]
    field = np.exp(1j * theta) * _chirp(N, alpha, 1.0)
    return np.fft.fft2(field, norm="ortho")


def _backward(F, alpha):
    N = F.shape[0]
    field = np.fft.ifft2(F, norm="ortho")
    return field * _chirp(N, alpha, -1.0)


def _nmse(theta, alpha, T):
    N = theta.shape[0]
    F = _forward(theta, alpha)
    P = (np.abs(F) ** 2) / float(N * N)          # sum(P) == 1 always (unitary FFT, |A0|=1)
    Q = T / T.sum()
    denom = float(np.sum(Q * Q))
    return float(np.sum((P - Q) ** 2)) / max(denom, 1e-12)


def _tv_penalty(theta):
    d1 = theta[1:, :] - theta[:-1, :]
    d2 = theta[:, 1:] - theta[:, :-1]
    w1 = np.arctan2(np.sin(d1), np.cos(d1))
    w2 = np.arctan2(np.sin(d2), np.cos(d2))
    total = float(np.sum(w1 ** 2) + np.sum(w2 ** 2))
    cnt = w1.size + w2.size
    return total / max(cnt, 1)


def objective(theta, TA, TB, aA, aB, lam):
    e = 0.5 * (_nmse(theta, aA, TA) + _nmse(theta, aB, TB))
    tv = _tv_penalty(theta)
    return e + lam * tv


def baseline_construction(N):
    """Trivial feasible construction the checker builds itself: an unmodulated
    (flat) phase plate. theta == 0 everywhere."""
    return np.zeros((N, N), dtype=np.float64)


def parse_answer(path, N):
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines()]
    if not lines:
        fail("empty output")
    toks0 = lines[0].split()
    if len(toks0) != 1:
        fail("first line must be a single integer N")
    try:
        n_out = int(toks0[0])
    except ValueError:
        fail("first token is not an integer")
    if n_out != N:
        fail("N mismatch: expected %d got %d" % (N, n_out))
    body = lines[1:]
    if len(body) < N:
        fail("expected %d phase rows, got %d" % (N, len(body)))
    theta = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        row = body[i].split()
        if len(row) != N:
            fail("row %d has %d tokens, expected %d" % (i, len(row), N))
        for j, tok in enumerate(row):
            try:
                v = float(tok)
            except ValueError:
                fail("non-numeric token at row %d col %d" % (i, j))
            if not math.isfinite(v):
                fail("non-finite value at row %d col %d" % (i, j))
            if abs(v) > MAX_PHASE:
                fail("phase magnitude too large at row %d col %d" % (i, j))
            theta[i, j] = v
    # any remaining non-blank lines are garbage / trailing junk
    for extra in body[N:]:
        if extra.strip():
            fail("trailing garbage after phase rows")
    return theta


def main():
    if len(sys.argv) != 4:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, aA, aB, lam, TA, TB = read_instance(in_path)
    theta = parse_answer(out_path, N)

    F = objective(theta, TA, TB, aA, aB, lam)
    if not math.isfinite(F) or F < 0:
        fail("non-finite or negative objective")

    B = objective(baseline_construction(N), TA, TB, aA, aB, lam)
    B = max(B, 1e-9)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    if not math.isfinite(ratio) or ratio < 0.0:
        fail("bad ratio computed")
    ratio = max(0.0, min(1.0, ratio))
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
