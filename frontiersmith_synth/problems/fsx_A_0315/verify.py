#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer (minimization).

Reads the instance (N, ceilings u) from <in> and the participant profile f from <out>.
Strictly validates feasibility; on ANY violation prints `Ratio: 0.0` and exits 0.
Otherwise computes c1(f), an internal baseline B (ceilings on the first ceil(N/3)
segments), and prints Ratio = min(1.0, 0.1 * B / c1(f)).

Pure Python, O(N^2), bit-for-bit deterministic.
"""
import sys, math


def fail(reason):
    sys.stdout.write("infeasible (%s). Ratio: 0.0\n" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as fh:
        toks = fh.read().split()
    N = int(toks[0])
    u = [float(t) for t in toks[1:1 + N]]
    if len(u) != N:
        raise ValueError("bad instance")
    return N, u


def read_profile(path, N):
    # bounded, strict token read
    try:
        with open(path) as fh:
            data = fh.read(64 * 1024 * 1024)  # cap 64MB
    except Exception:
        fail("cannot read output")
    toks = data.split()
    if len(toks) != N:
        fail("expected %d numbers, got %d" % (N, len(toks)))
    f = []
    for t in toks:
        try:
            x = float(t)
        except Exception:
            fail("non-numeric token")
        if not math.isfinite(x):
            fail("non-finite value")
        f.append(x)
    return f


def c1(f, N):
    # self-convolution peak: G(k) = sum_{i+j=k} f_i f_j, k = 0..2N-2
    s = 0.0
    for x in f:
        s += x
    if s <= 1e-12:
        return None
    peak = 0.0
    for k in range(2 * N - 1):
        acc = 0.0
        lo = k - (N - 1)
        if lo < 0:
            lo = 0
        hi = k
        if hi > N - 1:
            hi = N - 1
        i = lo
        while i <= hi:
            acc += f[i] * f[k - i]
            i += 1
        if acc > peak:
            peak = acc
    return 2.0 * N * peak / (s * s)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    N, u = read_instance(inf)

    f = read_profile(outf, N)

    # feasibility: bounds + positive total
    tol = 1e-6
    for i in range(N):
        if f[i] < -tol:
            fail("negative intensity")
        if f[i] > u[i] + tol:
            fail("exceeds ceiling at segment %d" % i)
    if sum(f) <= 1e-12:
        fail("zero total light")

    F = c1(f, N)
    if F is None:
        fail("degenerate profile")

    # internal baseline: ceilings on the first ceil(N/3) segments, off elsewhere
    m = (N + 2) // 3
    base = [u[i] if i < m else 0.0 for i in range(N)]
    B = c1(base, N)
    if B is None or B <= 0:
        B = 6.0  # safety; baseline should always be positive

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    sys.stdout.write("c1=%.6f baseline=%.6f Ratio: %.6f\n" % (F, B, ratio))


if __name__ == "__main__":
    main()
