#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans is an unused placeholder, per format-C contract)

Deterministic scorer for "Beacon Hill" (crt-schedule-packing).

Instance: M n; n lines p_i d_i; M integers w[0..M-1].
Artifact: n integers o_i (phase offsets), 0 <= o_i < p_i.

Job i transmits during [o_i + k*p_i, o_i + k*p_i + d_i) for k = 0 .. M/p_i - 1
(the run may wrap around within its own period: residues (o_i + tau) mod p_i).
load(t) = number of jobs transmitting at instant t.
Waste  F = sum_{t in [0,M)}  w[t] * max(0, load(t) - 1)      (minimize)

Baseline B = F under the all-zero phase assignment (o_i = 0 for all i).
Ratio = min(1000, 100 * B / max(1e-9, F)) / 1000.

Pure integer arithmetic, O(M + total number of runs); bit-for-bit deterministic.
"""
import sys


def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    it = iter(toks)
    try:
        M = int(next(it))
        n = int(next(it))
        jobs = []
        for _ in range(n):
            p = int(next(it))
            d = int(next(it))
            jobs.append((p, d))
        w = []
        for _ in range(M):
            w.append(int(next(it)))
    except StopIteration:
        raise ValueError("truncated input")
    # structural sanity (generator guarantees these; checker enforces them so the
    # simulation below is always well-defined)
    if M <= 0 or n <= 0:
        raise ValueError("bad header")
    for (p, d) in jobs:
        if p <= 0 or d <= 0 or d >= p or M % p != 0:
            raise ValueError("bad job")
    return M, n, jobs, w


def waste(M, jobs, w, offsets):
    diff = [0] * (M + 1)
    for (p, d), o in zip(jobs, offsets):
        reps = M // p
        if o + d <= p:
            for k in range(reps):
                s = o + k * p
                diff[s] += 1
                diff[s + d] -= 1
        else:
            # run wraps inside its own period: [o, p) and [0, o + d - p)
            e2 = o + d - p
            for k in range(reps):
                s = k * p
                diff[s] += 1
                diff[s + e2] -= 1
                diff[s + o] += 1
                diff[s + p] -= 1
    load = 0
    F = 0
    for t in range(M):
        load += diff[t]
        if load >= 2:
            F += w[t] * (load - 1)
    return F


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        M, n, jobs, w = read_instance(in_path)
    except Exception as e:
        print(f"BAD_INPUT: {e}")
        print("Ratio: 0.0")
        sys.exit(0)

    try:
        with open(out_path, "r") as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(raw) != n:
        fail(f"expected exactly {n} tokens, got {len(raw)}")

    offsets = []
    for idx, tok in enumerate(raw):
        try:
            v = int(tok)  # rejects 'nan', 'inf', '3.5', '1e3', ...
        except ValueError:
            fail(f"non-integer token: {tok!r}")
        except OverflowError:
            fail(f"token out of representable range: {tok!r}")
        p_i = jobs[idx][0]
        if v < 0 or v >= p_i:
            fail(f"offset {v} out of range [0, {p_i}) for job {idx}")
        offsets.append(v)

    F = waste(M, jobs, w, offsets)
    B = waste(M, jobs, w, [0] * n)
    if B <= 0:
        fail("degenerate instance (zero baseline)")

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    print(f"waste F={F} baseline B={B}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
