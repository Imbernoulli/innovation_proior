#!/usr/bin/env python3
"""
verify.py <in> <out> <ans> -- fsx_B_0755 checker.

Validates feasibility (N integers, finite, sum|h_j| <= B) then scores the
worst-case magnitude-response gap over the PASS/STOP bins against the
do-nothing (all-zero stencil) baseline. DONTCARE bins are never scored.
Always prints exactly one final "Ratio: <float>" line and exits 0.
"""
import sys
import math


def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    idx = 0
    N = int(toks[idx]); idx += 1
    B = int(toks[idx]); idx += 1
    M = int(toks[idx]); idx += 1
    roles = []
    for _ in range(M):
        k = int(toks[idx]); idx += 1
        typ = toks[idx]; idx += 1
        T = float(toks[idx]); idx += 1
        roles.append((k, typ, T))
    return N, B, roles


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0  # bad invocation")
        return 0

    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        N, B, roles = read_input(in_path)
    except Exception as e:
        print("Ratio: 0.0  # bad input file (%s)" % e)
        return 0

    try:
        with open(out_path) as f:
            raw = f.read()
    except Exception as e:
        print("Ratio: 0.0  # cannot read output (%s)" % e)
        return 0

    toks = raw.split()
    if len(toks) != N:
        print("Ratio: 0.0  # expected %d integers, got %d" % (N, len(toks)))
        return 0

    h = []
    for t in toks:
        # reject anything that is not a plain base-10 integer token (blocks
        # 'inf'/'nan'/'1e9'-style floats masquerading as ints, and huge blobs)
        if len(t) > 20:
            print("Ratio: 0.0  # token too long")
            return 0
        try:
            v = int(t)
        except ValueError:
            print("Ratio: 0.0  # non-integer token %r" % t)
            return 0
        if not math.isfinite(v):
            print("Ratio: 0.0  # non-finite")
            return 0
        if abs(v) > 10 ** 9:
            print("Ratio: 0.0  # coefficient out of range")
            return 0
        h.append(v)

    l1 = sum(abs(x) for x in h)
    if l1 > B:
        print("Ratio: 0.0  # L1 budget exceeded (%d > %d)" % (l1, B))
        return 0

    two_pi_over_N = 2.0 * math.pi / N
    dev = 0.0
    pass_targets = []
    # precompute nonzero-coefficient (index, value) pairs once -- O(N)
    nz = [(j, hj) for j, hj in enumerate(h) if hj != 0]

    for k, typ, T in roles:
        if typ == 'D':
            continue
        re = 0.0
        im = 0.0
        for j, hj in nz:
            ang = two_pi_over_N * j * k
            re += hj * math.cos(ang)
            im -= hj * math.sin(ang)
        mag = math.hypot(re, im)
        d = abs(mag - T)
        if d > dev:
            dev = d
        if typ == 'P':
            pass_targets.append(T)

    baseline = max(pass_targets) if pass_targets else 1.0
    F = dev
    if not math.isfinite(F):
        print("Ratio: 0.0  # non-finite deviation")
        return 0

    sc = min(1000.0, 100.0 * baseline / max(1e-9, F))
    print("carving-radio-filter dev=%.6f baseline=%.6f Ratio: %.6f" % (F, baseline, sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
