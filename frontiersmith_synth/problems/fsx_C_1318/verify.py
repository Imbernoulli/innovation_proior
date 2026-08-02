#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for alloy-composition-search.

Feasibility (any violation -> Ratio: 0.0):
  - output is exactly K whitespace-separated tokens, each parseable as a base-10
    integer (nan/inf/garbage all fail to parse -> reject)
  - every x_i >= 0
  - X = sum(x_i) <= MAXX = numBins*W - 1
  - IM = sum(b_i * x_i) <= T[X // W]   (the phase-boundary / brittleness gate)

Objective (maximize): F = sum_i s_i * sqrt(x_i)   (solid-solution strengthening)

Baseline B (checker's own trivial construction, also reproduced by
solutions/trivial.py): try EVERY element alone (all others at zero), each
filled to the largest amount that is itself feasible among amounts confined
to the first three bands (X <= 3*W-1); B is the best single-element result.

Ratio = min(1000, 100*F/max(1e-9,B)) / 1000.
"""
import math
import sys


def fail(reason):
    print(f"# infeasible: {reason}")
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        toks = f.read().split()
    idx = 0
    K = int(toks[idx]); idx += 1
    W = int(toks[idx]); idx += 1
    numBins = int(toks[idx]); idx += 1
    s = [int(toks[idx + i]) for i in range(K)]; idx += K
    b = [int(toks[idx + i]) for i in range(K)]; idx += K
    T = [int(toks[idx + i]) for i in range(numBins)]; idx += numBins
    MAXX = numBins * W - 1

    try:
        with open(out_path) as f:
            out_toks = f.read().split()
    except Exception as e:
        fail(f"cannot read output ({e})")

    if len(out_toks) != K:
        fail(f"expected {K} integers, got {len(out_toks)}")

    x = []
    for t in out_toks:
        try:
            v = int(t)
        except ValueError:
            fail(f"non-integer token '{t[:40]}'")
        x.append(v)

    for i, v in enumerate(x):
        if v < 0:
            fail(f"negative composition amount x[{i}]={v}")

    X = sum(x)
    if X > MAXX:
        fail(f"total solute X={X} exceeds MAXX={MAXX}")

    binIdx = X // W
    if binIdx < 0 or binIdx >= numBins:
        fail(f"bin index {binIdx} out of range")

    IM = sum(b[i] * x[i] for i in range(K))
    cap = T[binIdx]
    if IM > cap:
        fail(f"brittle: IM={IM} exceeds phase-boundary budget T[{binIdx}]={cap} (X={X})")

    F = sum(s[i] * math.sqrt(x[i]) for i in range(K))
    if not math.isfinite(F):
        fail("non-finite objective")

    # internal baseline: best SINGLE element alone, each tried at its largest
    # feasible amount confined to the first three bands (X <= 3*W-1)
    limit_bin = min(numBins - 1, 2)
    max_bx = (limit_bin + 1) * W - 1
    B = -1.0
    for i in range(K):
        Bx_i = 0
        for cand in range(0, max_bx + 1):
            cbin = cand // W
            if b[i] * cand <= T[cbin]:
                Bx_i = cand
        val = s[i] * math.sqrt(Bx_i)
        if val > B:
            B = val

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print(f"# F={F:.6f} B={B:.6f} X={X} IM={IM} bin={binIdx}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
