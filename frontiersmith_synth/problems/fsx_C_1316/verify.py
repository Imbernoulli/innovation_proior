#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the
substituent-property-tune problem. Prints 'Ratio: <float in [0,1]>' on its
own final line. Maximization objective: higher closeness-to-target is
better.
"""
import sys
import math


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = toks[pos]
        pos += 1
        return v

    N = int(nxt())
    K = int(nxt())
    budget = int(nxt())
    P0 = float(nxt())
    alpha = float(nxt())
    beta = float(nxt())
    s_thresh = float(nxt())
    target = float(nxt())
    window = float(nxt())
    lib = []
    for _ in range(K):
        e = float(nxt())
        s = float(nxt())
        c = int(nxt())
        lib.append((e, s, c))
    return N, K, budget, P0, alpha, beta, s_thresh, target, window, lib


def property_value(assign, N, lib, alpha, beta, s_thresh, P0):
    """assign[i] in {-1..K-1}; -1 means empty (H, no substituent)."""
    bulky = [s > s_thresh for (e, s, c) in lib]
    S = 0.0
    for i in range(N):
        t = assign[i]
        if t < 0:
            continue
        e, s, c = lib[t]
        S += e + alpha * s
    for i in range(N):
        j = (i + 1) % N
        ti, tj = assign[i], assign[j]
        if ti >= 0 and tj >= 0 and bulky[ti] and bulky[tj]:
            S -= beta * (lib[ti][1] + lib[tj][1])
    return P0 + S


def closeness(P, target, window):
    return 1.0 / (1.0 + abs(P - target) / window)


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    N, K, budget, P0, alpha, beta, s_thresh, target, window, lib = read_instance(inp)

    try:
        with open(outp) as f:
            toks = f.read().split()
    except Exception as e:
        fail("cannot read output: %s" % e)
        return

    if len(toks) != N:
        fail("expected exactly %d position assignments, got %d tokens" % (N, len(toks)))
        return

    assign = []
    for i, tok in enumerate(toks):
        try:
            v = int(tok)
        except ValueError:
            fail("token %d (%r) is not an integer" % (i, tok))
            return
        if not (0 <= v <= K):
            fail("position %d value %d out of range [0,%d] (0=H, 1..K=substituent index)" % (i, v, K))
            return
        assign.append(v - 1)  # shift: 0 -> -1 (empty), t -> t-1 (0-indexed type)

    total_cost = sum(lib[t][2] for t in assign if t >= 0)
    if total_cost > budget:
        fail("synthesis-step budget exceeded: used %d > budget %d" % (total_cost, budget))
        return

    P = property_value(assign, N, lib, alpha, beta, s_thresh, P0)
    if not math.isfinite(P):
        fail("non-finite property value")
        return

    F = closeness(P, target, window)
    if not math.isfinite(F) or F < 0:
        fail("non-finite or negative closeness")
        return

    # Internal reference baseline: the "do nothing" pattern (every position
    # left as H). Always feasible (cost 0 <= budget), always finite.
    empty_assign = [-1] * N
    P_ref = property_value(empty_assign, N, lib, alpha, beta, s_thresh, P0)
    F_ref = closeness(P_ref, target, window)

    sc = min(1000.0, 100.0 * F / max(1e-9, F_ref))
    print("P=%.6f target=%.6f window=%.6f F=%.6f F_ref=%.6f cost=%d/%d" %
          (P, target, window, F, F_ref, total_cost, budget))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
