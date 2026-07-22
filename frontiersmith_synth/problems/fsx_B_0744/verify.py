#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (<ans> is an unused placeholder)

Reads the enzyme-staging instance, replays the participant's enzyme vector
through the EXACT steady-state solver (per-node bisection, deterministic),
scores relative flux error against the target plus a mild enzyme-cost
penalty, and prints "... Ratio: <float in [0,1]>" on the final line.
"""
import sys
import math

X0 = 20.0
EPS = 1e-9
EPS_V = 0.05
BETA = 0.15
SC_CAP = 915.0


def fail(reason):
    print("INVALID: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_floats(toks):
    out = []
    for tk in toks:
        try:
            v = float(tk)
        except ValueError:
            return None
        if v != v or v in (float("inf"), float("-inf")):
            return None
        out.append(v)
    return out


def simulate(R, parent, yield_, kcat, Km, tau, e):
    children = [[] for _ in range(R + 1)]
    for j in range(R):
        children[parent[j]].append(j)
    x = [0.0] * R
    for i in range(R):
        xp_val = X0 if parent[i] == 0 else x[parent[i] - 1]
        v_i = e[i] * kcat[i] * xp_val / (Km[i] + xp_val)
        C = yield_[i] * v_i
        if C <= 0.0:
            x[i] = 0.0
            continue
        kids = children[i + 1]

        def Sfun(xi, kids=kids):
            s = 0.0
            for k in kids:
                s += e[k] * kcat[k] * xi / (Km[k] + xi)
            return s

        lo, hi = 0.0, tau[i] * C
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            val = mid - tau[i] * max(C - Sfun(mid), 0.0)
            if val < 0:
                lo = mid
            else:
                hi = mid
        x[i] = 0.5 * (lo + hi)
    xx = [X0] + x
    v = [0.0] * R
    for j in range(R):
        xp = xx[parent[j]]
        v[j] = e[j] * kcat[j] * xp / (Km[j] + xp)
    return x, v


def objective(R, parent, yield_, kcat, Km, tau, cost, v_target, e):
    _, v_ss = simulate(R, parent, yield_, kcat, Km, tau, e)
    E = sum(abs(v_ss[j] - v_target[j]) / (abs(v_target[j]) + EPS_V) for j in range(R)) / R
    C = sum(cost[j] * e[j] for j in range(R)) / R
    return E + BETA * C


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r") as f:
        in_toks = f.read().split()
    ptr = 0
    R = int(in_toks[ptr]); ptr += 1
    ptr += 1  # X0 (fixed constant, not needed beyond the module-level X0)
    parent = [0] * R
    yield_ = [0] * R
    kcat = [0.0] * R
    Km = [0.0] * R
    tau = [0.0] * R
    e_max = [0.0] * R
    cost = [0.0] * R
    for i in range(R):
        parent[i] = int(in_toks[ptr]); ptr += 1
        yield_[i] = int(in_toks[ptr]); ptr += 1
        kcat[i] = float(in_toks[ptr]); ptr += 1
        Km[i] = float(in_toks[ptr]); ptr += 1
        tau[i] = float(in_toks[ptr]); ptr += 1
        e_max[i] = float(in_toks[ptr]); ptr += 1
        cost[i] = float(in_toks[ptr]); ptr += 1
        if not (0 <= parent[i] <= i):
            fail("corrupt input instance (bad parent index)")
    ptr += R  # x_ref line, informational only, not needed by the checker
    v_target = [float(in_toks[ptr + k]) for k in range(R)]
    ptr += R

    if any(v < -1e-9 for v in v_target):
        fail("corrupt input instance (negative target flux)")

    # --- internal baseline B: e_i = e_max_i / 2 for every reaction ---------
    e_base = [e_max[i] / 2.0 for i in range(R)]
    B = objective(R, parent, yield_, kcat, Km, tau, cost, v_target, e_base)
    if B <= 0:
        fail("degenerate instance")

    # --- read participant output defensively --------------------------------
    try:
        with open(out_path, "r") as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")

    toks = raw.split()
    if len(toks) != R:
        fail(f"expected exactly {R} numbers, got {len(toks)}")
    e_sub = read_floats(toks)
    if e_sub is None:
        fail("non-finite or non-numeric token in output")

    for i in range(R):
        if e_sub[i] < -1e-6 or e_sub[i] > e_max[i] + 1e-6:
            fail(f"enzyme level {i+1} out of bounds [0,{e_max[i]}]: {e_sub[i]}")
        if e_sub[i] < 0.0:
            e_sub[i] = 0.0

    F = objective(R, parent, yield_, kcat, Km, tau, cost, v_target, e_sub)
    if F <= 0:
        fail("degenerate cost")

    sc = min(SC_CAP, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f" % (F, B))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
