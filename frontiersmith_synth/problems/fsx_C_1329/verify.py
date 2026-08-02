#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for zeolite-template-choose.
<ans> is an unused placeholder. Prints "Ratio: <float in [0,1]>" and exits 0.
"""
import sys
import math
import os

T_NORM = 100.0
PH_NORM = 3.0
Q_NORM = 2.0
F_NORM = 1.0
EPS = 1e-6
MAX_OUT_BYTES = 4096


def f_ideal(c):
    return 0.15 * c + 0.1


def fail(reason):
    print("Infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    K = int(nxt())
    c = int(nxt())
    D_target = float(nxt())
    q_target = float(nxt())
    Tf_lo = float(nxt())
    Tf_hi = float(nxt())
    pHf_lo = float(nxt())
    pHf_hi = float(nxt())
    w1 = float(nxt())
    w2 = float(nxt())
    w3 = float(nxt())
    templates = []
    for _ in range(K):
        s = float(nxt())
        q = float(nxt())
        fl = float(nxt())
        Tlo = float(nxt())
        Thi = float(nxt())
        pHlo = float(nxt())
        pHhi = float(nxt())
        Topt = float(nxt())
        pHopt = float(nxt())
        R = float(nxt())
        r = float(nxt())
        templates.append(dict(s=s, q=q, f=fl, Tlo=Tlo, Thi=Thi, pHlo=pHlo, pHhi=pHhi,
                               Topt=Topt, pHopt=pHopt, R=R, r=r))
    return dict(K=K, c=c, D_target=D_target, q_target=q_target,
                Tf_lo=Tf_lo, Tf_hi=Tf_hi, pHf_lo=pHf_lo, pHf_hi=pHf_hi,
                w1=w1, w2=w2, w3=w3, templates=templates)


def sdi(inst, t):
    D = inst["D_target"]
    qt = inst["q_target"]
    fid = f_ideal(inst["c"])
    size_m = max(0.0, 1.0 - abs(t["s"] - D) / D)
    charge_m = max(0.0, 1.0 - abs(t["q"] - qt) / Q_NORM)
    shape_m = max(0.0, 1.0 - abs(t["f"] - fid) / F_NORM)
    return inst["w1"] * size_m + inst["w2"] * charge_m + inst["w3"] * shape_m


def proximity(t, T, pH):
    dT = (T - t["Topt"]) / T_NORM
    dpH = (pH - t["pHopt"]) / PH_NORM
    dist = math.sqrt(dT * dT + dpH * dpH)
    return max(0.0, 1.0 - dist / t["R"])


def baseline(inst):
    """Internal baseline B: template 0 run at its own sweet spot (generator
    guarantees template 0's window contains the framework window, so this is
    always feasible and matches solutions/trivial.py exactly)."""
    t0 = inst["templates"][0]
    F = sdi(inst, t0) * proximity(t0, t0["Topt"], t0["pHopt"]) * (1.0 - t0["r"])
    return F


def main():
    if len(sys.argv) != 4:
        print("usage: verify.py <in> <out> <ans>", file=sys.stderr)
        sys.exit(1)
    in_path, out_path, _ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    inst = read_instance(in_path)

    if not os.path.exists(out_path) or os.path.getsize(out_path) > MAX_OUT_BYTES:
        fail("missing or oversized output")

    with open(out_path, "r") as f:
        content = f.read()
    toks = content.split()
    if len(toks) != 3:
        fail("expected exactly 3 tokens: idx T pH")

    idx_tok, T_tok, pH_tok = toks
    try:
        idx = int(idx_tok)
    except ValueError:
        fail("idx not an integer")
    try:
        T = float(T_tok)
        pH = float(pH_tok)
    except ValueError:
        fail("T/pH not floats")

    if not (math.isfinite(T) and math.isfinite(pH)):
        fail("non-finite T/pH")

    K = inst["K"]
    if not (0 <= idx < K):
        fail("idx out of range")

    t = inst["templates"][idx]
    Tf_lo, Tf_hi = inst["Tf_lo"], inst["Tf_hi"]
    pHf_lo, pHf_hi = inst["pHf_lo"], inst["pHf_hi"]

    if not (Tf_lo - EPS <= T <= Tf_hi + EPS):
        fail("T outside framework window")
    if not (t["Tlo"] - EPS <= T <= t["Thi"] + EPS):
        fail("T outside template window")
    if not (pHf_lo - EPS <= pH <= pHf_hi + EPS):
        fail("pH outside framework window")
    if not (t["pHlo"] - EPS <= pH <= t["pHhi"] + EPS):
        fail("pH outside template window")

    F = sdi(inst, t) * proximity(t, T, pH) * (1.0 - t["r"])
    B = baseline(inst)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f idx=%d" % (F, B, idx))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
