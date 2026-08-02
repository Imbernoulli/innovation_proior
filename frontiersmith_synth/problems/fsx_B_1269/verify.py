#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_1269 (cross-border-tax-route).

Scores a routing plan (a single path from jurisdiction 0 to jurisdiction
n-1 through the given treaty-link graph) on:
  1. structural validity (every hop is an actual treaty link),
  2. the TIMING window (total holding periods along the whole path must
     land in [T_min, T_max] -- both "too fast" round-trips and "too slow"
     routes that miss the filing window are non-compliant),
  3. the path-level SUBSTANCE / anti-conduit test: the aggregate treaty
     benefit claimed by the whole path (sum of baseline-minus-actual rate
     over every hop) sets a substance bar that the sum of intermediate
     jurisdictions' (mismatch-adjusted) substance scores must clear.
Any violation -> Ratio: 0.0. A compliant plan is scored by its net-of-tax
value relative to an internal baseline (the fixed "backbone" route shipped
in the input, a real/high-substance-only route that is always compliant).
"""
import sys, math

MAX_TOKENS = 5000


def read_instance(path):
    toks = open(path).read().split()
    ptr = 0
    n = int(toks[ptr]); ptr += 1
    m = int(toks[ptr]); ptr += 1
    V0 = float(toks[ptr]); ptr += 1
    baseline_rate_bp = float(toks[ptr]); ptr += 1
    gamma = float(toks[ptr]); ptr += 1
    T_min = int(toks[ptr]); ptr += 1
    T_max = int(toks[ptr]); ptr += 1
    substance = [int(toks[ptr + i]) for i in range(n)]; ptr += n
    edges = {}
    for _ in range(m):
        u = int(toks[ptr]); v = int(toks[ptr + 1])
        rate_bp = int(toks[ptr + 2]); hold = int(toks[ptr + 3]); itype = int(toks[ptr + 4])
        ptr += 5
        edges[(u, v)] = (rate_bp, hold, itype)
    b = int(toks[ptr]); ptr += 1
    backbone = [int(toks[ptr + i]) for i in range(b)]; ptr += b
    return dict(n=n, m=m, V0=V0, baseline_rate_bp=baseline_rate_bp, gamma=gamma,
                T_min=T_min, T_max=T_max, substance=substance, edges=edges,
                backbone=backbone)


def parse_path(text, n):
    toks = text.split()
    if len(toks) == 0:
        return None, "empty output"
    if len(toks) > MAX_TOKENS:
        return None, "too many tokens"
    try:
        vals = [int(t) for t in toks]
    except ValueError:
        return None, "non-integer token (nan/inf/garbage)"
    k = vals[0]
    if k < 2:
        return None, "path must have at least 2 nodes (source, target)"
    if len(vals) != 1 + k:
        return None, "token count does not match declared path length"
    path = vals[1:]
    for x in path:
        if x < 0 or x >= n:
            return None, "node id out of range"
    if path[0] != 0:
        return None, "path must start at source (node 0)"
    if path[-1] != n - 1:
        return None, "path must end at target (node n-1)"
    if len(set(path)) != len(path):
        return None, "path revisits a jurisdiction"
    return path, "ok"


def evaluate(path, inst):
    """Return (feasible: bool, reason: str, net_value: float or None)."""
    edges = inst["edges"]
    n = inst["n"]
    hops = []
    for a, b in zip(path, path[1:]):
        e = edges.get((a, b))
        if e is None:
            return False, f"no treaty link {a}->{b}", None
        hops.append(e)

    total_time = sum(h[1] for h in hops)
    if not (inst["T_min"] <= total_time <= inst["T_max"]):
        return False, (f"timing mismatch: total holding period {total_time} outside "
                        f"compliance window [{inst['T_min']},{inst['T_max']}]"), None

    benefit_bp = sum(max(0.0, inst["baseline_rate_bp"] - h[0]) for h in hops)
    required = math.ceil(inst["gamma"] * benefit_bp / 10000.0)

    intermediates = path[1:-1]
    S = 0
    for i, node in enumerate(intermediates):
        eff = inst["substance"][node]
        type_in = hops[i][2]        # edge into this node
        type_out = hops[i + 1][2]   # edge out of this node
        if type_in != type_out:
            eff = eff // 2
        S += eff
    if S < required:
        return False, (f"insufficient aggregate substance for claimed treaty benefit "
                        f"(anti-conduit rule): have {S}, need {required}"), None

    net = inst["V0"]
    for h in hops:
        net *= (1.0 - h[0] / 10000.0)
    return True, "ok", net


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0 (bad invocation)")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    try:
        inst = read_instance(in_path)
    except Exception as e:
        print(f"Ratio: 0.0 (bad instance: {e})")
        return 0

    try:
        text = open(out_path).read()
    except Exception:
        text = ""

    path, reason = parse_path(text, inst["n"])
    if path is None:
        print(f"Ratio: 0.0 ({reason})")
        return 0

    try:
        feasible, reason, net = evaluate(path, inst)
    except Exception as e:
        print(f"Ratio: 0.0 (evaluation error: {e})")
        return 0

    if not feasible:
        print(f"Ratio: 0.0 ({reason})")
        return 0

    if not math.isfinite(net) or net <= 0:
        print("Ratio: 0.0 (non-finite or non-positive value)")
        return 0

    # internal baseline: the fixed, always-compliant backbone route
    bb_feasible, bb_reason, bb_net = evaluate(inst["backbone"], inst)
    if not bb_feasible or bb_net is None or bb_net <= 0:
        # should never happen by construction; fall back to a conservative constant
        B = inst["V0"] * (1.0 - inst["baseline_rate_bp"] / 10000.0)
    else:
        B = bb_net

    sc = min(1000.0, 100.0 * net / max(1e-9, B))
    print(f"routing net-of-tax value={net:.4f} baseline={B:.4f} Ratio: {sc/1000.0:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
