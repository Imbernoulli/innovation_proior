#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic checker for instruction-set-select.

Feasibility: the submitted artifact is a set S of selected candidate instruction ids,
|S| <= K (encoding-space budget) and sum(area[c] for c in S) <= A (silicon budget), all
ids distinct and in range. On ANY violation (including non-finite/garbage tokens) prints
Ratio: 0.0.

Objective (maximize): recompile every application by a FIXED, deterministic single pass
per app -- occurrences of selected candidates are considered in the order
(start position asc, savings desc, size desc, candidate id asc) and an occurrence is
"applied" iff every code position it covers is still unclaimed; claiming it converts
`size[c]` base-cost positions (1 cycle each) into one fused op costing `cost[c]` cycles,
banking `size[c]-cost[c]` cycles of savings. F = total cycles saved, summed over all
applications. This models a realistic single-pass fusing compiler: it never needs to be
told which instructions win overlapping matches, that is a structural consequence of
which instructions you spent encoding-space/area on.

Baseline B: the same simulation run against S_triv = candidates picked in raw id order,
greedily added while they still fit both the K and A budgets (the "no ranking at all"
construction -- also what solutions/trivial.py implements).

Ratio (maximization) = min(1000, 100*F/max(1e-9,B)) / 1000.
"""
import sys

MAX_TOKENS = 2_000_000


def bail(msg):
    print("INVALID: " + msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints_line(line):
    return [int(x) for x in line.split()]


def main():
    if len(sys.argv) < 3:
        bail("bad invocation")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        in_lines = open(inf, "r").read().split("\n")
    except Exception:
        bail("cannot read input")

    idx = 0
    try:
        K, A = read_ints_line(in_lines[idx]); idx += 1
        C = int(in_lines[idx]); idx += 1
        area = [0] * C
        size = [0] * C
        cost = [0] * C
        for c in range(C):
            a, s, cst = read_ints_line(in_lines[idx]); idx += 1
            if a < 0 or s < 1 or cst < 1 or cst >= s:
                raise ValueError("bad candidate row")
            area[c], size[c], cost[c] = a, s, cst
        M = int(in_lines[idx]); idx += 1
        apps = []  # each: (L, [(cid,start,savings,size)...] sorted for the sweep)
        for m in range(M):
            L, O = read_ints_line(in_lines[idx]); idx += 1
            occs = []
            for _ in range(O):
                cid, start = read_ints_line(in_lines[idx]); idx += 1
                if not (0 <= cid < C):
                    raise ValueError("occurrence candidate id out of range")
                sv = size[cid] - cost[cid]
                if start < 0 or start + size[cid] > L:
                    raise ValueError("occurrence out of app bounds")
                occs.append((cid, start, sv, size[cid]))
            occs.sort(key=lambda o: (o[1], -o[2], -o[3], o[0]))
            apps.append((L, occs))
    except Exception:
        bail("malformed input (generator bug)")

    if K < 1 or C < 1 or M < 1:
        bail("degenerate input")

    def simulate(selected):
        total = 0
        for (L, occs) in apps:
            claimed = bytearray(L)
            for (cid, start, sv, sz) in occs:
                if cid not in selected:
                    continue
                free = True
                for p in range(start, start + sz):
                    if claimed[p]:
                        free = False
                        break
                if free:
                    for p in range(start, start + sz):
                        claimed[p] = 1
                    total += sv
        return total

    # ---- baseline: raw id-order first-fit under both budgets --------------
    triv = set()
    used_area = 0
    for c in range(C):
        if len(triv) >= K:
            break
        if used_area + area[c] <= A:
            triv.add(c)
            used_area += area[c]
    B = simulate(triv)

    # ---- read + validate participant output --------------------------------
    try:
        raw = open(outf, "r").read()
    except Exception:
        bail("cannot read output")
    toks = raw.split()
    if not toks:
        bail("empty output")
    if len(toks) > MAX_TOKENS:
        bail("output too large")

    def parse_int_strict(tok):
        # reject "nan"/"inf"/floats/garbage -- ids and counts must be plain ints
        if not (tok.lstrip("-").isdigit()):
            raise ValueError("non-integer token")
        return int(tok)

    try:
        S = parse_int_strict(toks[0])
    except Exception:
        bail("first token (count) is not an integer")
    if S < 0 or S > 5_000_000:
        bail("selection count out of range")
    if len(toks) < 1 + S:
        bail("insufficient id tokens for declared count")

    ids = []
    try:
        for i in range(S):
            v = parse_int_strict(toks[1 + i])
            ids.append(v)
    except Exception:
        bail("non-integer id token")

    if len(set(ids)) != len(ids):
        bail("duplicate candidate id")
    for v in ids:
        if not (0 <= v < C):
            bail(f"candidate id {v} out of range")
    if len(ids) > K:
        bail(f"selection size {len(ids)} exceeds encoding-space budget K={K}")
    sel_area = sum(area[v] for v in ids)
    if sel_area > A:
        bail(f"selection area {sel_area} exceeds area budget A={A}")

    selected = set(ids)
    F = simulate(selected)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print(f"K={K} A={A} C={C} |S|={len(ids)} area={sel_area} F={F} B={B}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
