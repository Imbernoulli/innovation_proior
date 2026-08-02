#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic scorer for the cell-layout / interleave-
distance / decoder-cost problem.

Feasibility (hard gate, any violation -> Ratio: 0.0):
  1. <out> parses to a well-formed partition of the N physical cells into codewords,
     each codeword tagged with a valid catalog code index whose word length matches the
     codeword's cell count.
  2. EVERY physically-contiguous multi-cell upset burst the statement promises to test
     (every window length Len in 1..LMAX, every starting offset) is corrected: for every
     such window, no codeword receives more upset cells from that window than its
     correction capability t. This is checked EXHAUSTIVELY (not sampled), so it is the
     exact analogue of Format D's "exact equivalence to the target" gate.

Objective (only reached if feasible): total decoder cost = sum of the chosen codewords'
`cost` fields (an abstract op-count / decoder-latency surrogate). Lower is better.
"""
import sys


def die0(msg):
    print(f"Infeasible: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints_safe(tokens, need, ctx):
    """Pop `need` tokens off the front of `tokens`, parse as int, reject non-finite/garbage."""
    if len(tokens) < need:
        die0(f"not enough tokens ({ctx})")
    out = []
    for _ in range(need):
        tok = tokens.pop(0)
        try:
            v = int(tok)
        except ValueError:
            try:
                fv = float(tok)
            except ValueError:
                die0(f"non-integer token {tok!r} ({ctx})")
            if fv != fv or fv in (float("inf"), float("-inf")):
                die0(f"non-finite token {tok!r} ({ctx})")
            die0(f"non-integer token {tok!r} ({ctx})")
        out.append(v)
    return out


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        itoks = f.read().split()
    N, M, LMAX = int(itoks[0]), int(itoks[1]), int(itoks[2])
    p = 3
    catalog = []  # (w, t, cost)
    for i in range(M):
        w, t, cost = int(itoks[p]), int(itoks[p + 1]), int(itoks[p + 2])
        p += 3
        catalog.append((w, t, cost))

    # ---- checker's own internal baseline: the "obvious, no-search" construction ----
    # contiguous blocking using the LARGEST word length offered that is strictly smaller
    # than N (i.e. NOT the dedicated whole-row safe option), with whatever correction
    # strength that plain blocking forces.
    menu_w = sorted(set(w for (w, t, c) in catalog if w < N))
    if not menu_w:
        die0("instance malformed: no sub-N catalog word length (generator bug)")
    w_top = menu_w[-1]
    req_t_base = min(LMAX, w_top)
    base_entry_cost = None
    for (w, t, c) in catalog:
        if w == w_top and t >= req_t_base:
            if base_entry_cost is None or c < base_entry_cost:
                base_entry_cost = c
    if base_entry_cost is None:
        die0("instance malformed: no valid baseline code option (generator bug)")
    baseline_cost = (N // w_top) * base_entry_cost
    if baseline_cost <= 0:
        die0("instance malformed: non-positive baseline (generator bug)")

    # ---- parse participant output (untrusted; bounded, finite-checked) ----
    try:
        raw = open(outf).read()
    except Exception:
        die0("cannot read output")
    otoks = raw.split()
    if len(otoks) == 0:
        die0("empty output")
    if len(otoks) > 4 * N + 100:
        die0("output too large (token flood)")

    q = list(otoks)
    (B,) = read_ints_safe(q, 1, "codeword count B")
    if B < 1 or B > N:
        die0(f"B={B} out of range [1,{N}]")

    assign = [-1] * N          # cell -> codeword id
    cw_code_idx = []           # codeword id -> catalog index
    for b in range(B):
        (code_idx,) = read_ints_safe(q, 1, f"codeword {b} code index")
        if code_idx < 0 or code_idx >= M:
            die0(f"codeword {b}: code index {code_idx} out of range [0,{M})")
        w, t, cost = catalog[code_idx]
        cells = read_ints_safe(q, w, f"codeword {b} cell list (w={w})")
        for c in cells:
            if c < 0 or c >= N:
                die0(f"codeword {b}: cell {c} out of range [0,{N})")
            if assign[c] != -1:
                die0(f"cell {c} assigned to more than one codeword")
            assign[c] = b
        cw_code_idx.append(code_idx)

    if any(a == -1 for a in assign):
        missing = sum(1 for a in assign if a == -1)
        die0(f"{missing} cell(s) never assigned to any codeword")

    cw_t = [catalog[ci][1] for ci in cw_code_idx]

    # ---- exhaustive multi-bit-upset-geometry correctness check ----
    for Len in range(1, LMAX + 1):
        for s in range(0, N - Len + 1):
            cnt = {}
            for c in range(s, s + Len):
                b = assign[c]
                cnt[b] = cnt.get(b, 0) + 1
            for b, ct in cnt.items():
                if ct > cw_t[b]:
                    die0(f"uncorrected burst: cells [{s},{s+Len-1}) put {ct} upset cells "
                         f"in codeword {b} (capability t={cw_t[b]})")

    # ---- objective: total decoder cost (lower is better) ----
    F = sum(catalog[ci][2] for ci in cw_code_idx)
    if F <= 0:
        die0("non-positive objective (should be impossible)")

    sc = min(1000.0, 100.0 * baseline_cost / max(1e-9, F))
    print(f"N={N} B={B} baseline_cost={baseline_cost} F={F}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
