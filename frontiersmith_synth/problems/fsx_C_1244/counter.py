#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic checker for the row-placement /
channel-congestion / timing problem. Prints '... Ratio: <float in [0,1]>' and exits 0.
"""
import sys

MAX_TOKENS = 2_000_000


def infeasible(reason):
    print(f"Infeasible: {reason} Ratio: 0.0")
    sys.exit(0)


def read_input(path):
    with open(path, "r") as f:
        toks = f.read().split()
    it = iter(toks)

    def nxt():
        return next(it)

    n_cells = int(nxt())
    n_nets = int(nxt())
    capacity = [int(nxt()) for _ in range(max(n_cells - 1, 0))]
    nets = []  # (terminals, is_crit, slack)
    for _ in range(n_nets):
        k = int(nxt())
        crit = int(nxt())
        slack = int(nxt())
        terms = [int(nxt()) for _ in range(k)]
        nets.append((terms, crit == 1, slack))
    return n_cells, n_nets, capacity, nets


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    n_cells, n_nets, capacity, nets = read_input(in_path)

    # ---- read participant output: bounded, strict, integer-only ----
    try:
        with open(out_path, "r") as f:
            blob = f.read(MAX_TOKENS + 1)
    except FileNotFoundError:
        infeasible("no output file.")

    raw_toks = blob.split()
    if len(raw_toks) != n_cells:
        infeasible(f"expected exactly {n_cells} tokens, got {len(raw_toks)}.")

    pos = []
    for t in raw_toks:
        try:
            v = int(t)
        except ValueError:
            infeasible(f"non-integer token '{t[:30]}'.")
        pos.append(v)

    if any(v < 0 or v >= n_cells for v in pos):
        infeasible("position out of range [0, n_cells-1].")
    if len(set(pos)) != n_cells:
        infeasible("output is not a permutation (duplicate slot).")

    # pos[i] = slot assigned to cell i
    spans = [0] * n_nets
    usage = [0] * max(n_cells - 1, 0)
    for idx, (terms, crit, slack) in enumerate(nets):
        slots = [pos[c] for c in terms]
        lo_s, hi_s = min(slots), max(slots)
        spans[idx] = hi_s - lo_s
        if crit and spans[idx] > slack:
            infeasible(f"timing-critical net {idx} span {spans[idx]} exceeds slack {slack}.")
        for g in range(lo_s, hi_s):
            usage[g] += 1

    for g, u in enumerate(usage):
        if u > capacity[g]:
            infeasible(f"channel {g} usage {u} exceeds capacity {capacity[g]}.")

    total_wirelength = sum(spans)  # F: objective to MINIMIZE

    # ---- internal baseline B: identity placement (pos[i] = i), recomputed from the
    #      input alone (independent of the generator's internal state; always feasible
    #      by construction of gen.py, but the checker does not assume that -- it simply
    #      uses identity's total span as the fixed reference cost). ----
    baseline_wirelength = 0
    for terms, crit, slack in nets:
        lo_c, hi_c = min(terms), max(terms)
        baseline_wirelength += (hi_c - lo_c)
    baseline_wirelength = max(baseline_wirelength, 1)

    sc = min(1000.0, 100.0 * baseline_wirelength / max(1e-9, total_wirelength))
    ratio = sc / 1000.0
    print("F=%d B=%d Ratio: %.6f" % (total_wirelength, baseline_wirelength, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
