import sys, math

# ---- fixed audit postures: (weight_F, weight_A, weight_R) ----------------------------
# Nine canonical ways an audit can weight the Functions / Assets / Risks triad of a
# functional analysis. Deliberately NOT shown to the solver: the statement only says such
# postures exist and differ in emphasis.
SCENARIOS = [
    (5, 1, 1), (1, 5, 1), (1, 1, 5), (3, 3, 3), (5, 5, 1),
    (5, 1, 5), (1, 5, 5), (4, 2, 2), (2, 4, 2),
]

ADJ_RATE = 0.6          # bp-equivalent penalty per bp of forced adjustment
FLAG_BASE = 60.0        # flat compliance-risk cost when outside range, damped by documentation
INSUFF_FLAG = 900.0     # heavy fixed cost when the submitted support is too thin
MAXW = 80               # weight cap used when building the weighted reference range
THRESH_P = 0.35         # scrutiny-threshold percentile (nearest-rank) of the weighted distance


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def read_ints(tokens, n):
    vals = []
    for _ in range(n):
        if not tokens:
            raise ValueError("out of tokens")
        vals.append(int(tokens.pop(0)))
    return vals


def nearest_rank(sorted_vals_weights, p):
    """sorted_vals_weights: list of (val, weight) sorted by val ascending. Returns the
    value at the smallest cumulative weight >= ceil(p * total_weight) (nearest-rank
    percentile, integer arithmetic only)."""
    W = sum(w for _, w in sorted_vals_weights)
    if W <= 0:
        return None
    pos = math.ceil(p * W)
    pos = max(1, min(W, pos))
    c = 0
    for v, w in sorted_vals_weights:
        c += w
        if c >= pos:
            return v
    return sorted_vals_weights[-1][0]


def main():
    try:
        in_tokens = open(sys.argv[1]).read().split()
        N = int(in_tokens[0])
        rest = in_tokens[1:]
        f0, a0, r0 = read_ints(rest, 3)
        REV, BUDGET, MIN_COMPS = read_ints(rest, 3)
        if N <= 0 or MIN_COMPS <= 0 or MIN_COMPS > N:
            fail("bad instance")
        comps = []
        for _ in range(N):
            margin, f, a, r, doc_cost = read_ints(rest, 5)
            comps.append((margin, f, a, r, doc_cost))
    except Exception:
        fail("bad input")

    # ---- internal baseline B: no functional-distance reasoning at all -- just the
    # MIN_COMPS candidates whose margin is closest to the sample median margin (a stable,
    # unremarkable pick), no documentation, plain average of their margins ----
    margins_all = sorted(c[0] for c in comps)
    med = margins_all[len(margins_all) // 2]
    base_order = sorted(range(N), key=lambda i: (abs(comps[i][0] - med), i))
    base_chosen = [i + 1 for i in base_order[:MIN_COMPS]]
    base_margins = [comps[i1 - 1][0] for i1 in base_chosen]
    base_M = sum(base_margins) // len(base_margins)
    base_doc = {i1: 0 for i1 in base_chosen}

    # ---- parse participant output --------------------------------------------------
    out_tokens = open(sys.argv[2]).read().split()
    try:
        if not out_tokens:
            raise ValueError("empty")
        S = int(out_tokens[0])
    except Exception:
        fail("parse S")
    if S < 1 or S > N:
        fail("S out of range")

    pos_tok = 1
    selection = []
    doc_depth = {}
    seen = set()
    try:
        for _ in range(S):
            if pos_tok + 1 >= len(out_tokens):
                raise ValueError("truncated selection")
            i1 = int(out_tokens[pos_tok]); d = int(out_tokens[pos_tok + 1])
            pos_tok += 2
            if i1 < 1 or i1 > N:
                fail("index out of range %d" % i1)
            if i1 in seen:
                fail("duplicate index %d" % i1)
            if d < 0 or d > 3:
                fail("doc depth out of range %d" % d)
            seen.add(i1)
            selection.append(i1)
            doc_depth[i1] = d
        if pos_tok >= len(out_tokens):
            raise ValueError("missing M")
        M = int(out_tokens[pos_tok])
    except Exception as e:
        fail("parse selection/M (%s)" % e)

    if M < 0 or M > 10000:
        fail("declared margin out of range")

    # every SUBMITTED comparable carries a baseline documentation/admin cost even at
    # depth 0 (gathering and filing its data at all costs doc_cost_i); each extra depth
    # level costs the same unit again -- so cost = doc_cost_i * (depth_i + 1). This is
    # what stops "just submit every candidate" from being a free move.
    spend = sum(comps[i1 - 1][4] * (doc_depth[i1] + 1) for i1 in selection)
    if spend > BUDGET:
        fail("documentation budget exceeded (%d > %d)" % (spend, BUDGET))

    def score(sel, docs, decl_M):
        fallback_vals = sorted((c[0], 1) for c in comps)
        fb_margin = nearest_rank(fallback_vals, 0.5)
        profits = []
        for (wF, wA, wR) in SCENARIOS:
            true_dist_all = []
            for c in comps:
                _, f, a, r, _ = c
                true_dist_all.append(wF * abs(f - f0) + wA * abs(a - a0) + wR * abs(r - r0))
            THRESH = nearest_rank([(v, 1) for v in sorted(true_dist_all)], THRESH_P)
            GAP = max(1, THRESH // 3)

            ref_list = []
            for c in comps:
                margin, f, a, r, _ = c
                d = wF * abs(f - f0) + wA * abs(a - a0) + wR * abs(r - r0)
                w = max(1, MAXW - d)
                ref_list.append((margin, w))
            ref_list.sort(key=lambda t: t[0])
            REF_Q1 = nearest_rank(ref_list, 0.25)
            REF_Q3 = nearest_rank(ref_list, 0.75)

            n_support = 0
            for i1 in sel:
                margin, f, a, r, _ = comps[i1 - 1]
                d = wF * abs(f - f0) + wA * abs(a - a0) + wR * abs(r - r0)
                excess = d - THRESH
                req = 0 if excess <= 0 else min(3, 1 + (excess - 1) // GAP)
                if docs.get(i1, 0) >= req:
                    n_support += 1
            supported = n_support >= MIN_COMPS
            avg_doc = sum(docs.get(i1, 0) for i1 in sel) / max(1, len(sel))

            if not supported:
                adjusted = fb_margin
                penalty = ADJ_RATE * abs(decl_M - adjusted) + INSUFF_FLAG
            elif REF_Q1 <= decl_M <= REF_Q3:
                adjusted = decl_M
                penalty = 0.0
            else:
                adjusted = REF_Q1 if decl_M < REF_Q1 else REF_Q3
                penalty = ADJ_RATE * abs(decl_M - adjusted) + FLAG_BASE / (1 + avg_doc)

            profit = max(0.0, (adjusted - penalty)) * REV / 10000.0
            profits.append(profit)
        return sum(profits) / len(profits)

    B = score(base_chosen, base_doc, base_M)
    B = max(1e-9, B)
    F = score(selection, doc_depth, M)

    sc = min(1000.0, 100.0 * F / B)
    print("F=%.2f B=%.2f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
