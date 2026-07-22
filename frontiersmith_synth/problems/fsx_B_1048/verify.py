#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for the moonshine-still cut-cascade problem.
Deterministic, O(K*G) after O(1)-per-query prefix sums. Prints 'Ratio: <float in [0,1]>'.
"""
import sys, math

EPS = 1e-9


def die(reason):
    print("INFEASIBLE:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    it = iter(toks)

    def nxt():
        try:
            return next(it)
        except StopIteration:
            die("truncated instance file")

    K = int(nxt()); G = int(nxt()); step1 = int(nxt())
    v = [float(nxt()) for _ in range(K)]
    m = [[float(nxt()) for _ in range(G)] for _ in range(K)]
    H = float(nxt()); energyCost = float(nxt())
    M_min = float(nxt()); cap_small = float(nxt())
    B = int(nxt())
    bands = [(float(nxt()), float(nxt())) for _ in range(B)]
    bands.sort(key=lambda t: t[0])
    return K, G, step1, v, m, H, energyCost, M_min, cap_small, bands


def band_mult(bands, purity):
    best = bands[0][1]
    for lo, mult in bands:
        if lo <= purity + 1e-12:
            best = mult
        else:
            break
    return best


def make_prefix(m, K, G):
    pref = [[0.0] * (G + 1) for _ in range(K)]
    for c in range(K):
        acc = 0.0
        row = m[c]
        prow = pref[c]
        for g in range(G):
            acc += row[g]
            prow[g + 1] = acc
    return pref


def seg_mass_purity(pref, K, a, b):
    masses = [pref[c][b] - pref[c][a] for c in range(K)]
    total = sum(masses)
    if total <= EPS:
        return 0.0, 0.0, 0
    dom = max(range(K), key=lambda c: masses[c])
    purity = masses[dom] / total
    return total, purity, dom


def seg_revenue(pref, v, bands, K, a, b, M_min=0.0, cap_small=1.0):
    """Purity-band price, with a batch-size qualifier: fractions with mass below
    M_min are capped at `cap_small` regardless of how pure they are (a small-lot
    penalty -- only a properly-sized batch clears certification for the full band)."""
    total, purity, dom = seg_mass_purity(pref, K, a, b)
    if total <= EPS:
        return 0.0
    mult = band_mult(bands, purity)
    if total < M_min:
        mult = min(mult, cap_small)
    return total * v[dom] * mult


def read_output_tokens(path):
    with open(path, "r") as f:
        content = f.read()
    return content.split()


def parse_int_strict(tok):
    """Reject anything that is not a clean base-10 integer (rejects nan/inf/huge-float
    garbage, +/- signs allowed)."""
    if tok is None:
        return None
    s = tok
    if s and s[0] in "+-":
        body = s[1:]
    else:
        body = s
    if body == "" or not body.isdigit():
        return None
    try:
        val = int(s)
    except ValueError:
        return None
    if val != val or abs(val) > 10**15:
        return None
    return val


def parse_cuts_and_actions(tokens, pos, G, allowed_actions, step=1):
    """Reads: c, c increasing ints in [1,G-1] (each a multiple of `step`), then c+1
    action tokens from allowed_actions. Returns (cuts, actions, new_pos) or die()."""
    if pos >= len(tokens):
        die("missing cut-count token")
    c = parse_int_strict(tokens[pos]); pos += 1
    if c is None or c < 0 or c > G - 1:
        die(f"invalid cut count {tokens[pos-1]!r}")
    cuts = []
    prev = 0
    for _ in range(c):
        if pos >= len(tokens):
            die("missing cut value")
        x = parse_int_strict(tokens[pos]); pos += 1
        if x is None or x < 1 or x > G - 1:
            die(f"cut out of range: {tokens[pos-1]!r}")
        if x <= prev:
            die("cuts not strictly increasing")
        if x % step != 0:
            die(f"pass-1 cut {x} is not a multiple of the coarse step {step}")
        cuts.append(x)
        prev = x
    nfrac = c + 1
    actions = []
    for _ in range(nfrac):
        if pos >= len(tokens):
            die("missing action token")
        tok = tokens[pos]; pos += 1
        if tok not in allowed_actions:
            die(f"bad action token {tok!r}")
        actions.append(tok)
    return cuts, actions, pos


def segments_from_cuts(cuts, G):
    bounds = [0] + cuts + [G]
    return list(zip(bounds[:-1], bounds[1:]))


def evaluate(inpath, outpath):
    K, G, step1, v, m, H, energyCost, M_min, cap_small, bands = read_instance(inpath)
    pref = make_prefix(m, K, G)

    tokens = read_output_tokens(outpath)
    pos = 0
    cuts1, actions1, pos = parse_cuts_and_actions(tokens, pos, G, {"S", "R", "D"}, step=step1)
    segs1 = segments_from_cuts(cuts1, G)
    if len(segs1) != len(actions1):
        die("segment/action count mismatch (pass 1)")

    F = 0.0
    recycle_mask = [False] * G  # which bins get carried to pass 2
    any_recycle = False
    for (a, b), act in zip(segs1, actions1):
        if act == "S":
            F += seg_revenue(pref, v, bands, K, a, b, M_min, cap_small) - H
        elif act == "D":
            pass
        elif act == "R":
            mass, _, _ = seg_mass_purity(pref, K, a, b)
            F -= energyCost * mass
            for g in range(a, b):
                recycle_mask[g] = True
            any_recycle = True

    if any_recycle:
        m2 = [[m[c][g] if recycle_mask[g] else 0.0 for g in range(G)] for c in range(K)]
        pref2 = make_prefix(m2, K, G)
        cuts2, actions2, pos = parse_cuts_and_actions(tokens, pos, G, {"S", "D"})
        segs2 = segments_from_cuts(cuts2, G)
        if len(segs2) != len(actions2):
            die("segment/action count mismatch (pass 2)")
        for (a, b), act in zip(segs2, actions2):
            if act == "S":
                F += seg_revenue(pref2, v, bands, K, a, b, M_min, cap_small) - H
            # D contributes 0

    if pos != len(tokens):
        die(f"trailing garbage in output ({len(tokens) - pos} extra tokens)")

    if not math.isfinite(F):
        die("non-finite objective")

    return F, K, G, step1, v, m, H, energyCost, M_min, cap_small, bands, pref


def baseline_B(K, G, step1, v, m, H, energyCost, M_min, cap_small, bands, pref):
    """Uniform K-slice single-pass sell-everything construction (no recycle),
    cuts snapped to the coarse pass-1 grid so it is itself a feasible plan."""
    raw = [round(i * G / K / step1) * step1 for i in range(1, K)]
    cuts = sorted({c for c in raw if 1 <= c <= G - 1})
    segs = segments_from_cuts(cuts, G)
    B = 0.0
    for a, b in segs:
        B += seg_revenue(pref, v, bands, K, a, b, M_min, cap_small) - H
    return max(B, 1e-6)


def main():
    if len(sys.argv) < 3:
        print("usage: verify.py <in> <out> <ans>")
        print("Ratio: 0.0")
        sys.exit(0)
    inpath, outpath = sys.argv[1], sys.argv[2]
    try:
        F, K, G, step1, v, m, H, energyCost, M_min, cap_small, bands, pref = evaluate(inpath, outpath)
    except SystemExit:
        raise
    except Exception as e:
        print("EXCEPTION:", repr(e))
        print("Ratio: 0.0")
        sys.exit(0)

    B = baseline_B(K, G, step1, v, m, H, energyCost, M_min, cap_small, bands, pref)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    sc = max(0.0, sc)
    print(f"F={F:.6f} B={B:.6f}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
