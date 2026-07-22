import sys, math
from fractions import Fraction

ALPHA = Fraction(9, 10)   # weight on average per-instrument evenness
BETA = Fraction(1, 10)    # weight on normalized merged spatial-dispersion entropy


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def read_instance(path):
    try:
        toks = open(path).read().split()
        it = iter(toks)
        N = int(next(it)); D = int(next(it)); R = int(next(it)); cap = int(next(it))
        chain = [int(next(it)) for _ in range(D)]
        free = [int(next(it)) for _ in range(R)]
        return N, D, R, cap, chain, free
    except Exception:
        fail("bad input")


def evenness(onsets, N):
    """mean^2 / (mean^2 + population-variance-of-circular-gaps); 1.0 = perfectly even,
    smoothly decreasing towards 0 as gaps get more irregular. Rotation-invariant (depends
    only on the multiset of gaps, not on phase)."""
    k = len(onsets)
    pts = sorted(onsets)
    if k == 1:
        return Fraction(1, 1)
    gaps = []
    for i in range(k):
        a = pts[i]; b = pts[(i + 1) % k]
        g = (b - a) if b > a else (b - a + N)
        gaps.append(g)
    mean = Fraction(N, k)
    var = sum((Fraction(g) - mean) ** 2 for g in gaps) / k
    m2 = mean * mean
    return m2 / (m2 + var)


BINS = 24  # fixed, instance-independent: coarse-graining of the cycle for the texture term


def merged_entropy_norm(all_points, N):
    """Shannon entropy of how the merged (deduplicated) onset set's DENSITY is spread across
    BINS equal-width arcs of the cycle, normalized by log2(BINS). This is the 'texture'
    bonus: a merged pattern that touches many different parts of the cycle fairly evenly
    scores high; a pattern that piles activity into a few arcs (or leaves long stretches
    silent) scores low. Unlike a raw inter-onset-GAP-length entropy, this rewards genuine
    spatial coverage rather than gap-length irregularity, so it does not reward chaotic
    jittering of an otherwise-even pattern."""
    pts = list(all_points)
    m = len(pts)
    if m == 0:
        return 0.0
    occ = [0] * BINS
    for p in pts:
        occ[(p * BINS) // N] += 1
    H = 0.0
    for c in occ:
        if c == 0:
            continue
        pfrac = c / m
        H -= pfrac * math.log2(pfrac)
    maxH = math.log2(min(BINS, m)) if m > 1 else 0.0
    return H / maxH if maxH > 0 else 0.0


def score_raw(instrument_onsets, N):
    """instrument_onsets: list of lists (onset positions per instrument)."""
    ev_sum = Fraction(0, 1)
    all_pts = set()
    for ons in instrument_onsets:
        ev_sum += evenness(ons, N)
        all_pts.update(ons)
    ev_avg = ev_sum / len(instrument_onsets)
    ent = merged_entropy_norm(all_pts, N)
    return float(ALPHA) * float(ev_avg) + float(BETA) * ent


def baseline_construction(N, D, R, chain, free):
    """Always-feasible trivial reference: chain node i (0-indexed, ascending k) gets the
    first k_i integers [0..k_i-1) -- a nested run of prefixes, trivially satisfying the
    subset hierarchy. Each free instrument gets its own consecutive block of f_j integers,
    but the blocks are parked at evenly spaced anchors around the full cycle (not all
    crammed next to the chain) so the merged pattern isn't a single tiny clump. Every
    instrument is still badly uneven (bunched consecutively), and every block is far
    smaller than the spacing between anchors, so nothing ever collides regardless of cap
    (cap is required >= D by the generator)."""
    groups = 1 + R
    spacing = N // groups
    instr = []
    for k in chain:
        instr.append(list(range(k)))
    for idx, f in enumerate(free):
        anchor = (idx + 1) * spacing
        instr.append([(anchor + x) % N for x in range(f)])
    return instr


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, D, R, cap, chain, free = read_instance(in_path)
    M = D + R
    k_list = chain + free

    try:
        raw_lines = open(out_path).read().splitlines()
    except Exception:
        fail("cannot read output")
    lines = [ln for ln in raw_lines if ln.strip() != ""]
    if len(lines) != M:
        fail("expected %d instrument lines, got %d" % (M, len(lines)))

    instr = []
    for i in range(M):
        toks = lines[i].split()
        if len(toks) != k_list[i]:
            fail("instrument %d expected %d onsets, got %d" % (i, k_list[i], len(toks)))
        vals = []
        for t in toks:
            try:
                v = int(t)
            except Exception:
                fail("non-integer onset %r at instrument %d" % (t, i))
            if not (0 <= v < N):
                fail("onset %d out of range at instrument %d" % (v, i))
            vals.append(v)
        if len(set(vals)) != len(vals):
            fail("duplicate onset within instrument %d" % i)
        instr.append(vals)

    # --- nesting: chain[i] onsets must be a subset of chain[i+1] onsets ---
    for i in range(D - 1):
        if not set(instr[i]).issubset(set(instr[i + 1])):
            fail("nesting violated between chain layer %d and %d" % (i, i + 1))

    # --- collision cap: no beat may be hit by more than `cap` distinct instruments ---
    cnt = [0] * N
    for ons in instr:
        for v in ons:
            cnt[v] += 1
    mx = max(cnt) if N > 0 else 0
    if mx > cap:
        fail("collision cap exceeded: %d > %d" % (mx, cap))

    F = score_raw(instr, N)
    base_instr = baseline_construction(N, D, R, chain, free)
    B = score_raw(base_instr, N)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
