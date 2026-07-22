# TIER: greedy
"""The obvious approach: run a single-pass revenue-maximizing DP over cut points (a
textbook 'segment the array to maximize value' DP) and sell everything pass 1 produces.
The statement offers a two-pass recycle mechanism, but a first-pass coder treats pass 1
as the whole problem: whatever partition maximizes pass-1 revenue IS the answer, full
stop. In particular this never notices the batch-size price qualifier: a small,
already-pure lobe still prices at par with everything else pass 1 decided to sell, and
nothing here ever asks whether combining several such lobes across CUT boundaries would
do better -- cuts are chosen purely to classify pass 1's own sale, never to shape what a
second pass could see."""
import sys

EPS = 1e-9


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    K = int(next(it)); G = int(next(it)); step1 = int(next(it))
    v = [float(next(it)) for _ in range(K)]
    m = [[float(next(it)) for _ in range(G)] for _ in range(K)]
    H = float(next(it)); energyCost = float(next(it))
    M_min = float(next(it)); cap_small = float(next(it))
    B = int(next(it))
    bands = [(float(next(it)), float(next(it))) for _ in range(B)]
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
        for g in range(G):
            acc += m[c][g]
            pref[c][g + 1] = acc
    return pref


def seg_mass_purity(pref, K, a, b):
    masses = [pref[c][b] - pref[c][a] for c in range(K)]
    total = sum(masses)
    if total <= EPS:
        return 0.0, 0.0, 0
    dom = max(range(K), key=lambda c: masses[c])
    return total, masses[dom] / total, dom


def seg_revenue(pref, v, bands, K, a, b, M_min=0.0, cap_small=1.0):
    total, purity, dom = seg_mass_purity(pref, K, a, b)
    if total <= EPS:
        return 0.0
    mult = band_mult(bands, purity)
    if total < M_min:
        mult = min(mult, cap_small)
    return total * v[dom] * mult


def optimal_partition(pref, v, bands, K, H, a, b, M_min, cap_small, step=1):
    """DP over [a,b): dp[g] = best net value of [a,g), each segment sold-or-dumped
    (max(0, rev-H)), boundaries restricted to multiples of `step` from `a`.
    Returns (value, list of (lo,hi,action in {S,D}))."""
    n = b - a
    assert n % step == 0
    idxs = list(range(0, n + 1, step))
    dp = [0.0] * (n + 1)
    choice = [-1] * (n + 1)
    act = [None] * (n + 1)
    for gi in idxs[1:]:
        g = a + gi
        best_val = -1.0
        best_j = 0
        best_act = "D"
        for ji in idxs:
            if ji >= gi:
                break
            j = a + ji
            rev = seg_revenue(pref, v, bands, K, j, g, M_min, cap_small)
            net = rev - H
            take = net if net > 0 else 0.0
            cand = dp[ji] + take
            if cand > best_val + 1e-12:
                best_val = cand
                best_j = ji
                best_act = "S" if net > 0 else "D"
        dp[gi] = best_val
        choice[gi] = best_j
        act[gi] = best_act
    # reconstruct
    segs = []
    gi = n
    while gi > 0:
        ji = choice[gi]
        segs.append((a + ji, a + gi, act[gi]))
        gi = ji
    segs.reverse()
    return dp[n], segs


def main():
    K, G, step1, v, m, H, energyCost, M_min, cap_small, bands = read_instance()
    pref = make_prefix(m, K, G)

    _, segs1 = optimal_partition(pref, v, bands, K, H, 0, G, M_min, cap_small, step=step1)

    actions1 = [act for (_, _, act) in segs1]  # every fraction is S or D, never R

    out = []
    cuts1 = [b for (a, b, _) in segs1[:-1]]
    out.append(f"{len(cuts1)} " + " ".join(str(c) for c in cuts1))
    out.append(" ".join(actions1))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
