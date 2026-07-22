# TIER: strong
"""Insight: a pass-1 cut is not a classification threshold on pass 1's own sale --
it is an operator that DESIGNS the distribution pass 2 will see. The batch-size price
qualifier means a small, already-pure lobe's problem is invisible to any PURITY-based
heuristic (greedy's threshold never fires on it) -- the only way to see the lost value
is to actually price it. This solver's central move (plan C) groups pass-1's sold
fractions by DOMINANT component and checks, per group, whether consolidating several
scattered same-component lobes into one recycle stream crosses the batch-size floor and
unlocks the full purity-band price for the combined lot -- deliberately routing several
individually-fine-looking cuts together rather than judging each in isolation. It also
tries a quarantine-window co-design (plan B) and the plain greedy plan (plan A) and
keeps whichever of the three is actually better, so it never does worse than the
obvious recipe."""
import sys

EPS = 1e-9
THRESH = 0.60


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


def total_mass(pref, K, a, b):
    return sum(pref[c][b] - pref[c][a] for c in range(K))


def optimal_partition(pref, v, bands, K, H, a, b, M_min, cap_small, step=1):
    """Full DP over [a,b): dp_local[i] = best net value of [a, a+i), sold-or-dumped,
    boundaries restricted to multiples of `step` from a. Returns (dp_local array of
    length n+1, choice[], act[]) -- only entries at multiples of `step` are meaningful."""
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
    return dp, choice, act


def reconstruct(a, choice, act, gi):
    segs = []
    while gi > 0:
        ji = choice[gi]
        segs.append((a + ji, a + gi, act[gi]))
        gi = ji
    segs.reverse()
    return segs


def emit(cuts1, actions1, cuts2, actions2):
    out = [f"{len(cuts1)} " + " ".join(str(c) for c in cuts1),
           " ".join(actions1)]
    if cuts2 is not None:
        out.append(f"{len(cuts2)} " + " ".join(str(c) for c in cuts2))
        out.append(" ".join(actions2))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    K, G, step1, v, m, H, energyCost, M_min, cap_small, bands = read_instance()
    pref = make_prefix(m, K, G)

    # ---- coarse pass-1 partition (shared basis for plans A and C) ----
    dp_full, choice_full, act_full = optimal_partition(
        pref, v, bands, K, H, 0, G, M_min, cap_small, step=step1)
    segsA = reconstruct(0, choice_full, act_full, G)
    cuts1_base = [b for (a, b, _) in segsA[:-1]]

    # ---- plan A: plain single-pass DP, then check if reprocessing the single
    # worst-purity fraction alone (at FINE resolution) pays off (mirrors
    # solutions/greedy.py exactly, so this is a genuine "no worse than the obvious
    # recipe" floor). Small, already-pure lobes never trip this purity check. ----
    worst_idx = None
    worst_purity = THRESH
    for i, (a, b, act) in enumerate(segsA):
        if act == "S":
            _, purity, _ = seg_mass_purity(pref, K, a, b)
            if purity < worst_purity:
                worst_purity = purity
                worst_idx = i

    actionsA = [act for (_, _, act) in segsA]
    segs2A = None
    if worst_idx is not None:
        a, b, _ = segsA[worst_idx]
        mass, _, _ = seg_mass_purity(pref, K, a, b)
        cur_net = seg_revenue(pref, v, bands, K, a, b, M_min, cap_small) - H
        dp_loc, choice_loc, act_loc = optimal_partition(pref, v, bands, K, H, a, b, M_min, cap_small)
        dp2_val = dp_loc[b - a]
        cand_segs2A = reconstruct(a, choice_loc, act_loc, b - a)
        recycle_net = dp2_val - energyCost * mass
        if recycle_net > cur_net:
            actionsA[worst_idx] = "R"
            segs2A = cand_segs2A

    valueA = 0.0
    for (a, b, _), act in zip(segsA, actionsA):
        if act == "S":
            valueA += seg_revenue(pref, v, bands, K, a, b, M_min, cap_small) - H
        elif act == "R":
            mass, _, _ = seg_mass_purity(pref, K, a, b)
            valueA -= energyCost * mass
    if segs2A is not None:
        for (a, b, act) in segs2A:
            if act == "S":
                valueA += seg_revenue(pref, v, bands, K, a, b, M_min, cap_small) - H

    best_value = valueA
    best_plan = ("A", cuts1_base, actionsA,
                 ([b for (a, b, _) in segs2A[:-1]] if segs2A is not None else None),
                 ([act for (_, _, act) in segs2A] if segs2A is not None else None))

    # ---- plan C: consolidate by DOMINANT COMPONENT to clear the batch-size gate.
    # For a candidate target component c, identify EVERY bin whose own raw dominant
    # component (by mass, ignoring how pass-1's revenue-only DP happened to group
    # its neighbors) is c, and recycle ALL of them as one non-contiguous pass-1
    # decision. This is built from the RAW per-bin data, not from segsA's sale
    # partition: segsA's boundaries were chosen to maximize pass-1 revenue alone and
    # can blend a lobe's edge into a neighboring sale fraction, which would make an
    # isolated-mask check built from those boundaries under-count how pure/consoli-
    # datable the component-c lobes really are. Every OTHER bin (whatever is left
    # once c's bins are pulled out) is independently re-optimized by the same
    # revenue-maximizing DP. This is invisible to a purity threshold -- every lobe
    # already looks "fine" on its own -- so it is a genuinely different move from
    # plan A's single-worst-fraction check: it routes by raw composition, not by
    # judging pass-1's already-decided sale fractions one at a time.
    #
    # Only ONE target component is chosen at a time: components are laid out
    # round-robin, so different components' lobes are spatially interleaved, and
    # recycling two different targets' bins together would sweep each other's
    # lobes into the same pass-2 fraction, diluting purity right back down. ----
    dom_bin = [None] * G
    for g in range(G):
        best_c, best_m = None, 0.0
        for c in range(K):
            mg = pref[c][g + 1] - pref[c][g]
            if mg > best_m:
                best_m, best_c = mg, c
        dom_bin[g] = best_c

    best_plan_c = None  # (value, cuts1, actions1, cuts2, actions2)
    for c in range(K):
        mask = [dom_bin[g] == c for g in range(G)]
        if not any(mask):
            continue
        pieces = []  # (a, b, action) left-to-right, contiguous, covers [0,G)
        remaining_value = 0.0
        mass_c = 0.0
        gpos = 0
        while gpos < G:
            cur = mask[gpos]
            gend = gpos
            while gend < G and mask[gend] == cur:
                gend += 1
            if cur:
                pieces.append((gpos, gend, "R"))
                mass_c += total_mass(pref, K, gpos, gend)
            else:
                dp_loc, choice_loc, act_loc = optimal_partition(
                    pref, v, bands, K, H, gpos, gend, M_min, cap_small)
                remaining_value += dp_loc[gend - gpos]
                pieces.extend(reconstruct(gpos, choice_loc, act_loc, gend - gpos))
            gpos = gend

        m_masked = [[m[cc][g] if mask[g] else 0.0 for g in range(G)] for cc in range(K)]
        pref_masked = make_prefix(m_masked, K, G)
        dp2, choice2, act2 = optimal_partition(pref_masked, v, bands, K, H, 0, G, M_min, cap_small)
        segs2C = reconstruct(0, choice2, act2, G)
        pass2_value = dp2[G]

        total_value = remaining_value - energyCost * mass_c + pass2_value
        if best_plan_c is None or total_value > best_plan_c[0] + 1e-9:
            cuts1C = [b for (a, b, _) in pieces[:-1]]
            actions1C = [act for (_, _, act) in pieces]
            cuts2C = [b for (a, b, _) in segs2C[:-1]]
            actions2C = [act for (_, _, act) in segs2C]
            best_plan_c = (total_value, cuts1C, actions1C, cuts2C, actions2C)

    if best_plan_c is not None and best_plan_c[0] > best_value + 1e-9:
        best_value = best_plan_c[0]
        best_plan = ("C",) + best_plan_c[1:]

    # ---- plan B: quarantine-window co-design (kept as a general-purpose fallback;
    # a pass-1 cut placed to deliberately sacrifice local revenue so pass 2 gets a
    # cleaner, wider stream). bestHead[lo]/bestTail[hi] = optimal COARSE partial
    # revenue of [0,lo) / [hi,G). ----
    bestHead = dp_full
    coarse_idxs = list(range(0, G + 1, step1))
    dp_back = [0.0] * (G + 1)
    choice_back = [-1] * (G + 1)
    act_back = [None] * (G + 1)
    for gi in reversed(coarse_idxs[:-1]):
        best_val = -1.0
        best_k = G
        best_act = "D"
        for k in coarse_idxs:
            if k <= gi:
                continue
            rev = seg_revenue(pref, v, bands, K, gi, k, M_min, cap_small)
            net = rev - H
            take = net if net > 0 else 0.0
            cand = take + dp_back[k]
            if cand > best_val + 1e-12:
                best_val = cand
                best_k = k
                best_act = "S" if net > 0 else "D"
        dp_back[gi] = best_val
        choice_back[gi] = best_k
        act_back[gi] = best_act
    bestTail = dp_back

    best_total = best_value
    best_lo = best_hi = None
    best_dp2_choice = best_dp2_act = None

    for lo in coarse_idxs[:-1]:
        n = G - lo
        dp2 = [0.0] * (n + 1)
        ch2 = [-1] * (n + 1)
        ac2 = [None] * (n + 1)
        for hi_i in range(1, n + 1):
            hi = lo + hi_i
            best_val = -1.0
            best_j = 0
            best_act = "D"
            for j_i in range(0, hi_i):
                j = lo + j_i
                rev = seg_revenue(pref, v, bands, K, j, hi, M_min, cap_small)
                net = rev - H
                take = net if net > 0 else 0.0
                cand = dp2[j_i] + take
                if cand > best_val + 1e-12:
                    best_val = cand
                    best_j = j_i
                    best_act = "S" if net > 0 else "D"
            dp2[hi_i] = best_val
            ch2[hi_i] = best_j
            ac2[hi_i] = best_act
            hi_full = hi
            if hi_full % step1 != 0 or hi_full - lo < 1:
                continue
            mass_q = total_mass(pref, K, lo, hi_full)
            if mass_q <= EPS:
                continue
            cand_total = bestHead[lo] + bestTail[hi_full] + dp2[hi_i] - energyCost * mass_q
            if cand_total > best_total + 1e-9:
                best_total = cand_total
                best_lo, best_hi = lo, hi_full
                best_dp2_choice, best_dp2_act = ch2[:], ac2[:]

    if best_lo is not None:
        lo, hi = best_lo, best_hi
        head_segs = reconstruct(0, choice_full, act_full, lo) if lo > 0 else []
        tail_segs = []
        k = hi
        while k < G:
            nk = choice_back[k]
            tail_segs.append((k, nk, act_back[k]))
            k = nk

        cuts1B = []
        actions1B = []
        for (a, b, act) in head_segs:
            actions1B.append(act)
            if b < lo:
                cuts1B.append(b)
        if lo > 0:
            cuts1B.append(lo)
        actions1B.append("R")
        if hi < G:
            cuts1B.append(hi)
        for (a, b, act) in tail_segs:
            actions1B.append(act)
            if b < G:
                cuts1B.append(b)
        cuts1B = sorted(c for c in cuts1B if 0 < c < G)

        n_q = hi - lo
        seg2B = []
        gi = n_q
        while gi > 0:
            ji = best_dp2_choice[gi]
            seg2B.append((lo + ji, lo + gi, best_dp2_act[gi]))
            gi = ji
        seg2B.reverse()
        cuts2B = [b for (a, b, _) in seg2B[:-1]]
        actions2B = [act for (a, b, act) in seg2B]

        best_plan = ("B", cuts1B, actions1B, cuts2B, actions2B)

    _, cuts1, actions1, cuts2, actions2 = best_plan
    emit(cuts1, actions1, cuts2, actions2)


if __name__ == "__main__":
    main()
