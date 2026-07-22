import sys, random

# gen.py <testId> -- prints ONE dragon-kiln firing instance to stdout.
#
# Grid: N cells in a row (a 1-D cross-section of the kiln wall). Burner sites and
# target cells sit at grid positions. Heat diffuses every step with a fixed,
# insulated-boundary linear rule (see statement.md / simulate_scaled() below).
# Feasibility requires every target cell to be at temperature >= H at EXACTLY
# step T.
#
# Each test's (N, T, H, Pmax, F0, burner/target placement) is TUNED by direct
# simulation search (seeded by testId, pure integer arithmetic -> fast & exact)
# so that:
#   - the checker's own baseline construction (full-window s=0,d=T blast from
#     each target's nearest burner, deduplicated) is always feasible.
#   - on "trap" tests, the naive per-target reflex (nearest burner, fire
#     immediately, duration picked by a distance-blind local formula that
#     pretends heat accumulates with NO diffusion loss) is INFEASIBLE: it
#     injects far too little total heat once the real dilution-over-distance
#     is accounted for.
#   - on "easy" tests the naive reflex stays feasible but wastes fuel relative
#     to a consolidated, kernel-timed plan (no dedup across shared burners,
#     no timing/power tuning).

PMAX = 10
F0 = 4


def simulate_scaled(N, firings, T):
    """firings: list of (pos, s, d, p), all non-negative ints.
    Exact integer diffusion: h'[i] = h[i]/2 + h[i-1]/4 + h[i+1]/4 (insulated /
    mirrored boundary). Returns (Hs, scale) with true heat[i] == Hs[i] / scale
    and scale == 4**T (kept as pure integers -- no floating point, no
    Fraction/GCD overhead, bit-for-bit deterministic)."""
    Hs = [0] * N
    scale = 1
    for t in range(T):
        for (pos, s, d, p) in firings:
            if s <= t < s + d:
                Hs[pos] += p * scale
        nHs = [0] * N
        for i in range(N):
            left = Hs[i - 1] if i - 1 >= 0 else Hs[i]
            right = Hs[i + 1] if i + 1 < N else Hs[i]
            nHs[i] = 2 * Hs[i] + left + right
        Hs = nHs
        scale *= 4
    return Hs, scale


def feasible(N, T, H, targets, firings_idx, burners):
    pos_firings = [(burners[bi], s, d, p) for (bi, s, d, p) in firings_idx]
    Hs, scale = simulate_scaled(N, pos_firings, T)
    need = H * scale
    return all(Hs[x] >= need for x in targets)


def nearest_burner_idx(pos, burners):
    best_i, best_d = 0, None
    for i, bp in enumerate(burners):
        dd = abs(pos - bp)
        if best_d is None or dd < best_d or (dd == best_d and i < best_i):
            best_d = dd
            best_i = i
    return best_i


def baseline_firings(N, T, Pmax, burners, targets):
    """Checker's internal baseline B: one full-window (s=0,d=T,p=Pmax) firing per
    DISTINCT nearest burner needed to cover the targets."""
    used = sorted({nearest_burner_idx(x, burners) for x in targets})
    return [(bi, 0, T, Pmax) for bi in used]


def greedy_firings(N, T, H, Pmax, burners, targets):
    """The 'obvious' reflex: for EACH target independently, fire its nearest
    burner immediately (s=0) with the shortest duration a naive no-diffusion,
    distance-blind estimate says is enough: d = ceil(H / Pmax), clipped to T.
    No deduplication across targets."""
    out = []
    d_naive = min(T, -(-H // Pmax))  # ceil div
    for x in targets:
        bi = nearest_burner_idx(x, burners)
        out.append((bi, 0, d_naive, Pmax))
    return out


# ---- per-test size / mode ladder ----
# (N, T, numBurners, numTargets, mode)  mode: "easy" or "trap"
SPECS = {
    1:  (10, 10, 2, 2, "easy"),
    2:  (14, 12, 2, 3, "easy"),
    3:  (18, 26, 3, 3, "trap"),
    4:  (16, 12, 3, 3, "easy"),
    5:  (20, 30, 3, 4, "trap"),
    6:  (18, 14, 4, 4, "easy"),
    7:  (22, 32, 3, 4, "trap"),
    8:  (20, 15, 4, 5, "easy"),
    9:  (24, 34, 4, 5, "trap"),
    10: (28, 36, 5, 6, "trap"),
}


def gen_placement(rng, N, numB, numT, mode):
    cells = list(range(N))
    burners = rng.sample(cells, numB)
    if mode == "easy":
        # targets sit ON (or one step off) a burner -> distance-blind naive
        # duration formula is not punished by missing transit time.
        targets = []
        for _ in range(numT):
            b = rng.choice(burners)
            off = rng.choice([-1, 0, 0, 1])
            x = min(N - 1, max(0, b + off))
            targets.append(x)
    else:
        # trap: every target sits FAR (>= ~N/5, at least 4) from every burner.
        min_dist = max(4, N // 5)
        targets = []
        tries = 0
        while len(targets) < numT and tries < 4000:
            tries += 1
            x = rng.choice(cells)
            if all(abs(x - b) >= min_dist for b in burners) and x not in targets:
                targets.append(x)
        if len(targets) < numT:
            return None
    return burners, targets


def search_H(N, T, burners, targets, mode):
    """Scan candidate H values; return the H meeting the mode's feasibility
    contract (found while baseline is still feasible), or None."""
    hi = PMAX * T
    best = None
    for H in range(2, hi + 1):
        b_firings = baseline_firings(N, T, PMAX, burners, targets)
        if not feasible(N, T, H, targets, b_firings, burners):
            break  # baseline's achieved heat is fixed; feasibility vs H is monotone
        g_firings = greedy_firings(N, T, H, PMAX, burners, targets)
        g_ok = feasible(N, T, H, targets, g_firings, burners)
        if mode == "easy":
            if g_ok:
                best = H
        else:
            if not g_ok:
                best = H
    return best


def build(tid):
    N, T, numB, numT, mode = SPECS[tid]
    rng = random.Random(7724110 + 91 * tid)
    for attempt in range(600):
        placement = gen_placement(rng, N, numB, numT, mode)
        if placement is None:
            continue
        burners, targets = placement
        H = search_H(N, T, burners, targets, mode)
        if H is None:
            continue
        if mode == "easy":
            target_H = max(2, int(H * 0.94))
            if target_H >= H:
                continue
            gf = greedy_firings(N, T, target_H, PMAX, burners, targets)
            bf = baseline_firings(N, T, PMAX, burners, targets)
            if not feasible(N, T, target_H, targets, bf, burners):
                continue
            if not feasible(N, T, target_H, targets, gf, burners):
                continue
        else:
            target_H = max(2, int(H * 0.90))
            bf = baseline_firings(N, T, PMAX, burners, targets)
            if not feasible(N, T, target_H, targets, bf, burners):
                continue
            gf = greedy_firings(N, T, target_H, PMAX, burners, targets)
            if feasible(N, T, target_H, targets, gf, burners):
                continue
        return N, T, target_H, PMAX, F0, burners, targets
    raise RuntimeError(f"gen.py: could not tune test {tid} (mode={mode}) after many attempts")


def main():
    tid = int(sys.argv[1])
    N, T, H, Pmax, F0v, burners, targets = build(tid)
    out = []
    out.append(f"{N} {T} {H} {Pmax} {F0v} {len(burners)} {len(targets)}")
    out.append(" ".join(map(str, burners)))
    out.append(" ".join(map(str, targets)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
