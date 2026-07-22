# TIER: strong
# Convolutional-coverage insight: reformulate scheduling as covering targets
# with diffusion-kernel "humps". For every (burner, window) we precompute the
# EXACT per-unit-power contribution each target receives at deadline T (a
# single impulse-response simulation per candidate window) -- this is the
# stated kernel K(distance, T-start) evaluated once. By linearity of the
# diffusion update, the effect of any set of firings is the exact sum of
# their per-unit contributions, so covering all targets reduces to a
# weighted set-cover over these precomputed kernel humps instead of a
# per-zone nearest-burner reflex. A pruning/shrinking exchange pass then
# removes redundant firings and tightens power to the exact minimum,
# re-verified by a full joint simulation (never trusts the linear-algebra
# shortcut for the final feasibility answer).
import sys


def simulate_scaled(N, firings, T):
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


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); T = int(next(it)); H = int(next(it))
    Pmax = int(next(it)); F0 = int(next(it))
    numB = int(next(it)); numT = int(next(it))
    burners = [int(next(it)) for _ in range(numB)]
    targets = [int(next(it)) for _ in range(numT)]

    def nearest_burner_idx(pos):
        bi_best, d_best = 0, None
        for i, bp in enumerate(burners):
            dd = abs(pos - bp)
            if d_best is None or dd < d_best or (dd == d_best and i < bi_best):
                d_best = dd; bi_best = i
        return bi_best

    # ---- candidate kernel windows: several durations, each anchored early
    # (s=0, max transit time to reach distant targets) and anchored late
    # (s=T-d, ends exactly at the deadline, no post-firing decay). ----
    fracs = [(1, 1), (5, 6), (2, 3), (1, 2), (1, 3), (1, 4), (1, 6), (1, 8)]
    durations = sorted({max(1, min(T, T * num // den)) for num, den in fracs}, reverse=True)
    windows = set()
    for d in durations:
        windows.add((0, d))
        windows.add((T - d, d))
    windows = sorted(windows)

    # precompute per-(burner,window) contribution-per-unit-power to each target
    contrib = {}
    for bi, bp in enumerate(burners):
        for (s, d) in windows:
            Hs, _ = simulate_scaled(N, [(bp, s, d, 1)], T)
            contrib[(bi, s, d)] = [Hs[x] for x in targets]

    def get_contrib(bi, s, d):
        key = (bi, s, d)
        if key not in contrib:
            Hs, _ = simulate_scaled(N, [(burners[bi], s, d, 1)], T)
            contrib[key] = [Hs[x] for x in targets]
        return contrib[key]

    SCALE = 4 ** T
    H_scaled = H * SCALE
    achieved = [0] * numT
    committed = []  # [bi, s, d, p]

    def deficits():
        return [max(0, H_scaled - achieved[i]) for i in range(numT)]

    # ---- greedy kernel-coverage: repeatedly commit the cheapest-per-newly-
    # covered-target (burner,window,power) until every target clears H ----
    guard = 0
    max_guard = 4 * numB * max(1, len(windows)) + 20
    while any(v > 0 for v in deficits()) and guard < max_guard:
        guard += 1
        defs = deficits()
        best = None
        for bi in range(numB):
            for (s, d) in windows:
                c = contrib[(bi, s, d)]
                relevant = [i for i in range(numT) if defs[i] > 0 and c[i] > 0]
                if not relevant:
                    continue
                p_needed = 1
                for i in relevant:
                    need_p = -(-defs[i] // c[i])
                    if need_p > p_needed:
                        p_needed = need_p
                if p_needed > Pmax:
                    p_needed = Pmax
                covered = sum(1 for i in relevant if defs[i] - p_needed * c[i] <= 0)
                if covered == 0:
                    continue
                fuel_cost = p_needed * d + F0
                ratio = fuel_cost / covered
                cand = (ratio, fuel_cost, bi, s, d, p_needed)
                if best is None or cand[:2] < best[:2]:
                    best = cand
        if best is None:
            break
        _, _, bi, s, d, p_needed = best
        committed.append([bi, s, d, p_needed])
        c = contrib[(bi, s, d)]
        for i in range(numT):
            achieved[i] += p_needed * c[i]

    # ---- fallback: guarantee feasibility for any still-deficient target via
    # its own nearest-burner full-window firing, on top of what's committed ----
    for i in range(numT):
        if deficits()[i] > 0:
            bi = nearest_burner_idx(targets[i])
            s, d = 0, T
            c = get_contrib(bi, s, d)
            committed.append([bi, s, d, Pmax])
            for j in range(numT):
                achieved[j] += Pmax * c[j]

    def joint_feasible(firings):
        pos_f = [(burners[bi], s, d, p) for (bi, s, d, p) in firings]
        Hs, scale = simulate_scaled(N, pos_f, T)
        need = H * scale
        return all(Hs[targets[i]] >= need for i in range(numT))

    # ---- exchange/pruning pass: drop redundant firings, shrink power to the
    # exact minimum feasible integer, re-verified by full joint simulation ----
    passes = 0
    changed = True
    while changed and passes < 6:
        changed = False
        passes += 1
        i = 0
        while i < len(committed):
            trial = committed[:i] + committed[i + 1:]
            if joint_feasible(trial):
                committed = trial
                changed = True
            else:
                i += 1
        for idx in range(len(committed)):
            bi, s, d, p = committed[idx]
            lo, hi, best_p = 1, p, p
            while lo <= hi:
                mid = (lo + hi) // 2
                trial = committed[:idx] + [[bi, s, d, mid]] + committed[idx + 1:]
                if joint_feasible(trial):
                    best_p = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            if best_p < p:
                committed[idx][3] = best_p
                changed = True

    if not committed or not joint_feasible(committed):
        used = sorted({nearest_burner_idx(x) for x in targets})
        committed = [[bi, 0, T, Pmax] for bi in used]

    out = [str(len(committed))]
    for (bi, s, d, p) in committed:
        out.append(f"{bi} {s} {d} {p}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
