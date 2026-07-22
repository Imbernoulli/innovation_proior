# TIER: strong
import sys


def allowed_hours(g, r):
    return [h for h in range(24) if h % g == r]


def coverage(starts, L):
    cov = [0] * 24
    for s in starts:
        for k in range(L):
            cov[(s + k) % 24] += 1
    return cov


def worst_case_peak(starts, L, C, base, profiles, days=3):
    cov = coverage(starts, L)
    cap = [C * c for c in cov]
    worst = 0
    T = days * 24
    for (sk, dk, ak) in profiles:
        Q = 0
        peak = 0
        for t in range(T):
            h = t % 24
            surge = ak if ((h - sk) % 24) < dk else 0
            lam = base[h] + surge
            Q = Q + lam - cap[h]
            if Q < 0:
                Q = 0
            if t >= 24 and Q > peak:
                peak = Q
        if peak > worst:
            worst = peak
    return worst


def uniform_pattern(A, W, offset):
    n = len(A)
    return [A[(offset + (i * n) // W) % n] for i in range(W)]


def main():
    toks = sys.stdin.read().split()
    idx = 0
    W = int(toks[idx]); idx += 1
    L = int(toks[idx]); idx += 1
    C = int(toks[idx]); idx += 1
    g = int(toks[idx]); idx += 1
    r = int(toks[idx]); idx += 1
    base = [int(toks[idx + i]) for i in range(24)]; idx += 24
    K = int(toks[idx]); idx += 1
    profiles = []
    for _ in range(K):
        sk = int(toks[idx]); dk = int(toks[idx + 1]); ak = int(toks[idx + 2])
        idx += 3
        profiles.append((sk, dk, ak))

    # KEY INSIGHT: headcount (W) and shift length (L) are fixed by the
    # instance -- the only thing we truly control is WHERE, in phase, the
    # unavoidable low-coverage hours of a uniform roster fall. A uniform
    # spread of W shifts of length L has a fixed *shape* of coverage dips;
    # rotating which allowed hour we anchor it at slides that dip shape
    # around the clock without changing total staff-hours at all. Since the
    # surge sweep's start hours are NOT uniformly spread over the day (the
    # published profiles cluster on a lattice), most rotations dodge them
    # completely even though every rotation uses exactly the same W workers.
    #
    # We search every phase offset (cheap: |A| <= 24 candidates) and keep
    # the one whose worst-case peak queue over the whole surge sweep is
    # smallest -- an exchange/rotation argument over the roster's phase,
    # not "greedy + more effort".
    A = allowed_hours(g, r)
    n = len(A)

    best_starts = None
    best_val = None
    for offset in range(n):
        cand = uniform_pattern(A, W, offset)
        val = worst_case_peak(cand, L, C, base, profiles)
        if best_val is None or val < best_val:
            best_val = val
            best_starts = cand

    # also try a couple of non-uniform variants: bias extra coverage toward
    # the hour-of-day with the highest base demand, keeping headcount fixed,
    # to squeeze a little more out of the same W workers.
    peak_hour = max(range(24), key=lambda h: base[h])
    closest = min(A, key=lambda h: min((h - peak_hour) % 24, (peak_hour - h) % 24))
    for target in range(W):
        # take the best uniform pattern and swap one worker's start to the
        # allowed hour closest to the base-demand peak, if that improves
        # the worst-case peak over the sweep.
        cand = list(best_starts)
        cand[target] = closest
        val = worst_case_peak(cand, L, C, base, profiles)
        if val < best_val:
            best_val = val
            best_starts = cand

    print(" ".join(str(x) for x in best_starts))


if __name__ == "__main__":
    main()
