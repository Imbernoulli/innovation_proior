# TIER: strong
# Insight (water-filling / KKT): since power is v^2, spreading a fixed
# distance D_i over more ticks at a lower, near-CONSTANT speed is strictly
# cheaper in energy than rushing (energy for a constant-speed run of
# distance D_i over T ticks is D_i^2 / T -- convex, minimized by evenness).
# So instead of serializing arms (greedy), run ALL arms in parallel, every
# tick, each at its own near-constant speed D_i/T -- the makespan T that
# arms literally CANNOT beat (by Cauchy-Schwarz, any feasible schedule has
# T >= sqrt(sum D_i^2 / P), and also T >= max_i ceil(D_i/floor(sqrt(P)))
# since one tick's speed alone can never exceed floor(sqrt(P))). The two
# lower bounds together give the smallest makespan a parallel/simultaneous
# schedule can hope to hit; from there we search a small window around the
# unconstrained cost-optimal T* = sqrt(sum D_i^2 / A) (balancing the A*T
# term against the 1/T energy term) for the true minimizer. Within a fixed
# T, per-arm speeds are built with an exact "balanced sequence" (like
# Bresenham rounding, phase-shifted per arm) so nobody's speed ever swings
# by more than 1 -- this equalizes the marginal delay-per-watt across arms
# instead of ever maxing one arm out while others sit idle.
import math
import sys


def build_rows(D, T, K):
    phases = [(i * 977) % T for i in range(K)]
    prev_f = [0] * K  # f(0) = phase_i // T = 0 for all i since phase_i < T
    rows = []
    tick_powers = []
    ok = True
    for t in range(1, T + 1):
        row = [0] * K
        tp = 0
        for i in range(K):
            f_t = (D[i] * t + phases[i]) // T
            v = f_t - prev_f[i]
            prev_f[i] = f_t
            row[i] = v
            tp += v * v
        rows.append(row)
        tick_powers.append(tp)
    return rows, tick_powers


def isqrt_ceil(numer, denom):
    """ceil(sqrt(numer/denom)) using exact integer arithmetic."""
    if numer <= 0:
        return 0
    lo, hi = 1, 2
    while hi * hi * denom < numer:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if mid * mid * denom >= numer:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    data = sys.stdin.read().split("\n")
    K, P, A = (int(x) for x in data[0].split())
    D = [int(x) for x in data[1].split()]

    vmax = math.isqrt(P)
    if vmax < 1:
        vmax = 1

    S = sum(d * d for d in D)
    Tcrit = max((d + vmax - 1) // vmax for d in D)
    Tfeas = max(Tcrit, isqrt_ceil(S, P))
    Tstar = math.sqrt(S / A) if A > 0 else Tfeas

    lo = max(1, Tfeas)
    hi = max(lo, int(Tstar) + 5) + 5

    best = None  # (cost, T, rows)
    T = lo
    tries = 0
    # First pass: scan the calibrated window.
    for T in range(lo, hi + 1):
        rows, tick_powers = build_rows(D, T, K)
        if max(tick_powers) > P:
            continue
        energy = sum(tick_powers)
        cost = A * T + energy
        if best is None or cost < best[0]:
            best = (cost, T, rows)

    # Safety net: if nothing in the window was feasible (can happen only on
    # pathological rounding), keep growing T until a feasible schedule is
    # found -- guaranteed to terminate since average tick power -> 0 as T
    # grows, while Tfeas already lower-bounds the true optimum.
    T = hi + 1
    while best is None and tries < 5000:
        rows, tick_powers = build_rows(D, T, K)
        if max(tick_powers) <= P:
            energy = sum(tick_powers)
            cost = A * T + energy
            best = (cost, T, rows)
        T += 1
        tries += 1

    _, T_final, rows = best
    out = "\n".join(" ".join(map(str, r)) for r in rows)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
