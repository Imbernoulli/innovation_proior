# TIER: strong
"""The insight: since the solubility curve is steep near T_min and nearly flat
near T_start (readable directly off the input breakpoints), the amount of
supersaturation a cooling schedule GENERATES per step is not tied to how much
temperature it removes -- it is tied to WHEN, along the schedule, it enters the
steep slice of the curve. Entering that slice early leaves most of the batch
time still available for growth to consume the released solute without
renucleating; entering it late leaves almost no time, forcing either a
nucleation burst or an outright missed yield target. So the schedule's cooling
RATE must FALL over time: drop fast while still in the flat region (cheap,
low-consequence), then decelerate through and after the steep region so growth
-- not new nuclei -- finishes the job.

This turns an intractable N-dimensional search over full temperature profiles
into a 2-dimensional one: which seed option (a surface-area vs. dilution
trade-off -- more seed area blunts an early supersaturation pulse, but too many
seed particles dilutes the FINAL mean size, since Delta-L growth adds the same
increment to every particle regardless of when it appeared) paired with a single
shape exponent p in T(t) = T_start - (T_start-T_min) * (t/N)^p. p < 1 is exactly
"cooling rate falls over time" (steep early, flat late); p >= 1 is the naive
constant-or-worse-shaped alternative. A small deterministic grid over (seed, p)
is simulated with the SAME population-balance model the checker uses, and the
best feasible (reaches the required yield) combination by mean crystal size is
emitted. This is a reformulation (search over a physically-motivated schedule
family) exploiting the planted curve-shape structure, not "greedy + more
iterations" over the raw schedule space."""
import sys

FSUB = 10
P_GRID = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
          0.50, 0.60, 0.70, 0.85, 1.00, 1.20, 1.50, 2.00, 3.00]


def ceq_interp(T, bps):
    if T <= bps[0][0]:
        return bps[0][1]
    if T >= bps[-1][0]:
        return bps[-1][1]
    for i in range(len(bps) - 1):
        T0, C0 = bps[i]
        T1, C1 = bps[i + 1]
        if T0 <= T <= T1:
            f = (T - T0) / (T1 - T0)
            return C0 + f * (C1 - C0)
    return bps[-1][1]


def simulate(N, T_start, T_min, kb, b, kg, g, r0, kv_rho, bps, seed_count, seed_radius, Tsched):
    mu0 = float(seed_count)
    mu1 = seed_count * seed_radius
    mu2 = seed_count * seed_radius ** 2
    mu3 = seed_count * seed_radius ** 3
    C0 = ceq_interp(T_start, bps)
    C = C0
    Tprev = T_start
    dt = 1.0 / FSUB
    for t in range(N):
        Tcur = Tsched[t]
        for f in range(FSUB):
            frac = (f + 1) / FSUB
            Tf = Tprev + (Tcur - Tprev) * frac
            Ceq_cur = ceq_interp(Tf, bps)
            S = C - Ceq_cur
            if S < 0.0:
                S = 0.0
            B = kb * (S ** b) * dt
            G = kg * (S ** g)
            mu0 += B
            mu1 += G * mu0 * dt + B * r0
            mu2 += 2.0 * G * mu1 * dt
            mu3 += 3.0 * G * mu2 * dt
            mass_t = kv_rho * mu3
            C = C0 - mass_t
            if C < 0.0:
                C = 0.0
        Tprev = Tcur
    mass_final = kv_rho * mu3
    yield_frac = mass_final / C0 if C0 > 1e-12 else 0.0
    mean_size = mu1 / mu0 if mu0 > 1e-9 else 0.0
    return yield_frac, mean_size


def power_sched(N, T_start, T_min, p):
    return [T_start - (T_start - T_min) * ((t + 1) / N) ** p for t in range(N)]


def fastcool_sched(N, T_min):
    return [T_min] * N


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    N = int(nxt())
    T_start = float(nxt())
    T_min = float(nxt())
    kb = float(nxt())
    b = float(nxt())
    kg = float(nxt())
    g = float(nxt())
    r0 = float(nxt())
    kv_rho = float(nxt())
    M = int(nxt())
    bps = []
    for _ in range(M):
        T = float(nxt())
        Ceq = float(nxt())
        bps.append((T, Ceq))
    required_yield = float(nxt())
    K = int(nxt())
    lib = []
    for _ in range(K):
        cnt = float(nxt())
        rad = float(nxt())
        lib.append((cnt, rad))

    best = None  # (mean_size, seed_idx, sched)
    for si, (cnt, rad) in enumerate(lib):
        for p in P_GRID:
            sched = power_sched(N, T_start, T_min, p)
            yld, ms = simulate(N, T_start, T_min, kb, b, kg, g, r0, kv_rho, bps, cnt, rad, sched)
            if yld >= required_yield - 1e-6 and (best is None or ms > best[0]):
                best = (ms, si, sched)

    if best is None:
        # fallback: no candidate in the grid reached the yield target -- fall
        # back to the checker's own always-feasible worst case so the output is
        # at least valid (score will simply be the baseline ~0.1).
        seed_idx = 1
        sched = fastcool_sched(N, T_min)
    else:
        _, si, sched = best
        seed_idx = si + 1

    out = [str(seed_idx)] + [repr(x) for x in sched]
    print(" ".join(out))


if __name__ == "__main__":
    main()
