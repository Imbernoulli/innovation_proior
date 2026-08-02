#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE crystallization-cool-schedule instance to stdout.
Deterministic: seeded only by testId. Difficulty ladder small -> large/adversarial.

A batch cooling crystallizer starts exactly saturated at T_start. The solver must
choose (a) which of K seed options to charge at t=0 and (b) a cooling schedule
T_1..T_N (temperature after each of N discrete batch steps, monotonic non-increasing,
T_min <= T_i <= T_start). A deterministic population-balance moment model (McCabe
Delta-L growth law + a supersaturation-driven primary nucleation term, sub-stepped
for numerical fidelity) turns that choice into a final crystal population; the score
is the MEAN crystal size at the end, subject to reaching a required crystallized-mass
yield within the N steps.

Two mechanisms compose to create a genuine trap for "cool at a constant rate":
  1. Nucleation rate is a strongly convex function of supersaturation S (exponent
     b > 2.5): B = kb * S^b. Jumping straight to T_min maximizes S instantly and
     detonates a massive nucleation burst (powder) -- this is the checker's own
     internal reference baseline (trivial), deliberately bad.
  2. The solubility curve (given as explicit breakpoints in the input) is STEEP
     near T_min and NEAR-FLAT near T_start (most of the crystallizable mass sits in
     the bottom slice of the temperature range). A CONSTANT cooling rate (linear
     schedule) therefore spends most of its steps generating almost no
     supersaturation, then rushes through the steep slice late, generating a big
     supersaturation pulse with little batch time left for the resulting crystals
     to grow -- fines, or an outright missed yield target. The insight is a
     cooling-rate-FALLS-over-time schedule (front-loaded, decelerating) that enters
     the steep slice EARLY, while most of the batch time remains for growth to
     consume the released solute without renucleating.

Seeding compounds this: an "obvious" heuristic that loads the most seed MASS
(few big seeds) starts with little seed surface area, so it does not blunt the
early supersaturation pulse either -- the insightful choice is a seed option with
good surface area (mu2 = count*radius^2) relative to its mass.

Plants >=5 of 10 "high-yield" trap cases where the required yield is pushed close
to what only an early-and-decelerating schedule can reach in time; a naive
linear-rate schedule paired with a big-seed-mass choice lands far below the
insightful (front-loaded, moderate-area-seed) schedule on mean crystal size.
"""
import sys
import random

FSUB = 10  # internal fine sub-steps per reported macro cooling step (checker-side realism)


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
            dmu0 = B
            dmu1 = G * mu0 * dt + B * r0
            dmu2 = 2.0 * G * mu1 * dt
            dmu3 = 3.0 * G * mu2 * dt
            mu0 += dmu0
            mu1 += dmu1
            mu2 += dmu2
            mu3 += dmu3
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


def build(tid):
    rng = random.Random(70019 + 131 * tid)

    n_sizes = {1: 22, 2: 24, 3: 28, 4: 30, 5: 34, 6: 38, 7: 42, 8: 46, 9: 52, 10: 58}
    N = n_sizes[tid]
    trap_ids = {2, 4, 5, 7, 8, 9, 10}  # >=5 of 10: high-yield traps
    is_trap = tid in trap_ids

    T_start = 100.0
    dT = round(rng.uniform(48.0, 59.5), 2)
    T_min = round(T_start - dT, 2)

    Cmin = round(rng.uniform(14.0, 22.0), 2)
    dC = round(rng.uniform(64.0, 82.0), 2)
    Cmax = round(Cmin + dC, 2)

    # concave solubility curve: steep near T_min, near-flat near T_start.
    # 4 successive intervals over [T_min, T_start] at fractions [0,.14,.30,.58,1.0]
    fracs = [0.0, 0.14, 0.30, 0.58, 1.0]
    w = [0.50, 0.27, 0.15, 0.08]  # base share of dC per interval, steepest first
    jitter = [rng.uniform(0.9, 1.1) for _ in range(4)]
    shares = [wi * ji for wi, ji in zip(w, jitter)]
    ssum = sum(shares)
    shares = [s / ssum for s in shares]
    bps = [(T_min, Cmin)]
    acc = Cmin
    for i in range(4):
        acc += dC * shares[i]
        T_here = round(T_min + fracs[i + 1] * dT, 3)
        bps.append((T_here, round(acc, 3)))
    bps[-1] = (T_start, Cmax)

    kb = round(rng.uniform(2.2e-4, 3.8e-4), 8)
    b = round(rng.uniform(3.0, 3.4), 3)
    kg = round(rng.uniform(0.017, 0.023), 5)
    g = 1.0
    r0 = 0.02
    kv_rho = 0.01

    # seed library: archetypes present in every case (positions shuffled per test)
    # 0: negligible seeding (tiny)
    # 1: few big seeds -> high total MASS, low total AREA (the "load up on mass" trap)
    # 2: moderate count/radius -> good AREA-to-mass ratio (a good pick)
    # 3: many small seeds -> lots of area but low individual size (dilutes mean size)
    # 4: another moderate, slightly different point (a second good pick)
    lib = [
        (round(rng.uniform(0.6, 1.6), 2), round(rng.uniform(0.03, 0.06), 3)),
        (round(rng.uniform(3.5, 6.5), 2), round(rng.uniform(0.26, 0.36), 3)),
        (round(rng.uniform(14.0, 24.0), 2), round(rng.uniform(0.12, 0.17), 3)),
        (round(rng.uniform(60.0, 95.0), 2), round(rng.uniform(0.06, 0.09), 3)),
        (round(rng.uniform(28.0, 44.0), 2), round(rng.uniform(0.10, 0.14), 3)),
    ]
    order = list(range(5))
    rng.shuffle(order)
    lib = [lib[i] for i in order]
    K = len(lib)

    # required_yield: fraction of what an immediate max-cooling schedule (the
    # checker's own baseline construction) actually achieves within N steps --
    # guarantees the baseline stays feasible with comfortable margin.
    fast_yield, _ = simulate(N, T_start, T_min, kb, b, kg, g, r0, kv_rho, bps,
                              lib[0][0], lib[0][1], fastcool_sched(N, T_min))
    if is_trap:
        frac = rng.uniform(0.72, 0.86)
    else:
        frac = rng.uniform(0.30, 0.52)
    required_yield = round(frac * fast_yield, 4)
    required_yield = max(0.05, min(required_yield, 0.94 * fast_yield))

    lines = [f"{N} {T_start} {T_min}",
             f"{kb} {b} {kg} {g} {r0} {kv_rho}",
             f"{len(bps)}"]
    for (T, Ceq) in bps:
        lines.append(f"{T} {Ceq}")
    lines.append(f"{required_yield}")
    lines.append(f"{K}")
    for (cnt, rad) in lib:
        lines.append(f"{cnt} {rad}")
    return "\n".join(lines) + "\n"


def main():
    tid = int(sys.argv[1])
    sys.stdout.write(build(tid))


if __name__ == "__main__":
    main()
