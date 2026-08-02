#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the
crystallization-cool-schedule problem. Prints 'Ratio: <float in [0,1]>' on its
own final line. Maximization objective: higher mean final crystal size (at the
required crystallized-mass yield) is better.

Population-balance moment model (McCabe Delta-L growth law + a supersaturation
-driven primary nucleation term), sub-stepped internally for numerical realism
independent of the solver's reported macro-step count. Fully deterministic:
every input constant comes from <in>; no randomness, no wall-clock use.
"""
import sys
import math

FSUB = 10  # internal fine sub-steps per reported macro cooling step


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = toks[pos]
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
    return dict(N=N, T_start=T_start, T_min=T_min, kb=kb, b=b, kg=kg, g=g, r0=r0,
                kv_rho=kv_rho, bps=bps, required_yield=required_yield, K=K, lib=lib)


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


def fastcool_sched(N, T_min):
    return [T_min] * N


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    inst = read_instance(inp)
    N, T_start, T_min = inst['N'], inst['T_start'], inst['T_min']
    K = inst['K']
    lib = inst['lib']

    try:
        with open(outp) as f:
            toks = f.read().split()
    except Exception as e:
        fail("cannot read output: %s" % e)
        return

    if len(toks) != N + 1:
        fail("expected 1 seed index + %d temperatures, got %d tokens" % (N, len(toks)))
        return

    try:
        seed_idx = int(toks[0])
    except ValueError:
        fail("seed index token %r is not an integer" % (toks[0],))
        return
    if not (1 <= seed_idx <= K):
        fail("seed index %d out of range [1,%d]" % (seed_idx, K))
        return

    Tsched = []
    for i, tok in enumerate(toks[1:]):
        try:
            v = float(tok)
        except ValueError:
            fail("temperature token %d (%r) is not a number" % (i, tok))
            return
        if not math.isfinite(v):
            fail("non-finite temperature at step %d" % (i + 1,))
            return
        Tsched.append(v)

    # feasibility: monotone non-increasing, bounded in [T_min, T_start] (tiny eps for float slop)
    eps = 1e-6 * max(1.0, abs(T_start))
    prev = T_start
    for i, v in enumerate(Tsched):
        if v > prev + eps:
            fail("temperature increased at step %d (%.6f > previous %.6f) -- schedule must be cooling" %
                 (i + 1, v, prev))
            return
        if v < T_min - eps or v > T_start + eps:
            fail("temperature at step %d (%.6f) outside [T_min=%.6f, T_start=%.6f]" %
                 (i + 1, v, T_min, T_start))
            return
        prev = v

    cnt, rad = lib[seed_idx - 1]
    yield_frac, mean_size = simulate(N, T_start, T_min, inst['kb'], inst['b'], inst['kg'], inst['g'],
                                      inst['r0'], inst['kv_rho'], inst['bps'], cnt, rad, Tsched)
    if not (math.isfinite(yield_frac) and math.isfinite(mean_size)):
        fail("non-finite simulation result")
        return
    if yield_frac < inst['required_yield'] - 1e-6:
        fail("crystallized yield %.6f below required %.6f" % (yield_frac, inst['required_yield']))
        return
    if mean_size <= 0.0:
        fail("no crystals present at end of batch")
        return

    F = mean_size

    # Internal reference baseline: cool to T_min immediately and hold there (maximum
    # instantaneous supersaturation -> nucleation burst / powder), charged with seed
    # option 1. Always feasible by construction (gen.py sizes required_yield against
    # exactly this baseline's achieved yield).
    ref_cnt, ref_rad = lib[0]
    ref_sched = fastcool_sched(N, T_min)
    ref_yield, ref_mean = simulate(N, T_start, T_min, inst['kb'], inst['b'], inst['kg'], inst['g'],
                                    inst['r0'], inst['kv_rho'], inst['bps'], ref_cnt, ref_rad, ref_sched)
    B = ref_mean if ref_mean > 1e-9 else 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("yield=%.6f required=%.6f mean_size=%.6f baseline_mean_size=%.6f (baseline_yield=%.6f)" %
          (yield_frac, inst['required_yield'], F, B, ref_yield))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
