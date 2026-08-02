#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the burst-nucleation
nanoparticle synthesis problem.

Reads the instance from <in> (see gen.py for the exact layout). Reads the
participant's schedule from <out>:
    temp_1 ... temp_T          (T ints in [0, L-1])
    inject_1 ... inject_T      (T non-negative finite floats)
    surf                       (1 int in [0, S-1])

Feasibility (ANY violation -> "Ratio: 0.0"):
  - exact token count (no missing/trailing tokens)
  - every temp_t is an integer in [0, L-1]
  - every inject_t is a finite float, inject_t >= 0, inject_t <= max_inject + eps
  - sum(inject_t) <= C0 + eps
  - surf is an integer in [0, S-1]

Scoring (feasible only): run the deterministic population-balance simulation
(burst-nucleation-separation + surfactant-capping self-throttled growth +
Ostwald-ripening redistribution while the monomer pool is starved) and compute
the count-weighted in-window quality F in [0,1]. The checker builds its own
baseline B the same way, from a fixed "slow continuous nucleation" reference
schedule (constant mid-level heat, precursor fed evenly across all steps --
exactly what solutions/trivial.py submits). Score:
    ratio = min(1000, 100*F/B) / 1000
"""
import math
import sys

EPS = 1e-6


def fail(reason):
    print("INFEASIBLE:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_tokens(path):
    with open(path, "r") as f:
        return f.read().split()


def parse_instance(in_path):
    toks = read_tokens(in_path)
    pos = 0

    def nxt():
        nonlocal pos
        v = toks[pos]
        pos += 1
        return v

    def nxt_int():
        return int(nxt())

    def nxt_float():
        return float(nxt())

    T = nxt_int(); L = nxt_int(); S = nxt_int()
    r0 = nxt_float(); v0 = nxt_float()
    theta_ripen = nxt_float(); ripening_rate = nxt_float()
    thr, cap, gcoef = [], [], []
    for _ in range(L):
        thr.append(nxt_float())
        cap.append(nxt_int())
        gcoef.append(nxt_float())
    surf = []
    for _ in range(S):
        b = nxt_float(); p = nxt_float()
        surf.append((b, p))
    C0 = nxt_float(); max_inject = nxt_float()
    target = nxt_float(); disp_limit = nxt_float()
    return dict(T=T, L=L, S=S, r0=r0, v0=v0, theta_ripen=theta_ripen,
                ripening_rate=ripening_rate, thr=thr, cap=cap, gcoef=gcoef,
                surf=surf, C0=C0, max_inject=max_inject, target=target,
                disp_limit=disp_limit)


def simulate(inst, temp_sched, inject_sched, surf_idx):
    """Deterministic population-balance model. cohorts = list of
    [count, radius, coverage], one per distinct nucleation event (birth time)."""
    thr = inst['thr']; cap = inst['cap']; gcoef = inst['gcoef']
    v0 = inst['v0']; r0 = inst['r0']
    theta_ripen = inst['theta_ripen']; ripening_rate = inst['ripening_rate']
    bind_rate, p = inst['surf'][surf_idx]

    M = 0.0
    cohorts = []
    for t in range(inst['T']):
        lvl = temp_sched[t]
        M += inject_sched[t]
        # 1) burst-nucleation-separation: a step can only spawn nuclei while the
        #    monomer pool exceeds this level's threshold, and even then only up
        #    to this level's per-step kinetic cap -- a genuine burst needs BOTH
        #    a big enough excess AND a hot-enough level to burn through it in
        #    one shot; otherwise leftover excess re-triggers nucleation later.
        if M > thr[lvl]:
            max_possible = math.floor((M - thr[lvl]) / v0)
            n_new = min(max_possible, cap[lvl])
            if n_new > 0:
                M -= n_new * v0
                cohorts.append([n_new, r0, 0.0])
        # 2) surfactant-capping: each cohort's own growth self-throttles as its
        #    surfactant coverage approaches saturation.
        for c in cohorts:
            throttle = (1.0 - c[2]) ** p
            dr = gcoef[lvl] * throttle
            c[1] += dr
            c[2] = min(1.0, c[2] + bind_rate * (1.0 - c[2]) * dr)
        # 3) Ostwald ripening: while the pool is starved (below theta_ripen),
        #    below-mean particles shrink and feed above-mean ones. A population
        #    with cohorts of many different ages/radii gets pulled apart by
        #    this; a single homogeneous (same-age) cohort is a fixed point of
        #    this rule (mean == every particle -> no redistribution).
        if M < theta_ripen and cohorts:
            total_count = sum(c[0] for c in cohorts)
            r_mean = sum(c[0] * c[1] for c in cohorts) / total_count
            for c in cohorts:
                delta = ripening_rate * (c[1] - r_mean)
                c[1] = max(0.0, c[1] + delta)

    total = sum(c[0] for c in cohorts)
    if total <= 0:
        return 0.0
    target = inst['target']; disp = inst['disp_limit']
    num = 0.0
    for c in cohorts:
        q = max(0.0, 1.0 - abs(c[1] - target) / disp)
        num += c[0] * q
    return num / total


def baseline_quality(inst):
    """The checker's own reference: slow CONTINUOUS nucleation -- constant
    mid-level heat, precursor fed evenly across every step, weakest
    surfactant. Always feasible, always positive. Exactly solutions/trivial.py."""
    T = inst['T']
    temp = [2] * T
    inject = [inst['C0'] / T] * T
    return simulate(inst, temp, inject, 0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    inst = parse_instance(in_path)
    T, L, S = inst['T'], inst['L'], inst['S']

    raw = read_tokens(out_path)
    need = T + T + 1
    if len(raw) != need:
        fail(f"expected {need} tokens (T temps + T injections + 1 surfactant idx), got {len(raw)}")

    idx = 0

    def read_int_tok(lo, hi, what):
        nonlocal idx
        tok = raw[idx]; idx += 1
        try:
            v = int(tok)
        except ValueError:
            fail(f"non-integer token for {what}: {tok!r}")
        if not (lo <= v <= hi):
            fail(f"{what}={v} out of range [{lo},{hi}]")
        return v

    def read_float_tok(what):
        nonlocal idx
        tok = raw[idx]; idx += 1
        try:
            v = float(tok)
        except ValueError:
            fail(f"non-numeric token for {what}: {tok!r}")
        if not math.isfinite(v):
            fail(f"non-finite token for {what}: {tok!r}")
        return v

    temp_sched = [read_int_tok(0, L - 1, f"temp[{i}]") for i in range(T)]
    inject_sched = []
    for i in range(T):
        v = read_float_tok(f"inject[{i}]")
        if v < -EPS:
            fail(f"inject[{i}]={v} negative")
        v = max(0.0, v)
        if v > inst['max_inject'] + 1e-6 * max(1.0, inst['max_inject']):
            fail(f"inject[{i}]={v} exceeds max_inject={inst['max_inject']}")
        inject_sched.append(v)
    total_inject = sum(inject_sched)
    if total_inject > inst['C0'] + 1e-6 * max(1.0, inst['C0']):
        fail(f"total injection {total_inject} exceeds budget C0={inst['C0']}")
    surf_idx = read_int_tok(0, S - 1, "surfactant index")

    assert idx == len(raw)

    F = simulate(inst, temp_sched, inject_sched, surf_idx)
    B = baseline_quality(inst)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print(f"OK: F={F:.6f} baseline={B:.6f}")
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
