#!/usr/bin/env python3
"""Generator for fsx_A_1174 -- Rate constants from concentration snapshots.

`python3 gen.py <testId>` prints ONE instance to stdout.

The network is a disjoint union of 2-species reaction modules:
  - "chain" modules: a single forward mass-action edge u->v with a SLOW rate k
    (this rate is fully identifiable from the snapshot trajectory).
  - "pair" modules: a reversible forward/reverse pair u<->v (kf, kr) whose sum
    kf+kr is FAST relative to the first snapshot time T1, so the pair has
    already relaxed to its mass-action equilibrium by the time of the very
    first observation. Only the ratio kf/kr (fixed by the observed equilibrium
    split of the conserved total mass u+v) is determined by the data; kf and
    kr individually are NOT identifiable from these snapshots.

Only the topology, snapshot TIMES and NOISY snapshot CONCENTRATIONS are printed.
The true rate constants are NEVER printed -- they are exactly what the solver
must recover (or, for the fast pairs, recognize as partially unrecoverable).
This same `build_instance` function is duplicated verbatim in verify.py so the
checker can reconstruct the identical ground truth (incl. the held-out
initial condition) from testId alone, without any importable shared module.
"""
import sys, math, random

# ---------------------------------------------------------------------------
# Fixed, public constants (also duplicated in verify.py; not secret).
# ---------------------------------------------------------------------------
K_MAX = 400.0
N_CHAIN = 5
N_PAIR = 3
T1 = 0.1
EXTRA_TIMES = [0.5, 1.5, 3.5, 7.0, 12.0]
HELD_TIMES = [0.003, 0.01, 0.03, 0.15, 1.0, 4.0]
BASELINE_CHAIN_RATE = 0.2
BASELINE_PAIR_RATE = 50.0
CHAIN_K_RANGE = (0.03, 0.9)
PAIR_S_RANGE = (200.0, 380.0)
PAIR_R_RANGE = (0.08, 12.0)
SEED_BASE = 90000
SEED_MULT = 131


def build_instance(testid):
    """Deterministically build the full instance (incl. hidden ground truth
    and the held-out probe) from testid alone. Identical in gen.py & verify.py."""
    rng = random.Random(SEED_BASE + SEED_MULT * testid)
    noise_amp = 0.008 + 0.006 * testid

    modules = []
    true_rates = {}
    eid = 0
    sid = 0
    for _ in range(N_CHAIN):
        u, v = sid, sid + 1
        sid += 2
        k = rng.uniform(*CHAIN_K_RANGE)
        e = eid; eid += 1
        true_rates[e] = k
        modules.append({'id': len(modules), 'type': 'chain', 'u': u, 'v': v, 'edges': [e]})
    for _ in range(N_PAIR):
        u, v = sid, sid + 1
        sid += 2
        S = rng.uniform(*PAIR_S_RANGE)
        r = rng.uniform(*PAIR_R_RANGE)
        kf = S * r / (1.0 + r)
        kr = S / (1.0 + r)
        e1 = eid; eid += 1
        e2 = eid; eid += 1
        true_rates[e1] = kf
        true_rates[e2] = kr
        modules.append({'id': len(modules), 'type': 'pair', 'u': u, 'v': v, 'edges': [e1, e2]})
    n_species = sid

    ic = [0.0] * n_species
    for m in modules:
        u, v = m['u'], m['v']
        ic[u] = rng.uniform(1.0, 10.0)
        ic[v] = rng.uniform(1.0, 10.0)

    times = [T1] + EXTRA_TIMES

    def sim_module(mtype, u0, v0, t, rate_list):
        if mtype == 'chain':
            k = max(0.0, rate_list[0])
            eu = math.exp(-k * t)
            cu = u0 * eu
            cv = v0 + u0 * (1.0 - eu)
        else:
            kf = max(0.0, rate_list[0]); kr = max(0.0, rate_list[1])
            S = kf + kr
            Mtot = u0 + v0
            if S <= 1e-12:
                cu, cv = u0, v0
            else:
                u_eq = Mtot * kr / S
                v_eq = Mtot * kf / S
                eS = math.exp(-S * t)
                cu = u_eq + (u0 - u_eq) * eS
                cv = v_eq + (v0 - v_eq) * eS
        return cu, cv

    snapshots = []
    for t in times:
        conc = [0.0] * n_species
        for m in modules:
            u, v = m['u'], m['v']
            rl = [true_rates[e] for e in m['edges']]
            cu, cv = sim_module(m['type'], ic[u], ic[v], t, rl)
            conc[u], conc[v] = cu, cv
        noisy = [c * (1.0 + rng.uniform(-noise_amp, noise_amp)) for c in conc]
        snapshots.append(noisy)

    ic2 = [0.0] * n_species
    for m in modules:
        u, v = m['u'], m['v']
        if m['type'] == 'pair':
            M2 = rng.uniform(3.0, 15.0)
            if rng.random() < 0.5:
                ic2[u], ic2[v] = M2, 0.0
            else:
                ic2[u], ic2[v] = 0.0, M2
        else:
            ic2[u] = rng.uniform(1.0, 10.0)
            ic2[v] = rng.uniform(0.0, 5.0)

    return dict(n_species=n_species, modules=modules, true_rates=true_rates,
                ic=ic, times=times, snapshots=snapshots, ic2=ic2,
                held_times=HELD_TIMES, sim_module=sim_module)


def main():
    testid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if testid < 1:
        testid = 1
    inst = build_instance(testid)
    modules = inst['modules']
    n_edges = sum(len(m['edges']) for m in modules)
    out = []
    out.append(str(testid))
    out.append("%d %d %d %d %.4f" % (inst['n_species'], len(modules), n_edges,
                                      len(inst['times']), K_MAX))
    for m in modules:
        tcode = 0 if m['type'] == 'chain' else 1
        out.append("%d %d %d %d" % (m['id'], tcode, m['u'], m['v']))
    for m in modules:
        if m['type'] == 'chain':
            out.append("%d %d %d %d" % (m['edges'][0], m['id'], m['u'], m['v']))
        else:
            out.append("%d %d %d %d" % (m['edges'][0], m['id'], m['u'], m['v']))
            out.append("%d %d %d %d" % (m['edges'][1], m['id'], m['v'], m['u']))
    out.append(" ".join("%.6f" % t for t in inst['times']))
    for snap in inst['snapshots']:
        out.append(" ".join("%.8f" % c for c in snap))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
