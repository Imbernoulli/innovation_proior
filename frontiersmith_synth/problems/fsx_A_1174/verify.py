#!/usr/bin/env python3
"""Deterministic checker for fsx_A_1174 (Format C, reaction-rate-recover).

Usage: python3 verify.py <in> <out> <ans>
  <in>  : the instance as printed by gen.py (testId, topology, snapshot data)
  <out> : the participant artifact -- one "edge_id rate" line per edge
  <ans> : ignored (empty placeholder)

The checker reconstructs the FULL ground truth (true rate constants, plus a
held-out initial condition and comparison times) purely from testId via the
same `build_instance` function shipped in gen.py (duplicated here verbatim --
no importable shared module, so nothing extra is exposed to a solution even
if it could read the problem directory). It then:
  * strictly validates the submitted rate vector (schema, range, finiteness),
  * simulates BOTH the true and the submitted rate constants forward from the
    held-out initial condition (closed-form, exact, no ODE integration),
  * scores by conservation-normalised RMS trajectory error on that held-out
    probe, normalised against an internal flat-rate baseline.

Any violation -> `Ratio: 0.0`. Deterministic and O(size).
"""
import sys, math, random

# ---------------------------------------------------------------------------
# Fixed, public constants (identical to gen.py; not secret).
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


def fail(msg):
    sys.stdout.write("reason: %s\n" % msg)
    sys.stdout.write("Ratio: 0.0\n")
    sys.exit(0)


def held_out_error(inst, rates):
    """Conservation-normalised trajectory error of `rates` on the held-out
    initial condition, over HELD_TIMES. `rates`: dict edge_id->float.

    Each module contributes one RMS term, averaged with EQUAL weight across
    modules (not pooled across all species*time terms): a network mixes a
    slow module (small concentration swings) with a fast module (large,
    near-instant swings), and pooling would let the fast modules' larger
    absolute residuals swamp the slow modules' signal entirely. Per-module
    averaging keeps both the "fit the identifiable slow rates" and the
    "get the fast-pair ratio/scale right" components of the objective
    visible in the final score."""
    modules = inst['modules']
    ic2 = inst['ic2']
    sim_module = inst['sim_module']
    mod_rms = []
    for m in modules:
        u, v = m['u'], m['v']
        u0, v0 = ic2[u], ic2[v]
        Mtot = max(u0 + v0, 1e-6)
        rl_true = [inst['true_rates'][e] for e in m['edges']]
        rl_pred = [rates[e] for e in m['edges']]
        sq_acc = 0.0
        n_terms = 0
        for t in inst['held_times']:
            cu_t, cv_t = sim_module(m['type'], u0, v0, t, rl_true)
            cu_p, cv_p = sim_module(m['type'], u0, v0, t, rl_pred)
            sq_acc += ((cu_p - cu_t) / Mtot) ** 2
            sq_acc += ((cv_p - cv_t) / Mtot) ** 2
            n_terms += 2
        mod_rms.append(math.sqrt(sq_acc / max(1, n_terms)))
    return sum(mod_rms) / max(1, len(mod_rms))


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        in_toks = open(in_path).read().split()
        testid = int(in_toks[0])
    except Exception:
        fail("cannot read/parse instance")

    try:
        inst = build_instance(testid)
    except Exception:
        fail("cannot rebuild instance")

    modules = inst['modules']
    all_edges = set()
    for m in modules:
        for e in m['edges']:
            all_edges.add(e)
    n_edges = len(all_edges)

    # ---- read + validate participant output ----
    try:
        raw = open(out_path).read()
    except Exception:
        fail("cannot read output")
    if len(raw) > 2_000_000:
        fail("output too large")

    rates = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            fail("malformed line (expected 'edge_id rate'): %r" % line[:80])
        eid_tok, rate_tok = parts
        try:
            eid = int(eid_tok)
        except Exception:
            fail("non-integer edge id: %r" % eid_tok[:40])
        try:
            rate = float(rate_tok)
        except Exception:
            fail("non-numeric rate: %r" % rate_tok[:40])
        if not math.isfinite(rate):
            fail("non-finite rate on edge %d" % eid)
        if eid not in all_edges:
            fail("unknown edge id %d" % eid)
        if eid in rates:
            fail("duplicate edge id %d" % eid)
        if rate < -1e-9 or rate > K_MAX + 1e-9:
            fail("rate %.6f on edge %d out of [0, %.1f]" % (rate, eid, K_MAX))
        rates[eid] = max(0.0, rate)

    if len(rates) != n_edges:
        missing = sorted(all_edges - set(rates.keys()))
        fail("missing %d/%d edges (e.g. %s)" % (n_edges - len(rates), n_edges, missing[:5]))

    # ---- baseline: flat per-reaction-class guess (same magnitude regardless
    # of the module's actual data -- chain edges get a generic "slow" default,
    # pair edges get a generic "fast, 1:1" default) ----
    base_rates = {}
    for m in modules:
        rate = BASELINE_CHAIN_RATE if m['type'] == 'chain' else BASELINE_PAIR_RATE
        for e in m['edges']:
            base_rates[e] = rate
    B = held_out_error(inst, base_rates)
    F = held_out_error(inst, rates)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    sys.stdout.write("baseline_B=%.8f participant_F=%.8f\n" % (B, F))
    sys.stdout.write("Ratio: %.6f\n" % ratio)


if __name__ == "__main__":
    main()
