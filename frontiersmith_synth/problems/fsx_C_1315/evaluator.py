#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1315 -- "Telling People Where To Go When They Will Not Listen"
(family: evacuation-guidance-policy; format B, quality-metric).

THEME.  A venue must be cleared through several exits.  People are grouped into
ZONES.  Every step, some people newly step out of their zone (a doorway/egress
rate limits how many can even START moving per step) and either (a) FOLLOW the
guidance broadcast for their zone this step, or (b) IGNORE it and head for their
zone's DEFAULT (nearest) exit anyway -- the fraction that follows is the zone's
CURRENT effective compliance.  Compliance starts at each zone's baseline rate but
DECAYS every time the guidance for that zone CHANGES from the previous step
(contradictory messaging erodes trust) and creeps back up while the message
stays the same (consistency rebuilds trust).  Exits have a nominal per-step
throughput; when the queue at an exit exceeds that throughput the excess backs
up AND degrades the exit's own effective throughput for as long as the backlog
persists (a crush: congestion begets more congestion).  The candidate submits
ONE guidance grid (which exit each zone is told to use, every step) up front;
the evaluator simulates the whole horizon deterministically and scores total
people evacuated.

MECHANISMS COMPOSED: compliance-probability (partial, per-zone, eroded/rebuilt
by message stability) x route-congestion-feedback (exit backlog persists and
compounds) x information-credibility-decay (changing a zone's directive costs
trust, multiplicatively, forever until rebuilt).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the full
          schema; every dynamics constant the sim uses is IN this object.
  stdout: ONE JSON object: {"guidance": [[g_i0, g_i1, ..., g_i(T-1)], ...]}
          Z rows (one per zone), each of length T, g in [0, n_exits) and
          reachable[i][g] must be 1.  Any shape/type/range/reachability
          violation, a crash, timeout, or non-JSON output -> that instance
          scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes,
itself, two references by running the SAME simulator:
    q_lb   = min(sum(population), sum(capacity)*T)   # congestion-free, full-
             throughput, unreachable-in-practice ideal
    q_base = evacuated total under the "always tell every zone its own default
             (nearest) exit, forever" policy -- the naive, do-nothing-clever
             recommendation (guided == default, so compliance is moot and the
             directive never changes)
    q_cand = evacuated total achieved by running the CANDIDATE's guidance grid
             through the simulator
  and normalize with an affine anchor (naive baseline -> 0.1, ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, q_lb - q_base), 0, 1 )
  Doing worse than the naive "just point at the nearest exit" baseline scores
  BELOW 0.1 (this happens to reactive, full-compliance-assuming rerouting on
  the low-compliance / shared-bottleneck instances -- see statement.md).

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance (which, here, IS
the full instance -- an offline planning problem with no held-out oracle data,
same shape as the format's other quality-metric problems).

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    def nxtf(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u = ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)
        return lo + u * (hi - lo)

    return nxt, nxtf


# ----------------------------- instance family ------------------------------
def _build_instance(spec):
    """spec: dict with seed + structural knobs. Returns a full instance dict
    (everything the candidate needs is PUBLIC -- offline full-information
    planning problem)."""
    seed = spec["seed"]
    nxt, nxtf = _rng(seed)
    Z, E, T = spec["Z"], spec["E"], spec["T"]

    # positions on a line [0,100] for zones and exits -> distance -> reachability
    zpos = [nxtf(0, 100) for _ in range(Z)]
    if spec.get("bottleneck"):
        # cluster most zones near exit 0 to force a shared "obvious" default exit
        epos = [3.0] + [nxtf(40, 100) for _ in range(E - 1)]
        for i in range(Z):
            if nxt(0, 99) < spec.get("cluster_pct", 75):
                zpos[i] = nxtf(0, 15)
    else:
        epos = [nxtf(0, 100) for _ in range(E)]

    dist = [[abs(zpos[i] - epos[e]) for e in range(E)] for i in range(Z)]
    reach_radius = spec.get("reach_radius", 65.0)
    reachable = [[1 if dist[i][e] <= reach_radius else 0 for e in range(E)] for i in range(Z)]
    default_exit = []
    for i in range(Z):
        order = sorted(range(E), key=lambda e: dist[i][e])
        best = order[0]
        reachable[i][best] = 1  # default exit is always reachable
        default_exit.append(best)

    pop_lo, pop_hi = spec.get("pop_range", (30, 160))
    population = [round(nxtf(pop_lo, pop_hi), 2) for _ in range(Z)]

    clo, chi = spec.get("compliance_range", (0.3, 0.75))
    base_compliance = [round(nxtf(clo, chi), 3) for _ in range(Z)]

    cap_lo, cap_hi = spec.get("cap_range", (10, 26))
    capacity = [round(nxtf(cap_lo, cap_hi), 2) for _ in range(E)]

    elo, ehi = spec.get("egress_range", (10, 30))
    egress_cap = [round(nxtf(elo, ehi), 2) for _ in range(Z)]

    dlo, dhi = spec.get("decay_range", (0.45, 0.8))
    credibility_decay = [round(nxtf(dlo, dhi), 3) for _ in range(Z)]
    rlo, rhi = spec.get("recover_range", (0.04, 0.12))
    credibility_recover = [round(nxtf(rlo, rhi), 3) for _ in range(Z)]

    blo, bhi = spec.get("beta_range", (2.5, 6.0))
    congestion_beta = [round(nxtf(blo, bhi), 3) for _ in range(E)]

    return {
        "name": spec["name"], "n_zones": Z, "n_exits": E, "T": T,
        "population": population, "capacity": capacity, "egress_cap": egress_cap,
        "base_compliance": base_compliance, "default_exit": default_exit,
        "credibility_decay": credibility_decay, "credibility_recover": credibility_recover,
        "congestion_beta": congestion_beta, "reachable": reachable,
    }


def _build_instances():
    specs = [
        dict(name="ev01_balanced", seed=101, Z=5, E=3, T=8, bottleneck=False,
             compliance_range=(0.45, 0.8), cap_range=(14, 26)),
        dict(name="ev02_balanced_wide", seed=102, Z=6, E=3, T=9, bottleneck=False,
             compliance_range=(0.4, 0.85), cap_range=(12, 24)),
        dict(name="ev03_shared_hicompl", seed=103, Z=6, E=3, T=8, bottleneck=True,
             cluster_pct=70, compliance_range=(0.65, 0.9), cap_range=(12, 22)),
        dict(name="ev04_shared_locompl_TRAP", seed=104, Z=6, E=3, T=8, bottleneck=True,
             cluster_pct=78, compliance_range=(0.1, 0.28), cap_range=(11, 20)),
        dict(name="ev05_shared_locompl_tight_TRAP", seed=105, Z=7, E=3, T=7, bottleneck=True,
             cluster_pct=80, compliance_range=(0.12, 0.3), cap_range=(10, 18),
             pop_range=(40, 170)),
        dict(name="ev06_shared_locompl_wide_TRAP", seed=106, Z=8, E=4, T=9, bottleneck=True,
             cluster_pct=75, compliance_range=(0.15, 0.32), cap_range=(10, 20)),
        dict(name="ev07_mixed_compliance", seed=107, Z=6, E=3, T=9, bottleneck=True,
             cluster_pct=60, compliance_range=(0.2, 0.85), cap_range=(12, 24)),
        dict(name="ev08_many_zones", seed=108, Z=8, E=4, T=10, bottleneck=False,
             compliance_range=(0.35, 0.8), cap_range=(12, 24)),
        dict(name="ev09_tight_capacity", seed=109, Z=6, E=3, T=6, bottleneck=True,
             cluster_pct=72, compliance_range=(0.18, 0.35), cap_range=(9, 16),
             pop_range=(45, 180)),
        dict(name="ev10_holdout_large", seed=110, Z=9, E=4, T=10, bottleneck=True,
             cluster_pct=68, compliance_range=(0.25, 0.6), cap_range=(11, 22),
             pop_range=(35, 170)),
    ]
    return [_build_instance(s) for s in specs]


# ----------------------------- simulator -------------------------------------
def _simulate(inst, guidance_fn):
    """guidance_fn(i, t) -> exit index. Runs the full deterministic fluid sim
    and returns total evacuated by horizon T. `guidance_fn` may be a lookup
    into a candidate's fixed grid, or a fixed-policy closure (for baselines)."""
    Z, E, T = inst["n_zones"], inst["n_exits"], inst["T"]
    pop = inst["population"]; cap = inst["capacity"]; egress = inst["egress_cap"]
    base_c = inst["base_compliance"]; default_exit = inst["default_exit"]
    decay = inst["credibility_decay"]; recover = inst["credibility_recover"]
    beta = inst["congestion_beta"]

    remaining = list(pop)
    pending = [[0.0] * E for _ in range(Z)]
    cred = [1.0] * Z
    prev_g = [None] * Z
    evacuated = 0.0

    for t in range(T):
        for i in range(Z):
            g = guidance_fn(i, t)
            if t > 0 and prev_g[i] is not None:
                if g != prev_g[i]:
                    cred[i] *= decay[i]
                else:
                    cred[i] = min(1.0, cred[i] + recover[i])
            prev_g[i] = g
            eff_c = base_c[i] * cred[i]
            depart = remaining[i] if remaining[i] < egress[i] else egress[i]
            remaining[i] -= depart
            guided_amt = depart * eff_c
            default_amt = depart - guided_amt
            pending[i][g] += guided_amt
            pending[i][default_exit[i]] += default_amt
        for e in range(E):
            total_wait = sum(pending[i][e] for i in range(Z))
            if total_wait <= 1e-12:
                continue
            ratio = total_wait / cap[e]
            mult = 1.0 if ratio <= 1.0 else 1.0 / (1.0 + beta[e] * (ratio - 1.0))
            cap_eff = cap[e] * mult
            served = total_wait if total_wait < cap_eff else cap_eff
            frac = served / total_wait
            for i in range(Z):
                s = pending[i][e] * frac
                pending[i][e] -= s
                evacuated += s
    return evacuated


def _baseline_evacuated(inst):
    de = inst["default_exit"]
    return _simulate(inst, lambda i, t: de[i])


def _ideal_ub(inst):
    return min(sum(inst["population"]), sum(inst["capacity"]) * inst["T"])


# ----------------------------- validation ------------------------------------
def _validate_and_run(inst, answer):
    """Return evacuated total, or None if the answer is infeasible/malformed."""
    if not isinstance(answer, dict):
        return None
    guidance = answer.get("guidance")
    if not isinstance(guidance, list):
        return None
    Z, T, E = inst["n_zones"], inst["T"], inst["n_exits"]
    reach = inst["reachable"]
    if len(guidance) != Z:
        return None
    grid = []
    for i in range(Z):
        row = guidance[i]
        if not isinstance(row, list) or len(row) != T:
            return None
        r_i = []
        for g in row:
            if isinstance(g, bool) or not isinstance(g, int):
                return None
            if g < 0 or g >= E or not reach[i][g]:
                return None
            r_i.append(g)
        grid.append(r_i)
    return _simulate(inst, lambda i, t: grid[i][t])


# ----------------------------- scoring driver ---------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        q_lb = _ideal_ub(inst)
        q_base = _baseline_evacuated(inst)
        denom = q_lb - q_base
        if denom < 1e-9:
            denom = 1e-9
        public = {k: v for k, v in inst.items()}  # everything is public (full-info planning)
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _validate_and_run(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_cand - q_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
