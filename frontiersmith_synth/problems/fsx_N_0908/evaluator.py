#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_N_0908 -- "Deterrence Beat: Staffing the Review Queue Against an
Unpredictable Swarm" (family: deterrence-coverage-review; format B, quality-metric).

THEME.  A trust & safety review queue polices K abuse categories (fake accounts, payment
fraud, phishing links, spam listings, account takeover, review manipulation, ...).  Each
round the review team has a limited THROUGHPUT budget R_t it must split across categories.
The attacker POPULATION is adaptive: whichever category currently receives strong, EFFECTIVE
review coverage gets partly deterred and MIGRATES elsewhere, following a fixed but
category-specific incentive graph M (some tactic switches are cheap, others are not).  ON
TOP of that predictable drift, every round some UNPREDICTABLE fraction of each category's
mass takes one more opportunistic hop along the SAME graph M -- a tactic shift the reviewer
could not have forecast from the deterministic law alone.  Because a submission is a single
one-shot T-round plan with no live feedback, a policy that plans PRECISELY against the
deterministic (noise-free) trajectory is exposed on every round the real population deviates
from that plan -- while a policy that spends part of its budget on a STEADY, migration-
incentive-weighted floor (covering every category roughly in proportion to how much
structural pressure the graph routes toward it, not to how much fraud that category shows
in the noise-free forecast) is robust to exactly where the unpredictable hop lands.

CANDIDATE CONTRACT (isolated stdin -> stdout program, SINGLE SHOT -- no live interaction).
The candidate sees the full deterministic law and all instance data and must submit the
WHOLE T-round allocation plan in one shot; it has no way to observe or predict the hidden
per-round noise (see NOISE below), which is generated and used ONLY inside this evaluator.
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the schema.
  stdout: ONE JSON object: {"alloc": [[r_00..r_0,K-1], ..., [r_{T-1},0..r_{T-1},K-1]]}
          alloc[t][j] >= 0 is the review capacity assigned to category j in round t;
          sum_j alloc[t][j] <= R[t] for every round.

VALIDATION.  Malformed shape/type, non-finite numbers, negative capacity, or a round that
overspends its budget (beyond a tiny float tolerance) -> that instance scores 0.0.

SCORING (deterministic; no wall-time -- randomness is SEEDED from the fixed instance seed,
never from wall-clock/OS entropy).  The evaluator REPLAYS the fixed dynamical law (including
its own seeded hidden noise) using the candidate's submitted `alloc` (see _simulate) to get
net_score = total harm PREVENTED minus a penalty for reviews WASTED on near-zero-volume
categories far beyond any reasonable proactive margin.  It also computes two INTERNAL
references purely from the instance (never seen by the candidate as a "target"):
    q_base = net_score of a UNIFORM, non-adaptive allocation (weak baseline)
    q_top  = 1.2 * max(q_base*1.3, net_score of a "match the noise-free forecast exactly"
             reference, net_score of a "steady migration-incentive-weighted floor" reference)
and normalizes with an affine anchor (weak baseline -> 0.1, an inflated near-ceiling -> 1.0):
    r = clamp( 0.1 + 0.9 * (net_score - q_base) / max(1e-9, q_top - q_base), 0, 1 )
A candidate matching the uniform baseline scores ~0.1; doing worse scores below 0.1.  q_top is
deliberately inflated above both internal references, so even an excellent candidate stays
below 1.0 -- there is real headroom.

ISOLATION.  The candidate runs in a FRESH SUBPROCESS via `isorun.run_candidate`; it only ever
sees the PUBLIC instance.  All references (and the hidden noise) are computed by THIS parent
process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

# ----------------------------- global constants (same across the family) --------------------
BETA_DEFAULT = 0.35          # deterrence/migration rate: fraction of a category's mass that
                              # migrates away per round, scaled by that round's EFFECTIVE coverage
LAM_WASTE = 0.6               # penalty per unit of capacity poured into a category far beyond
                              # both its current volume AND a reasonable proactive margin
CAP_MULT = 3.0                 # proactive margin: waste only counted past max(volume, CAP_MULT*R_t/K)
NOISE_AMP = 0.5                # hidden per-round shock amplitude (never sent to the candidate)
SLACK = 1.2                    # headroom multiplier on the internal near-ceiling reference


# ----------------------------- deterministic RNG (LCG) ---------------------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    box = [state]

    def nxt_float():
        box[0] = (box[0] * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (box[0] >> 11) / float(1 << 53)

    return nxt_float


# ----------------------------- migration-incentive graphs -------------------------------------
def _make_M(K, mode, nf):
    """Deterministic KxK migration-incentive matrix. Row i sums to 1 over j != i; M[i][i] = 0."""
    M = [[0.0] * K for _ in range(K)]
    if mode == "ring":
        for i in range(K):
            j1 = (i + 1) % K
            M[i][j1] = 0.6
            rest = [j for j in range(K) if j != i and j != j1]
            if rest:
                share = 0.4 / len(rest)
                for j in rest:
                    M[i][j] = share
            else:
                M[i][j1] = 1.0
    elif mode == "hub":
        hubs = sorted(set([0, K // 2])) if K >= 4 else [0]
        for i in range(K):
            others = [j for j in range(K) if j != i]
            if i in hubs:
                w = [0.5 + nf() for _ in others]
                s = sum(w)
                for j, wj in zip(others, w):
                    M[i][j] = wj / s
            else:
                hub_targets = [h for h in hubs if h != i] or list(hubs)
                non_hub = [j for j in others if j not in hubs]
                for h in hub_targets:
                    M[i][h] += 0.6 / len(hub_targets)
                if non_hub:
                    share = 0.4 / len(non_hub)
                    for j in non_hub:
                        M[i][j] += share
                else:
                    for h in hub_targets:
                        M[i][h] += 0.4 / len(hub_targets)
                s = sum(M[i])
                M[i] = [x / s for x in M[i]]
    elif mode == "pair":
        partner = list(range(K))
        for i in range(0, K - 1, 2):
            partner[i], partner[i + 1] = i + 1, i
        for i in range(K):
            p = partner[i]
            others = [j for j in range(K) if j != i and j != p]
            M[i][p] = 0.65
            if others:
                share = 0.35 / len(others)
                for j in others:
                    M[i][j] = share
            else:
                M[i][p] = 1.0
    else:  # "random_mild" -- softer, less structured incentive graph (generalization instances)
        for i in range(K):
            others = [j for j in range(K) if j != i]
            w = [0.3 + nf() for _ in others]
            s = sum(w)
            for j, wj in zip(others, w):
                M[i][j] = wj / s
    return M


# ----------------------------- instance family -------------------------------------------------
def _build_instance(pub_id, seed, hidden_seed, K, T, mode, V0, g, rho):
    """`pub_id` is an opaque public label carrying NO information about `seed`. `seed` drives
    every PUBLIC field (value/p0/M/V/R); a candidate is free to try to infer it from those
    (e.g. by curve-fitting the V wiggle phase) -- that reveals nothing useful, because
    `hidden_seed` -- which alone drives the per-round shock stream -- is an UNRELATED constant
    with no arithmetic relationship to `seed` or `pub_id`, and never appears in public_view()."""
    nf = _rng(seed)
    value = [round(1.0 + 4.0 * nf(), 3) for _ in range(K)]
    raw_p0 = [-math.log(1e-9 + nf()) for _ in range(K)]
    s = sum(raw_p0)
    p0 = [x / s for x in raw_p0]
    M = _make_M(K, mode, nf)
    V, R = [], []
    for t in range(T):
        wiggle = 1.0 + 0.06 * math.sin(0.7 * t + seed * 0.001)
        Vt = V0 * ((1 + g) ** t) * wiggle
        V.append(round(Vt, 4))
        Rt = rho * Vt * (1.0 + 0.05 * nf())
        R.append(round(Rt, 4))
    # HIDDEN per-round shock stream: an INDEPENDENT RNG keyed off `hidden_seed` alone. Never
    # exposed via public_view(); no field in public_view() depends on `hidden_seed`.
    hidden_nf = _rng(hidden_seed)
    shocks = [[hidden_nf() * NOISE_AMP for _ in range(K)] for _ in range(T)]
    return {
        "name": pub_id, "K": K, "T": T, "beta": BETA_DEFAULT,
        "lam_waste": LAM_WASTE, "cap_mult": CAP_MULT, "value": value, "p0": p0, "M": M,
        "V": V, "R": R, "mode": mode, "shocks": shocks,
    }


# (public_id, gen_seed, hidden_seed, K, T, mode, V0, g, rho). `public_id` is an opaque label;
# `hidden_seed` is an arbitrary constant with NO arithmetic relationship to `gen_seed` or
# `public_id` -- it cannot be derived from anything the candidate can see.
_SPECS = [
    ("queue-cascade",  1001, 7365182337, 6, 24, "ring", 160, 0.02, 0.55),
    ("queue-harbor",   1002, 2094857613, 6, 24, "hub", 180, 0.02, 0.55),
    ("queue-outpost",  1003, 8531904471, 7, 26, "ring", 200, 0.03, 0.55),   # harder / held-out
    ("queue-lattice",  1004, 4462790159, 6, 20, "pair", 150, 0.01, 0.55),
    ("queue-summit",   1005, 9271036825, 8, 26, "hub", 220, 0.03, 0.55),    # harder / held-out
    ("queue-junction", 1006, 3187650249, 5, 20, "ring", 120, 0.02, 0.55),
    ("queue-meridian", 1007, 6809214753, 6, 22, "random_mild", 150, 0.02, 0.55),
    ("queue-relay",    1008, 1548392067, 7, 24, "hub", 190, 0.03, 0.55),    # harder / held-out
    ("queue-corridor", 1009, 5926710843, 8, 26, "pair", 210, 0.02, 0.55),   # harder / held-out
    ("queue-basin",    1010, 8073461295, 5, 18, "random_mild", 130, 0.02, 0.55),
]


def make_instances():
    return [_build_instance(*spec) for spec in _SPECS]


def public_view(inst):
    return {"name": inst["name"], "K": inst["K"], "T": inst["T"], "beta": inst["beta"],
            "lam_waste": inst["lam_waste"], "cap_mult": inst["cap_mult"],
            "value": list(inst["value"]), "p0": list(inst["p0"]),
            "M": [list(row) for row in inst["M"]], "V": list(inst["V"]), "R": list(inst["R"])}


# ----------------------------- the deterministic (evaluator-side) dynamical law ----------------
def _simulate(inst, alloc):
    """Replay the coverage/migration law + HIDDEN per-round shock under `alloc` (a T x K
    matrix). Returns net_score, or None if `alloc` is infeasible (shape/type/budget violation).
    The shock is deterministic GIVEN the instance's hidden seed (reproducible), but it is never
    derivable by the candidate, which only sees `public_view(inst)`."""
    K, T = inst["K"], inst["T"]
    R, V, value, M, beta = inst["R"], inst["V"], inst["value"], inst["M"], inst["beta"]
    lam, cap_mult, shocks = inst["lam_waste"], inst["cap_mult"], inst["shocks"]

    if not isinstance(alloc, list) or len(alloc) != T:
        return None
    p = list(inst["p0"])
    total = 0.0
    for t in range(T):
        row = alloc[t]
        if not isinstance(row, list) or len(row) != K:
            return None
        clean = []
        for x in row:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return None
            fx = float(x)
            if fx != fx or fx in (float("inf"), float("-inf")) or fx < -1e-9:
                return None
            clean.append(max(0.0, fx))
        Rt, Vt = R[t], V[t]
        if sum(clean) > Rt * (1.0 + 1e-6) + 1e-6:
            return None
        Dj = (Rt / K) * cap_mult
        cov = [0.0] * K
        prevented = 0.0
        wasted = 0.0
        for j in range(K):
            vol_j = Vt * p[j]
            rj = clean[j]
            caught = min(rj, vol_j)
            prevented += value[j] * caught
            wasted += max(0.0, rj - max(vol_j, Dj))
            cov[j] = 0.0 if vol_j <= 1e-12 else min(1.0, rj / vol_j)
        total += prevented - lam * wasted
        # deterministic coverage-driven migration
        newp = [0.0] * K
        for i in range(K):
            stay = p[i] * (1.0 - beta * cov[i])
            newp[i] += stay
            leave = p[i] * beta * cov[i]
            if leave > 0:
                Mi = M[i]
                for j in range(K):
                    w = Mi[j]
                    if j != i and w > 0:
                        newp[j] += leave * w
        # HIDDEN unpredictable extra hop along the same graph (never known to the candidate)
        shock_row = shocks[t]
        final = list(newp)
        for i in range(K):
            sh = min(0.9, shock_row[i]) * newp[i]
            if sh > 0:
                final[i] -= sh
                Mi = M[i]
                for j in range(K):
                    if j != i and Mi[j] > 0:
                        final[j] += sh * Mi[j]
        s = sum(final)
        p = [x / s for x in final] if s > 0 else final
    return total


# ----------------------------- internal reference policies (never seen by the candidate) ------
def _trivial_ref(inst):
    K, T, R = inst["K"], inst["T"], inst["R"]
    return [[R[t] / K] * K for t in range(T)]


def _structural_weight(inst, iters=25):
    """Power-iterate the migration graph M (damped) to get the long-run incentive-weighted
    'where does migration pressure accumulate' vector, then weight by category value."""
    K, M, value = inst["K"], inst["M"], inst["value"]
    w = [1.0 / K] * K
    for _ in range(iters):
        nw = [0.0] * K
        for i in range(K):
            Mi = M[i]
            for j in range(K):
                if Mi[j] > 0:
                    nw[j] += w[i] * Mi[j]
        s = sum(nw)
        nw = [x / s for x in nw] if s > 1e-12 else [1.0 / K] * K
        w = [0.85 * a + 0.15 * (1.0 / K) for a in nw]
        s = sum(w)
        w = [x / s for x in w]
    ww = [w[j] * value[j] for j in range(K)]
    s = sum(ww)
    return [x / s for x in ww]


def _soft_prop_ref(inst, floor_frac, smooth):
    """Reactive core: allocate proportional to the NOISE-FREE forecast (volume * value),
    optionally blended with a static structural floor and EMA-smoothed toward last round."""
    K, T = inst["K"], inst["T"]
    V, R, value, M, beta = inst["V"], inst["R"], inst["value"], inst["M"], inst["beta"]
    w = _structural_weight(inst) if floor_frac > 0 else None
    p = list(inst["p0"])
    prev = [0.0] * K
    alloc = []
    for t in range(T):
        Vt, Rt = V[t], R[t]
        vol = [Vt * p[j] for j in range(K)]
        raw_w = [max(vol[j], 1e-9) * value[j] for j in range(K)]
        s = sum(raw_w)
        raw = [Rt * x / s for x in raw_w]
        if floor_frac > 0:
            sf = [Rt * w[j] for j in range(K)]
            raw = [(1 - floor_frac) * raw[j] + floor_frac * sf[j] for j in range(K)]
        if t and smooth > 0:
            row = [smooth * prev[j] * Rt / R[t - 1] + (1 - smooth) * raw[j] for j in range(K)]
        else:
            row = raw
        s = sum(row)
        row = [x * Rt / s for x in row]
        alloc.append(row)
        # candidate self-simulates the NOISE-FREE law (it does not know the hidden shock)
        cov = [0.0 if vol[j] <= 1e-12 else min(1.0, row[j] / vol[j]) for j in range(K)]
        newp = [0.0] * K
        for i in range(K):
            stay = p[i] * (1 - beta * cov[i])
            newp[i] += stay
            leave = p[i] * beta * cov[i]
            if leave > 0:
                Mi = M[i]
                for j in range(K):
                    if j != i and Mi[j] > 0:
                        newp[j] += leave * Mi[j]
        p = newp
        prev = list(row)
    return alloc


# ----------------------------- scoring driver ---------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = make_instances()

    vec = []
    for inst in instances:
        q_base = _simulate(inst, _trivial_ref(inst))
        forecast_score = _simulate(inst, _soft_prop_ref(inst, 0.0, 0.0))
        steady_score = _simulate(inst, _soft_prop_ref(inst, 0.8, 0.6))
        # q_top is genuinely driven by the two policy references (both always beat q_base on
        # this family); the tiny +1e-6 guard only protects against a degenerate zero-spread
        # instance and is never the binding term in practice.
        q_top = max(forecast_score, steady_score, q_base + 1e-6) * SLACK
        denom = q_top - q_base
        if denom < 1e-9:
            denom = 1e-9

        public = public_view(inst)
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            alloc = ans.get("alloc") if isinstance(ans, dict) else None
            obj = _simulate(inst, alloc)
        except Exception:
            obj = None
        if obj is None:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (obj - q_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
