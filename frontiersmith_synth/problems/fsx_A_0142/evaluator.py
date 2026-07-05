#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0142 -- "Tell Qadesh Dig: Crating the Catalogued Finds"
(family: online-heuristic-simulator; format B, quality-metric).

THEME.  A season at the Tell Qadesh excavation has finished.  Every cell of the dig
grid has been swept and each recovered artifact catalogued with an integer *mass*.
The finds must now be sealed into identical archival crates for the museum convoy.
Each crate has two hard limits:

    * a MASS capacity C  -- the total mass of the artifacts inside may not exceed C
      (the crate would buckle), and
    * a SLOT limit  K   -- at most K artifacts fit, because every find gets its own
      padded, humidity-sealed slot regardless of how small it is.

Sealing and hauling a crate costs one crate.  The registrar wants to ship the whole
catalogue using as FEW crates as possible.

This is 1-D bin packing (FunSearch's bin-packing family) generalized with a
CARDINALITY (k-item) constraint: artifacts = items with integer sizes, crate mass
capacity = bin capacity, crate slot limit = a per-bin item-count cap, "crates
shipped" = bins used, which we MINIMIZE.  A crate is now constrained on BOTH mass
and slot count, so a rule that only watches mass wastes crates whenever slots bind
(and vice-versa) -- there is no single greedy that wins everywhere.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "capacity": C (int), "slots": K (int), "n": N (int),
             "masses": [m_0, ..., m_{N-1}]}   # integer masses, 1 <= m_i <= C
  stdout: ONE JSON object:
            {"assign": [c_0, ..., c_{N-1}]}
          where c_i >= 0 is the crate index artifact i is sealed into.  Crate
          indices need not be contiguous; a crate "exists" iff >=1 artifact is
          assigned to it, and the number of DISTINCT non-empty crates is the cost.

  A crating is VALID iff `assign` is a list of exactly N non-negative integers and,
  for every crate, the summed mass does not exceed C AND the artifact count does not
  exceed K.  Invalid output, wrong length, an overfilled/over-slotted crate, a
  crash, a timeout, or non-JSON  ->  that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute three references:
    q_lb   = max( ceil(sum(masses)/C), ceil(N/K) )   # L1 / cardinality lower bound
    q_base = crates used by the internal FIRST-FIT registrar in catalogue order
             (mass- AND slot-aware)                    # the weak reference
    q_cand = distinct crates used by the candidate crating
  and normalize with an affine anchor (weak first-fit -> 0.1, lower bound -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  A candidate reproducing first-fit scores ~0.1; one reaching the (generally
  unreachable) lower bound scores 1.0; doing worse than first-fit scores < 0.1.

  Because the lower bound is LOOSE (mass and slot bounds are computed independently
  and rarely coincide), even strong packers -- first-fit-decreasing, best-fit-
  decreasing, plus local search -- stay strictly below 1.0 on most instances, so
  there is genuine headroom and no easy optimum.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(lower bound, first-fit baseline) are computed by THIS parent process, so a
frame-walking / introspecting candidate learns nothing useful and cannot forge a
score.

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

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family -----------------------------
def _build_masses(seed, n, C, dist):
    """Return a list of N integer artifact masses in [1, C]. Deterministic."""
    ni = _rng(seed)
    out = []
    for _ in range(n):
        if dist == "uni":                       # full spread
            m = ni(1, C)
        elif dist == "small":                   # many light sherds -> slots tend to bind
            m = ni(1, max(1, C // 4))
        elif dist == "half":                    # masses near C/2 -> hard to pair, mass binds
            m = ni(max(1, (2 * C) // 5), (3 * C) // 5)
        elif dist == "bimodal":                 # light sherds + heavy statuary
            m = ni(1, max(1, C // 5)) if ni(0, 99) < 60 else ni((3 * C) // 5, (9 * C) // 10)
        else:
            m = ni(1, C)
        if m < 1:
            m = 1
        if m > C:
            m = C
        out.append(m)
    return out


def _build_instances():
    """Deterministic instance family: (seed, n, C, K, dist).

    K is the per-crate slot limit.  Small K + light masses makes slots bind;
    large K + heavy masses makes mass bind; mixed instances reward rules that
    watch both.  The last block is harder / larger held-out data.
    """
    specs = [
        (101, 30, 24,  6, "uni"),
        (102, 30, 24,  4, "small"),
        (103, 28, 20,  8, "half"),
        (104, 32, 24,  5, "bimodal"),
        (105, 30, 20,  6, "uni"),
        (106, 28, 24,  3, "small"),
        (107, 30, 24,  7, "half"),
        (108, 32, 20,  5, "bimodal"),
        # harder / larger held-out instances
        (211, 46, 24,  6, "uni"),
        (212, 48, 24,  4, "small"),
        (213, 44, 20,  9, "half"),
        (214, 50, 24,  5, "bimodal"),
    ]
    out = []
    for seed, n, C, K, dist in specs:
        masses = _build_masses(seed, n, C, dist)
        out.append({"name": f"qadesh{seed}", "capacity": C, "slots": K,
                    "n": n, "masses": masses, "dist": dist})
    return out


# ----------------------------- references ----------------------------------
def _lower_bound(masses, C, K):
    n = len(masses)
    by_mass = -(-sum(masses) // C)      # ceil(sum / C)
    by_slot = -(-n // K)                # ceil(n / K)
    return max(by_mass, by_slot, 1)


def _first_fit(masses, C, K):
    """Weak reference registrar: in catalogue order, seal each artifact into the
    lowest-index existing crate that still has room in BOTH mass and slots; open a
    new crate only if none fits.  Never reorders, never looks back."""
    loads = []   # (mass_used, count)
    for m in masses:
        placed = False
        for j in range(len(loads)):
            mu, cnt = loads[j]
            if mu + m <= C and cnt + 1 <= K:
                loads[j] = (mu + m, cnt + 1)
                placed = True
                break
        if not placed:
            loads.append((m, 1))
    return len(loads)


# ----------------------------- validation ----------------------------------
def _crates_used(inst, answer):
    """Validate answer against the instance. Return crate count or None."""
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    masses = inst["masses"]
    C = inst["capacity"]
    K = inst["slots"]
    N = inst["n"]
    if len(assign) != N:
        return None
    mass_load = {}
    slot_load = {}
    for i, c in enumerate(assign):
        if isinstance(c, bool) or not isinstance(c, int):
            return None
        if c < 0:
            return None
        mass_load[c] = mass_load.get(c, 0) + masses[i]
        slot_load[c] = slot_load.get(c, 0) + 1
        if mass_load[c] > C or slot_load[c] > K:
            return None
    return len(mass_load)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        C = inst["capacity"]
        K = inst["slots"]
        masses = inst["masses"]
        q_lb = _lower_bound(masses, C, K)
        q_base = _first_fit(masses, C, K)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "capacity": C, "slots": K,
                  "n": inst["n"], "masses": list(masses)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _crates_used(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_base - q_cand) / denom
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
