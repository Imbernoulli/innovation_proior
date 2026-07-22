#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0528 -- "Blackstart Triage: Proactive Load Shedding under Rolling Line Trips"
(family: replay-cascade-triage; format B, quality-metric; theme: power-grid load shedding).

THEME.  A transmission grid is a ring of L corridors ("lines"), each carrying a real
power load.  Line l fails the instant its load exceeds its capacity cap[l].  An
exogenous SCHEDULE of rolling line trips (known in advance) knocks lines out one at a
time.  When a line goes dark its load does not vanish -- it is REDISTRIBUTED equally
onto its still-live ring neighbors ("rerouting").  If that pushes a neighbor over its
own cap, THAT line trips too, and the failure CASCADES.  Power that a dead line was
carrying and that no live neighbor can absorb is lost as unmet demand.

The operator's only lever is PROACTIVE SHEDDING: at each step it may curtail load
from any lines (removing it from the grid).  Curtailed power is unmet demand too --
but shedding a little early creates the margin a line needs to ABSORB an incoming
redistribution wave instead of tripping and dumping its whole load downstream.

BACKLOG CARRYOVER (mechanism 1).  Unmet demand -- whether curtailed by the operator or
lost in a cascade -- accumulates into a monotone backlog B_t that carries over.  Every
remaining step pays a holding cost on the whole outstanding backlog, and the cost is
CONVEX in B_t.  So an early cascade loss is paid many times over and hurts far more
than the same units shed proactively and never re-incurred.

CASCADING REDISTRIBUTION (mechanism 2).  The transition above (equal-split reroute +
overload-triggered secondary trips propagated to a fixpoint) is what turns one
exogenous trip into a multi-line collapse.

INNOVATION HOOK (what `strong` exploits).  The trip schedule is public, so the cascade
frontier is PREDICTABLE.  Shedding proactively along the *predicted* frontier -- just
enough on each line that will receive a redistribution wave to keep the incoming load
below that line's margin -- stops the cascade at its source.  A myopic policy that only
relieves the currently-most-stressed line reroutes effort onto the wrong corridor and
still eats the cascade two steps later.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "L": int, "T": int,
             "cap":   [c_0 ... c_{L-1}],          # line capacities (>0)
             "load0": [x_0 ... x_{L-1}],          # initial loads (0 <= x_l <= cap_l)
             "nbr":   [[...] ...],                # nbr[l] = ring/chord neighbors of line l
             "schedule": [e_0 ... e_{T-1}],       # exogenous line tripped at each step (-1 = none)
             "alpha": float, "beta": float, "scale": float}   # backlog cost coefficients
  stdout: ONE JSON object:
            {"shed": [[s_{t,0} ... s_{t,L-1}] for t in 0..T-1]}
          s_{t,l} >= 0 units curtailed from line l at the START of step t (before the
          scheduled trip).  A shed larger than the line's current load is clamped.

  VALID iff `shed` is a list of exactly T lists, each of exactly L finite numbers >= 0
  (no NaN/inf/bool/negative).  Any violation, crash, timeout, or non-JSON -> 0.0 on
  that instance.

TRANSITION per step t (deterministic):
  1. curtail: for each live line l, remove min(s_{t,l}, load[l]); add it to this step's
     unmet demand u.
  2. exogenous trip: if e_t is a live line, enqueue it.
  3. cascade: while queue non-empty, pop x, mark it dead, take its load L_x; split L_x
     EQUALLY among x's currently-live neighbors; each such neighbor's load rises and,
     if it now exceeds its cap, it is enqueued (it will trip carrying its FULL load).
     If x has no live neighbor, L_x is lost -> add to u.
  4. backlog: B_t = B_{t-1} + u;  penalty += alpha*B_t + beta*B_t*B_t/scale.
  Total episode cost = the accumulated penalty (LOWER is better).

SCORING (deterministic; no wall-time).  Per instance:
    P_cand = episode cost of the candidate plan.
    P_base = episode cost of the DO-NOTHING plan (shed nothing) -- the weak reference.
    r = clamp( 0.1 + 0.9 * (1 - P_cand / P_base), 0, 1 )
  Do-nothing scores exactly 0.1; a (generally unreachable) zero-backlog plan scores
  1.0; a plan worse than doing nothing scores below 0.1.  Because every scheduled trip
  is engineered so its live neighbors' total margin is smaller than its load, some
  unmet demand is UNAVOIDABLE -- so even an optimal plan keeps P_cand > 0 and r < 1.
  The final score is the mean of r over 10 fixed seeded instances.

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  The do-nothing
reference is computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
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


# ----------------------------- transition / cost ---------------------------
def simulate(inst, shed):
    """Replay the deterministic cascade with a shedding plan. Return episode cost."""
    L, T = inst["L"], inst["T"]
    cap = inst["cap"]
    nbr = inst["nbr"]
    schedule = inst["schedule"]
    alpha, beta, scale = inst["alpha"], inst["beta"], inst["scale"]

    budget = inst["budget"]
    load = [float(x) for x in inst["load0"]]
    alive = [True] * L
    backlog = 0.0
    penalty = 0.0

    for t in range(T):
        u = 0.0
        row = shed[t]
        # 1. proactive curtailment, capped by the per-step shed budget.  If the plan
        #    requests more than `budget` this step, all its sheds are scaled down
        #    proportionally (deterministic, order-independent) to meet the cap.
        raw = 0.0
        for l in range(L):
            if alive[l] and row[l] > 0.0:
                raw += row[l]
        f = 1.0 if raw <= budget else budget / raw
        for l in range(L):
            if not alive[l]:
                continue
            s = row[l] * f
            if s > load[l]:
                s = load[l]
            if s > 0.0:
                load[l] -= s
                u += s
        # 2. exogenous trip
        queue = []
        e = schedule[t]
        if 0 <= e < L and alive[e]:
            queue.append(e)
        # 3. cascade redistribution to a fixpoint
        qi = 0
        while qi < len(queue):
            x = queue[qi]
            qi += 1
            if not alive[x]:
                continue
            alive[x] = False
            lx = load[x]
            load[x] = 0.0
            live = [y for y in nbr[x] if alive[y]]
            if not live:
                u += lx
            else:
                share = lx / len(live)
                for y in live:
                    load[y] += share
                    if load[y] > cap[y]:
                        queue.append(y)
        # 4. backlog carryover + convex penalty
        backlog += u
        penalty += alpha * backlog + beta * backlog * backlog / scale
    return penalty


# ----------------------------- instance family -----------------------------
def _ring_nbr(L, chords=None):
    nb = [[(l - 1) % L, (l + 1) % L] for l in range(L)]
    if chords:
        for a, b in chords:
            if b not in nb[a]:
                nb[a].append(b)
            if a not in nb[b]:
                nb[b].append(a)
    return nb


def _corridor(seed, L, T, corr_len, alpha, beta, scale, budget):
    """Trap topology: a fragile low-cap corridor + high-utilisation decoys elsewhere.
    The schedule trips the MIDDLE of the corridor (its neighbours' margin < its load),
    while the decoys are the most-stressed lines -- so a hotspot-relief policy chases
    the decoys and eats the corridor cascade."""
    ni = _rng(seed)
    cap = [120] * L
    load = [ni(40, 60) for _ in range(L)]
    # fragile corridor
    cs = ni(1, L - corr_len - 1)
    for k in range(corr_len):
        cap[cs + k] = 55
        load[cs + k] = 48                      # margin 7
    mid = cs + corr_len // 2
    # decoys: high utilisation, far from corridor, but with roomy neighbours
    d1 = (cs + L // 2) % L
    d2 = (cs + L // 2 + 2) % L
    for d in (d1, d2):
        if cs <= d < cs + corr_len:
            continue
        cap[d] = 120
        load[d] = 112                          # util ~0.93 -> greedy magnet
    schedule = [-1] * T
    schedule[2] = mid                          # first wave into the corridor
    if T >= 7:
        schedule[5] = cs                       # second wave at the corridor edge
    return {"L": L, "T": T, "cap": cap, "load0": load, "nbr": _ring_nbr(L),
            "schedule": schedule, "alpha": alpha, "beta": beta, "scale": scale, "budget": budget}


def _diffuse(seed, L, T, alpha, beta, scale, budget):
    """Greedy-friendly topology: the scheduled trip hits the current hotspot itself,
    and its neighbours carry enough spare margin that relieving the hotspot BEFORE it
    trips (the myopic recipe) genuinely stops the secondary cascade -- so the obvious
    recipe pays off here, unlike on the fragile-corridor traps."""
    ni = _rng(seed)
    cap = [ni(95, 125) for _ in range(L)]
    load = [int(cap[l] * (ni(35, 50) / 100.0)) for l in range(L)]
    # one clear hotspot; neighbours roomy (~0.55) so a pre-shed of the hotspot averts
    # the cascade, but their combined free margin is < the hotspot load (forced loss > 0).
    h = ni(1, L - 2)
    load[h] = int(cap[h] * 0.95)
    for y in ((h - 1) % L, (h + 1) % L):
        load[y] = int(cap[y] * 0.55)
    schedule = [-1] * T
    schedule[2] = h
    if T >= 6:
        h2 = (h + 4) % L
        load[h2] = int(cap[h2] * 0.93)
        for y in ((h2 - 1) % L, (h2 + 1) % L):
            if y != h:
                load[y] = int(cap[y] * 0.55)
        schedule[4] = h2
    return {"L": L, "T": T, "cap": cap, "load0": load, "nbr": _ring_nbr(L),
            "schedule": schedule, "alpha": alpha, "beta": beta, "scale": scale, "budget": budget}


def _twin(seed, L, T, alpha, beta, scale, budget):
    """Held-out: two fragile corridors hit in successive waves, plus a chord that lets
    a wave jump between them.  Even good frontier shedding cannot save everything."""
    ni = _rng(seed)
    cap = [120] * L
    load = [ni(40, 58) for _ in range(L)]
    a = ni(1, L // 2 - 4)
    b = ni(L // 2 + 1, L - 5)
    for base in (a, b):
        for k in range(4):
            cap[base + k] = 52
            load[base + k] = 46                # margin 6
    # a decoy to distract hotspot-relief
    d = (a + b) // 2
    cap[d] = 120
    load[d] = 113
    chords = [(a + 1, b + 1)]                   # cross-corridor bridge
    schedule = [-1] * T
    schedule[1] = a + 2
    schedule[4] = b + 2
    if T >= 8:
        schedule[6] = a + 1
    return {"L": L, "T": T, "cap": cap, "load0": load, "nbr": _ring_nbr(L, chords),
            "schedule": schedule, "alpha": alpha, "beta": beta, "scale": scale, "budget": budget}


def _build_instances():
    out = []
    # (name, kind, seed, L, T, corr_len, alpha, beta, scale, budget)
    specs = [
        ("corr1", "corr", 5301, 16, 8, 5, 0.5, 1.0, 40.0, 18.0),
        ("corr2", "corr", 5302, 18, 8, 5, 0.5, 1.0, 45.0, 16.0),
        ("corr3", "corr", 5303, 16, 9, 4, 0.4, 1.0, 35.0, 16.0),
        ("diff1", "diff", 5311, 16, 8, 0, 0.5, 1.0, 40.0, 16.0),
        ("diff2", "diff", 5312, 18, 8, 0, 0.5, 1.0, 45.0, 16.0),
        ("diff3", "diff", 5313, 15, 7, 0, 0.5, 1.0, 40.0, 15.0),
        ("corr4", "corr", 5304, 20, 9, 6, 0.5, 1.0, 50.0, 18.0),
        ("twin1", "twin", 5321, 22, 8, 0, 0.5, 1.0, 45.0, 16.0),
        ("twin2", "twin", 5322, 24, 9, 0, 0.4, 1.0, 50.0, 16.0),
        ("diff4", "diff", 5314, 20, 9, 0, 0.5, 1.0, 45.0, 16.0),
    ]
    for name, kind, seed, L, T, corr_len, a, b, sc, bud in specs:
        if kind == "corr":
            inst = _corridor(seed, L, T, corr_len, a, b, sc, bud)
        elif kind == "diff":
            inst = _diffuse(seed, L, T, a, b, sc, bud)
        else:
            inst = _twin(seed, L, T, a, b, sc, bud)
        inst["name"] = name
        out.append(inst)
    return out


# ----------------------------- validation ----------------------------------
def _valid_plan(inst, answer):
    """Return a T x L shed matrix of non-negative floats, or None if malformed."""
    if not isinstance(answer, dict):
        return None
    shed = answer.get("shed")
    if not isinstance(shed, list) or len(shed) != inst["T"]:
        return None
    L = inst["L"]
    out = []
    for row in shed:
        if not isinstance(row, list) or len(row) != L:
            return None
        r = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            v = float(v)
            if v != v or v in (float("inf"), float("-inf")) or v < 0.0:
                return None
            r.append(v)
        out.append(r)
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        zero = [[0.0] * inst["L"] for _ in range(inst["T"])]
        p_base = simulate(inst, zero)
        if p_base <= 1e-9:
            p_base = 1e-9
        public = {"name": inst["name"], "L": inst["L"], "T": inst["T"],
                  "cap": list(inst["cap"]), "load0": list(inst["load0"]),
                  "nbr": [list(x) for x in inst["nbr"]],
                  "schedule": list(inst["schedule"]),
                  "alpha": inst["alpha"], "beta": inst["beta"], "scale": inst["scale"],
                  "budget": inst["budget"]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            plan = _valid_plan(inst, ans)
            if plan is None:
                vec.append(0.0)
                continue
            p_cand = simulate(inst, plan)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (1.0 - p_cand / p_base)
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
