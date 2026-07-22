#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_N_0819 -- "The Cyclone's Wake"
(family: latent-regime-prepositioning; format B, quality-metric).

THEME. A disaster-relief logistics desk manages D depots grouped into G
geographic clusters. Over T steps, a storm may be active over exactly one
cluster at a time (or the region may be calm). Each step, depots serve local
demand from on-hand stock; unmet demand is penalized. The desk may also ship
stock between depots, but every shipment takes L steps (the LEAD TIME) to
arrive.

TWO COMPOSED MECHANISMS drive the objective:

  (1) latent-markov-regime-shocks. A HIDDEN Markov chain over states
      {calm, storm_0, ..., storm_{G-1}} drives demand. From calm the storm
      can ignite at cluster 0. Once active at cluster g it either stays
      (prob p_stay), ADVANCES to the next cluster in a FIXED CYCLIC ORDER
      (g+1) % G (prob p_advance), or subsides to calm (remainder). This
      cyclic-advance structure is public (transition probabilities are
      given), but the SAMPLED regime path is hidden. All depots in the
      active cluster receive a large demand shock. The only public clue is
      a COARSE, per-CLUSTER, per-BIN sensor reading `cluster_activity[b][g]`
      -- the fraction of that cluster's depot-steps that read "active"
      during bin b (a fixed span of `bin_width` consecutive steps),
      themselves individually noisy (flip probability signal_flip_prob).
      Binning deliberately withholds exact within-bin timing: a bin can
      straddle multiple short dwells (or a dwell and a calm gap), so
      knowing a bin's aggregate activity does not hand you the exact strike
      step -- reading ahead to a future bin only tells you WHICH cluster is
      likely to matter around then, never WHEN within that window, so the
      cyclic transition model is still required to time a pre-positioning
      shipment.

  (2) transshipment-lead-time-delay. Stock shipped to depot j at step t is
      only usable at j starting step t+L. Shipments can originate from
      another depot (which cannot send more than it currently holds, and
      each depot-to-depot lane has a per-step capacity), OR from a single
      regional relief HUB (virtual location index D) that is never demand-
      constrained but costs more per unit and is throttled by its own
      per-step capacity -- so even "unlimited" supply must be committed
      ahead of the lead time, in the right quantity, to the right depot.

  THE TRAP. A policy that reacts to the sensor only AT THE STEP a depot
  currently looks stressed, and requests supply only once local stock is
  visibly short, sends a shipment that lands L steps later -- by which
  time, on any instance where the storm's typical dwell time is shorter
  than L, the storm has already cycled on to the NEXT cluster. The
  reactive shipment arrives at an already-calm depot while the real
  shortage has moved elsewhere, wasting both the lead time and the
  shipping cost. The insight is to track the noisy per-bin cluster
  readings over time to identify which cluster is *currently* active with
  confidence, exploit the KNOWN cyclic-advance structure to predict the
  NEXT cluster the storm will hit, and pre-stage stock toward that
  cluster's depots the moment confidence crosses a threshold -- not after
  the transition is observed.

CAUSALITY (why the candidate is invoked once PER BIN, not once for the
whole horizon). A single upfront JSON containing sensor readings for the
ENTIRE horizon would let a "candidate" simply read the reading for
whichever future bin a shipment is aimed at and ship accordingly --
solving the instance by lookup, not by inference, and defeating the
lead-time trap entirely (confirmed empirically: such a lookup policy beats
genuine regime-tracking outright). To make genuine forecasting-from-
partial-information the only viable strategy, the SAME candidate program
is invoked ONCE PER BIN, in bin order, each time in a FRESH isolated
subprocess that receives ONLY `known_activity` for bins `0..epoch`
(inclusive) -- bins after the current one are never in the payload,
because they have not been generated for that call. The candidate decides
shipments for ONLY that bin's steps `[epoch_start, epoch_end)`; the
evaluator accumulates the union of all bins' shipments into the final
plan before replaying it.

CANDIDATE CONTRACT (isolated stdin -> stdout program, invoked once per bin).
  stdin : ONE JSON object (the PUBLIC per-epoch view) -- see statement.md
          for the full schema. Static fields every call (D, T, L, G,
          clusters, base_demand, initial_stock, capacity, lane_cost,
          lane_capacity, hub_cost, hub_capacity, unmet_penalty, shock_mean,
          shock_std, signal_flip_prob, bin_width, transition) plus
          per-epoch fields: `epoch` (0-indexed bin number), `epoch_start`,
          `epoch_end` (this bin's step range `[epoch_start, epoch_end)`),
          and `known_activity` (a list of `epoch+1` rows, each `G`
          per-cluster activity fractions, covering bins `0..epoch` only).
          No seed or per-instance identifier is exposed, and no per-step/
          per-depot sensor reading is exposed -- only the coarse per-bin,
          per-cluster aggregate, and only for bins already "elapsed".
  stdout: ONE JSON object: {"shipments": [[t, i, j, qty], ...]}
          Each entry ships `qty` units from source i (a depot 0..D-1, or
          the hub, index D) to depot j (0..D-1, j != i when i < D), issued
          at step t with `epoch_start <= t < epoch_end` for THIS call
          (a shipment for any other step must be emitted by the call whose
          epoch owns that step). Arrivals land at step t+L. Malformed
          output at ANY bin (wrong shape/types, a t outside this bin's
          range, non-finite/negative quantity, a crash, a timeout, or
          non-JSON) makes the WHOLE instance score `0.0`; physically
          infeasible quantities (more than a depot currently holds, or
          exceeding a lane's/the hub's per-step capacity) are simply
          CLIPPED down to what's feasible during replay, not rejected.

SCORING (deterministic; no wall-time). The evaluator REPLAYS the submitted
shipment plan against the TRUE hidden demand array (never shown to the
candidate) to obtain:
    obj = unmet_penalty * (total unmet demand) + (total shipping cost)
Reference:
    b  = baseline(inst)   -- obj of the empty (do-nothing) plan
    LB = LB_FRAC * b      -- a generous, hard-to-reach lower reference
    r  = clamp(0.1 + 0.9 * (b - obj) / max(b - LB, 1e-9), 0, 1)
`LB_FRAC` is picked well below what even a strong, well-calibrated policy
can achieve (inherent shock noise + the sensor's own false-negative window
+ unavoidable base shipping cost mean obj never gets close to LB), so real
headroom remains above any reference solution.

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the public instance. The hidden
regime path, per-step noise, and true demand array live only in this
parent process.

CLI: python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / (1 << 53)

    return nxt


def _gauss(nx):
    u1 = max(nx(), 1e-12)
    u2 = nx()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ----------------------------- shared constants -----------------------------
D = 6
G = 3
CLUSTERS = [0, 0, 1, 1, 2, 2]     # depots 0,1 -> cluster0; 2,3 -> cluster1; 4,5 -> cluster2
BASE_NOISE_STD = 0.30
RESUPPLY_MULT = 1.15              # routine per-step resupply, a modest margin over base_demand
BIN_WIDTH = 5                      # public sensor readings are aggregated over bins this wide;
                                   # exact within-bin timing is never revealed, only the trap
                                   # dwell length (~1.5-2 steps) and the lead time L are both
                                   # well inside a single bin -- so an accurate GUESS at "which
                                   # cluster" from a future bin still leaves genuine "when within
                                   # this ~BIN_WIDTH-step window" uncertainty that only the known
                                   # cyclic transition model (not the sensor) can help resolve
HUB = D                            # virtual source index: the regional relief hub
HUB_UNIT_COST = 2.0                # per-unit cost from the hub -- pricier than depot-to-depot
LB_FRAC = 0.12                    # fraction of baseline treated as the (near-unreachable) floor
MAX_SHIP_ENTRIES = 20000
MAX_QTY = 1.0e7


def _lane_cost_matrix():
    m = [[0.0] * D for _ in range(D)]
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            m[i][j] = 1.0 if CLUSTERS[i] == CLUSTERS[j] else 2.2
    return m


# ----------------------------- instance construction -------------------------
def _build_instance(seed, T, L, p_stay, p_advance, shock_mean, shock_std,
                     lane_capacity, hub_capacity, unmet_penalty, signal_flip_prob,
                     sustain_frac, note):
    nx = _rng(seed)
    p_start = 0.35
    p_calm_return = max(0.0, 1.0 - p_stay - p_advance)

    base_demand = [round(3.0 + 4.0 * nx(), 3) for _ in range(D)]
    # Buffer stock is sized against SHOCK exposure, not cumulative background demand:
    # each depot also receives a routine per-step resupply equal to its own base_demand
    # (see _simulate), so an undisturbed depot is self-sustaining and initial_stock only
    # has to absorb storm shocks, not the whole horizon's background consumption.
    initial_stock = [round(shock_mean * (1.0 + 0.6 * nx()), 3) for _ in range(D)]
    capacity = [round(initial_stock[i] + shock_mean * 2.2 + 10.0, 3) for i in range(D)]
    lane_cost = _lane_cost_matrix()

    # --- sample the hidden regime path (calm=None, storm=g in 0..G-1) ---
    regime = []
    state = None  # None = calm
    for t in range(T):
        regime.append(state)
        r = nx()
        if state is None:
            state = 0 if r < p_start else None
        else:
            if r < p_stay:
                pass
            elif r < p_stay + p_advance:
                state = (state + 1) % G
            else:
                state = None

    # onset[t] marks the FIRST step a storm becomes active at its current cluster
    # (a fresh strike). Subsequent steps of the same dwell carry a smaller, sustained
    # elevated demand -- the damage is concentrated at the strike itself, which a
    # policy that only reacts once it observes a shortfall (arriving L steps later)
    # always misses, while a policy that predicts the strike from the cyclic
    # transition structure and pre-stages beforehand can actually catch it.
    onset = [False] * T
    prev_state = None
    for t in range(T):
        cur = regime[t]
        if cur is not None and cur != prev_state:
            onset[t] = True
        prev_state = cur

    # --- true hidden demand + noisy public signal ---
    demand_true = [[0.0] * D for _ in range(T)]
    signal = [[0] * D for _ in range(T)]
    for t in range(T):
        active = regime[t]
        for i in range(D):
            shock = 0.0
            base_ind = 1 if (active is not None and CLUSTERS[i] == active) else 0
            if base_ind:
                if onset[t]:
                    # damage is concentrated at the strike itself
                    shock = max(0.0, shock_mean + shock_std * _gauss(nx))
                else:
                    # a smaller sustained relief demand for as long as the dwell
                    # continues -- real (and, unlike the strike, reachable by a
                    # reactive shipment if the dwell outlasts the lead time), but
                    # secondary to the strike itself. How much secondary damage
                    # there is to catch reactively varies by instance.
                    shock = max(0.0, sustain_frac * shock_mean
                                + 0.4 * sustain_frac * shock_std * _gauss(nx))
            d = max(0.0, base_demand[i] + shock + BASE_NOISE_STD * _gauss(nx))
            demand_true[t][i] = round(d, 4)
            bit = base_ind
            if nx() < signal_flip_prob:
                bit = 1 - bit
            signal[t][i] = bit

    # --- aggregate the per-step sensor into COARSE per-cluster, per-bin readings.
    # This is the ONLY sensor information exposed publicly: exact within-bin timing
    # is never revealed (see BIN_WIDTH comment above), so a candidate cannot read
    # off "the exact future strike step" the way a raw per-step array would allow.
    num_bins = (T + BIN_WIDTH - 1) // BIN_WIDTH
    cluster_activity = [[0.0] * G for _ in range(num_bins)]
    for b in range(num_bins):
        lo, hi = b * BIN_WIDTH, min(T, (b + 1) * BIN_WIDTH)
        for g in range(G):
            depots_g = [i for i in range(D) if CLUSTERS[i] == g]
            vals = [signal[t][i] for t in range(lo, hi) for i in depots_g]
            cluster_activity[b][g] = round(sum(vals) / len(vals), 4) if vals else 0.0

    return dict(
        seed=seed, D=D, T=T, L=L, G=G, clusters=list(CLUSTERS),
        base_demand=base_demand, initial_stock=initial_stock, capacity=capacity,
        lane_cost=lane_cost, lane_capacity=lane_capacity,
        hub_cost=HUB_UNIT_COST, hub_capacity=hub_capacity, unmet_penalty=unmet_penalty,
        shock_mean=shock_mean, shock_std=shock_std, signal_flip_prob=signal_flip_prob,
        bin_width=BIN_WIDTH, cluster_activity=cluster_activity,
        transition=dict(p_stay=p_stay, p_advance=p_advance,
                         p_calm_return=p_calm_return, p_start=p_start),
        demand_true=demand_true, note=note,
    )


def _build_instances():
    specs = [
        # seed,  T,  L, p_stay, p_adv, shock_mean, shock_std, lane_cap, hub_cap, penalty, flip, sustain, note
        (4001, 40, 3, 0.85, 0.15, 16.0, 3.0, 22.0, 26.0, 7.0, 0.10, 0.35, "easy-slow-dwell"),
        (4002, 40, 3, 0.35, 0.65, 18.0, 3.0, 22.0, 26.0, 7.0, 0.10, 0.04, "trap-fast-cycle"),
        (4003, 40, 3, 0.40, 0.60, 24.0, 4.0, 20.0, 24.0, 7.0, 0.12, 0.04, "trap-fast-strong-shock"),
        (4004, 40, 3, 0.60, 0.30, 17.0, 3.0, 24.0, 28.0, 7.0, 0.10, 0.20, "medium"),
        (4005, 40, 3, 0.30, 0.70, 16.0, 3.0, 14.0, 17.0, 7.0, 0.10, 0.04, "trap-fast-tightcap"),
        (4006, 40, 2, 0.80, 0.20, 15.0, 3.0, 24.0, 28.0, 7.0, 0.08, 0.35, "easy-slow-dwell2"),
        (4007, 40, 3, 0.35, 0.65, 20.0, 4.0, 20.0, 24.0, 8.0, 0.12, 0.04, "trap-fast-highpenalty"),
        (4008, 40, 3, 0.50, 0.35, 18.0, 3.0, 22.0, 26.0, 7.0, 0.10, 0.15, "medium2"),
        (4009, 42, 3, 0.82, 0.18, 16.0, 3.0, 22.0, 26.0, 7.0, 0.08, 0.35, "easy-slow-dwell3"),
        (4010, 45, 3, 0.32, 0.68, 19.0, 3.5, 18.0, 22.0, 7.0, 0.11, 0.04, "trap-heldout"),
    ]
    out = []
    for (seed, T, L, p_stay, p_advance, shock_mean, shock_std, lane_capacity,
         hub_capacity, unmet_penalty, flip, sustain_frac, note) in specs:
        out.append(_build_instance(seed, T, L, p_stay, p_advance, shock_mean, shock_std,
                                    lane_capacity, hub_capacity, unmet_penalty, flip,
                                    sustain_frac, note))
    return out


# ----------------------------- simulation / scoring -----------------------------
def _simulate(inst, shipments):
    """Replay a (validated, parsed) shipment list against the TRUE hidden demand.
    shipments: list of (t,i,j,qty) with t,i,j already range-checked ints and qty a
    finite non-negative float. Returns total objective (unmet_penalty*unmet + cost)."""
    Dn, T, L = inst["D"], inst["T"], inst["L"]
    stock = list(inst["initial_stock"])
    capacity = inst["capacity"]
    lane_cost = inst["lane_cost"]
    lane_capacity = inst["lane_capacity"]
    hub_cost = inst["hub_cost"]
    hub_capacity = inst["hub_capacity"]
    unmet_penalty = inst["unmet_penalty"]
    demand_true = inst["demand_true"]

    by_t = [[] for _ in range(T)]
    for (t, i, j, qty) in shipments:
        by_t[t].append((i, j, qty))

    base_demand = inst["base_demand"]
    arrivals = {}
    total_unmet = 0.0
    total_cost = 0.0
    for t in range(T):
        # routine per-step resupply -- a modest margin ABOVE background demand, so an
        # undisturbed depot's buffer can slowly build genuine slack over calm stretches,
        # capped at storage capacity. Only demand above the background level (storm
        # shocks) meaningfully draws the buffer down.
        for i in range(Dn):
            stock[i] = min(capacity[i], stock[i] + RESUPPLY_MULT * base_demand[i])
        for (j, qty) in arrivals.pop(t, []):
            stock[j] = min(capacity[j], stock[j] + qty)
        hub_used_this_step = 0.0
        lane_used_this_step = {}   # (i,j) -> cumulative qty already shipped this step
        for (i, j, qty) in by_t[t]:
            if i == HUB:
                room = max(0.0, hub_capacity - hub_used_this_step)
                shipped = min(qty, room)
                if shipped <= 0:
                    continue
                hub_used_this_step += shipped
                total_cost += shipped * hub_cost
            else:
                lane_room = max(0.0, lane_capacity - lane_used_this_step.get((i, j), 0.0))
                avail = stock[i]
                shipped = min(qty, avail, lane_room)
                if shipped <= 0:
                    continue
                stock[i] -= shipped
                lane_used_this_step[(i, j)] = lane_used_this_step.get((i, j), 0.0) + shipped
                total_cost += shipped * lane_cost[i][j]
            arrivals.setdefault(t + L, []).append((j, shipped))
        for i in range(Dn):
            d = demand_true[t][i]
            served = min(stock[i], d)
            stock[i] -= served
            total_unmet += (d - served)

    return unmet_penalty * total_unmet + total_cost


def baseline(inst):
    """Do-nothing plan: no shipments at all."""
    return _simulate(inst, [])


def _parse_epoch_answer(inst, answer, epoch_start, epoch_end):
    """Strictly validate one epoch's `answer`; every shipment's t must fall
    within [epoch_start, epoch_end) -- the steps THIS epoch call owns.
    Returns (ok, parsed_shipments)."""
    if not isinstance(answer, dict):
        return False, None
    ships = answer.get("shipments")
    if not isinstance(ships, list):
        return False, None
    if len(ships) > MAX_SHIP_ENTRIES:
        return False, None
    Dn = inst["D"]
    parsed = []
    for entry in ships:
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            return False, None
        t, i, j, qty = entry
        if isinstance(t, bool) or isinstance(i, bool) or isinstance(j, bool):
            return False, None
        if not isinstance(t, int) or not isinstance(i, int) or not isinstance(j, int):
            return False, None
        if not (epoch_start <= t < epoch_end) or not (0 <= i <= Dn) or not (0 <= j < Dn) or i == j:
            return False, None
        if isinstance(qty, bool) or not isinstance(qty, (int, float)):
            return False, None
        qty = float(qty)
        if not (qty == qty) or qty in (float("inf"), float("-inf")):
            return False, None
        if qty < 0.0 or qty > MAX_QTY:
            return False, None
        parsed.append((t, i, j, qty))
    return True, parsed


def score(inst, shipments):
    """Replay an already-validated, already-assembled shipment list against
    the TRUE hidden demand. Returns (ok: bool, obj: float)."""
    obj = _simulate(inst, shipments)
    if not (obj == obj) or obj in (float("inf"), float("-inf")) or obj < 0.0:
        return False, 0.0
    return True, obj


STATIC_FIELDS = (
    "D", "T", "L", "G", "clusters", "base_demand", "initial_stock",
    "capacity", "lane_cost", "lane_capacity", "hub_cost", "hub_capacity",
    "unmet_penalty", "shock_mean", "shock_std", "signal_flip_prob", "bin_width",
    "transition")
EPOCH_TIMEOUT = 8.0


def _run_instance(cand, inst):
    """Invoke `cand` once per bin (causal order), accumulate its validated
    shipments, then replay the full plan. Returns (ok, obj)."""
    T, bin_width = inst["T"], inst["bin_width"]
    activity = inst["cluster_activity"]
    num_bins = len(activity)
    static = {k: inst[k] for k in STATIC_FIELDS}

    all_shipments = []
    for epoch in range(num_bins):
        epoch_start = epoch * bin_width
        epoch_end = min(T, (epoch + 1) * bin_width)
        payload = dict(static)
        payload["epoch"] = epoch
        payload["epoch_start"] = epoch_start
        payload["epoch_end"] = epoch_end
        payload["known_activity"] = [list(row) for row in activity[:epoch + 1]]

        ans, st = isorun.run_candidate(cand, payload, timeout=EPOCH_TIMEOUT)
        if st != "OK":
            return False, 0.0
        try:
            ok, ships = _parse_epoch_answer(inst, ans, epoch_start, epoch_end)
        except Exception:
            ok, ships = False, None
        if not ok:
            return False, 0.0
        all_shipments.extend(ships)

    try:
        return score(inst, all_shipments)
    except Exception:
        return False, 0.0


# ----------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        ok, obj = _run_instance(cand, inst)
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        lb = LB_FRAC * b
        denom = max(b - lb, 1e-9)
        r = 0.1 + 0.9 * (b - obj) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
