#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0712 -- "Seed-Swap Network: Heirloom Steward"
(family: barter-cycle-steward; format B, quality-metric).

THEME.  A community seed-swap network trades heirloom plant varieties (types
0..T-1).  Type 0 ("Moon-and-Stars") is a coveted heirloom almost everyone
wants but almost nobody currently holds.  Gardeners arrive over `n_rounds`
rounds, each holding one packet of `have` and wanting one packet of `want`
(have != want), and linger in the pool for `patience` consecutive rounds
before giving up.  A policy may, each round, execute disjoint SWAP CYCLES of
length 2 or 3 among gardeners currently present: a cycle [a0,...,a(k-1)]
means a_i receives a_{(i+1) mod k}'s packet, requiring want[a_i] ==
have[a_{(i+1) mod k}] for every i.  Executing a cycle satisfies every
gardener in it; a packet is used at most once.

MECHANISMS COMPOSED.
  - exchange-cycle-selection: the policy chooses which disjoint length<=3
    cycles to fire, and WHEN (which round).
  - pool-composition-externality: unmatched gardeners persist across rounds
    (bounded by patience) -- what you leave in the pool this round shapes
    which cycles are even possible in future rounds, and what expires.
  - scarce-type-banking: type 0 is supply-scarce (rare `have`, heavy
    `want`).  A holder of type 0 can only ever close ONE cycle, so WHICH
    cycle it closes (and when) matters a great deal.

INNOVATION HOOK.  A cycle's value is not just "N gardeners satisfied this
round" -- it also determines which gardeners remain in the pool (and which
expire).  Because type-flow is conserved along a cycle (every id appears
exactly once as a giver and once as a receiver), a policy that tracks GLOBAL
have/want counts per type can identify exactly which types are supply-scarce
and refuse to spend a scarce-`have` holder on the first cycle that happens to
be available; it holds ("banks") them for a larger/better cycle, which
strictly dominates immediately maximizing cycles found each round.

TRAP.  A myopic policy that greedily executes every currently-available cycle
the instant it appears (starting with the easy length-2 cycles) will burn a
rare Moon-and-Stars holder on the very first reciprocal partner it meets --
even when a length-3 cycle, one round away, would have used that SAME
holder to rescue two otherwise-doomed gardeners instead of one, with no way
to undo the choice afterward.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON public instance (see statement.md):
    {"name","n_types","type_names","n_rounds","arrivals": [round0 list, ...]}
    each arrival: {"id","have","want","round","patience"}.
  stdout: ONE JSON object {"rounds": [ {"cycles": [[..],...]}, ... ]}
          exactly n_rounds entries, in order.  A cycle [a0..a(k-1)], k in
          {2,3}, is valid at round r iff all ids are distinct, known, not yet
          used, currently present (round[a_i] <= r <= round[a_i]+patience[a_i]-1),
          and want[a_i] == have[a_{(i+1) mod k}] for every i.

  Malformed TOP-LEVEL shape (wrong length, non-list cycles/rounds, bad types)
  scores the whole instance 0.0.  An individual bad cycle (unknown/reused/
  expired id or a broken chain) is simply skipped, no further penalty.

SCORING (deterministic; no wall-time).  Per instance:
    y_base = gardeners satisfied by a WEAK reference: immediate, length-2-
             cycles-ONLY, first-found, no hoarding, computed by the
             evaluator itself.
    y_ub   = total number of gardeners who ever arrive (loose upper bound).
    y_cand = gardeners satisfied by the candidate's schedule.
    r = clamp( 0.1 + 0.9 * (y_cand - y_base) / max(1e-9, y_ub - y_base), 0, 1)

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the public instance.  All
validation and reference computation happen in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

KEYSTONE = 0
TYPE_NAMES = [
    "Moon-and-Stars Melon", "Cherokee Purple Tomato", "Dragon Tongue Bean",
    "Glass Gem Corn", "Amish Paste Tomato", "Painted Mountain Corn",
]
T = len(TYPE_NAMES)


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance construction ------------------------
def _make_cluster(next_id, r0, t1, t2, patience=3):
    """A planted scarcity trap: a Moon-and-Stars donor + a decoy direct
    partner + a connector + a closer, arriving one round apart."""
    ids = list(range(next_id, next_id + 4))
    agents = [
        {"id": ids[0], "have": t1, "want": KEYSTONE, "round": r0, "patience": patience},
        {"id": ids[1], "have": KEYSTONE, "want": t1, "round": r0, "patience": patience},
        {"id": ids[2], "have": t1, "want": t2, "round": r0 + 1, "patience": patience},
        {"id": ids[3], "have": t2, "want": KEYSTONE, "round": r0 + 1, "patience": patience},
    ]
    return agents, next_id + 4


def _make_filler_pair(next_id, r, t1, t2):
    """A mutual-reciprocal orbit-type pair: trivially closable at round r by
    ANY policy (never touches the keystone type), pure background noise."""
    ids = [next_id, next_id + 1]
    agents = [
        {"id": ids[0], "have": t1, "want": t2, "round": r, "patience": 2},
        {"id": ids[1], "have": t2, "want": t1, "round": r, "patience": 2},
    ]
    return agents, next_id + 2


def _make_filler_triple(next_id, r, t1, t2, t3):
    """A SAME-round orbit-type 3-cycle t1->t2->t3->t1 (never touches the
    keystone type): any 3-cycle-aware policy (greedy or strong) closes it
    immediately, but the length-2-only weak baseline can never see it --
    this is what makes even the textbook greedy clear the weak baseline."""
    ids = [next_id, next_id + 1, next_id + 2]
    agents = [
        {"id": ids[0], "have": t1, "want": t2, "round": r, "patience": 2},
        {"id": ids[1], "have": t2, "want": t3, "round": r, "patience": 2},
        {"id": ids[2], "have": t3, "want": t1, "round": r, "patience": 2},
    ]
    return agents, next_id + 3


def _trap_instance(seed, name, n_clusters, spacing, filler_n, r_extra=6, filler_triple_n=4):
    ni = _rng(seed)
    next_id = 0
    R = n_clusters * spacing + r_extra
    agents = []
    orbit = list(range(1, T))
    pairs = []
    for i in range(n_clusters):
        t1 = orbit[i % len(orbit)]
        t2 = orbit[(i + 1 + ni(0, 1)) % len(orbit)]
        if t2 == t1:
            t2 = orbit[(i + 2) % len(orbit)]
        pairs.append((t1, t2))
    for i in range(n_clusters):
        r0 = i * spacing
        t1, t2 = pairs[i]
        cl, next_id = _make_cluster(next_id, r0, t1, t2)
        agents.extend(cl)
    for _ in range(filler_triple_n):
        r = ni(0, R - 1)
        t1 = orbit[ni(0, len(orbit) - 1)]
        t2 = orbit[ni(0, len(orbit) - 1)]
        t3 = orbit[ni(0, len(orbit) - 1)]
        if len({t1, t2, t3}) < 3:
            continue
        ft, next_id = _make_filler_triple(next_id, r, t1, t2, t3)
        agents.extend(ft)
    for _ in range(filler_n):
        r = ni(0, R - 1)
        t1 = orbit[ni(0, len(orbit) - 1)]
        t2 = orbit[ni(0, len(orbit) - 1)]
        if t1 == t2:
            t2 = orbit[(orbit.index(t1) + 1) % len(orbit)]
        fp, next_id = _make_filler_pair(next_id, r, t1, t2)
        agents.extend(fp)
    return name, R, agents


def _organic_instance(seed, name, R, arrivals_per_round_max, keystone_have_p=0.06,
                       keystone_want_p=0.32, patience_lo=2, patience_hi=5):
    ni = _rng(seed)
    next_id = 0
    agents = []
    orbit = list(range(1, T))
    for r in range(R):
        n_new = ni(0, arrivals_per_round_max)
        for _ in range(n_new):
            is_keystone_have = ni(0, 999) < int(keystone_have_p * 1000)
            if is_keystone_have:
                have = KEYSTONE
                want = orbit[ni(0, len(orbit) - 1)]
            else:
                have = orbit[ni(0, len(orbit) - 1)]
                wants_keystone = ni(0, 999) < int(keystone_want_p * 1000)
                if wants_keystone:
                    want = KEYSTONE
                else:
                    want = orbit[ni(0, len(orbit) - 1)]
                    if want == have:
                        want = orbit[(orbit.index(have) + 1) % len(orbit)]
            patience = ni(patience_lo, patience_hi)
            agents.append({"id": next_id, "have": have, "want": want,
                            "round": r, "patience": patience})
            next_id += 1
    return name, R, agents


def _build_instances():
    specs = []
    # ---- trap instances (explicit planted clusters + light filler noise) ----
    specs.append(_trap_instance(101, "swap-trap-a", n_clusters=4, spacing=5, filler_n=6))
    specs.append(_trap_instance(102, "swap-trap-b", n_clusters=5, spacing=5, filler_n=8))
    specs.append(_trap_instance(103, "swap-trap-c", n_clusters=4, spacing=6, filler_n=4))
    specs.append(_trap_instance(104, "swap-trap-d", n_clusters=6, spacing=5, filler_n=10))
    specs.append(_trap_instance(105, "swap-trap-e", n_clusters=3, spacing=5, filler_n=3))
    # ---- organic / held-out instances (statistical scarcity, no hand-placed
    #      clusters -- tests that the strategy generalizes) ----
    specs.append(_organic_instance(211, "swap-organic-a", R=45, arrivals_per_round_max=4,
                                    keystone_have_p=0.07, keystone_want_p=0.34, patience_lo=2, patience_hi=4))
    specs.append(_organic_instance(212, "swap-organic-b", R=50, arrivals_per_round_max=4,
                                    keystone_have_p=0.06, keystone_want_p=0.34, patience_lo=2, patience_hi=4))
    specs.append(_organic_instance(213, "swap-organic-c", R=42, arrivals_per_round_max=4,
                                    keystone_have_p=0.05, keystone_want_p=0.37, patience_lo=2, patience_hi=4))
    specs.append(_organic_instance(214, "swap-organic-d-hard", R=55, arrivals_per_round_max=4,
                                    keystone_have_p=0.05, keystone_want_p=0.36, patience_lo=2, patience_hi=3))
    specs.append(_organic_instance(215, "swap-organic-e-hard", R=48, arrivals_per_round_max=4,
                                    keystone_have_p=0.045, keystone_want_p=0.4, patience_lo=2, patience_hi=3))

    out = []
    for (name, R, agents) in specs:
        arrivals = [[] for _ in range(R)]
        for a in agents:
            arrivals[a["round"]].append(dict(a))
        out.append({"name": name, "n_rounds": R, "agents": agents, "arrivals": arrivals})
    return out


# ----------------------------- reference / scoring --------------------------
def _weak_baseline(agents, R):
    """Immediate, length-2-cycles-ONLY, first-found, no hoarding."""
    by_id = {a["id"]: a for a in agents}
    present = set()
    matched = set()
    for r in range(R):
        for a in agents:
            if a["round"] == r:
                present.add(a["id"])
        # drop expired
        present = {i for i in present if by_id[i]["round"] + by_id[i]["patience"] - 1 >= r}
        # index by (have,want) among currently present & unmatched
        pool = [i for i in present if i not in matched]
        by_pair = {}
        for i in pool:
            a = by_id[i]
            by_pair.setdefault((a["have"], a["want"]), []).append(i)
        for i in sorted(pool):
            if i in matched:
                continue
            a = by_id[i]
            rec = (a["want"], a["have"])
            lst = by_pair.get(rec)
            if lst:
                j = None
                for cand in lst:
                    if cand not in matched and cand != i:
                        j = cand
                        break
                if j is not None:
                    matched.add(i)
                    matched.add(j)
    return len(matched)


def _agents_by_id(inst):
    return {a["id"]: a for a in inst["agents"]}


def baseline(inst):
    return _weak_baseline(inst["agents"], inst["n_rounds"])


def _upper_bound(inst):
    return len(inst["agents"])


def score(inst, answer):
    by_id = _agents_by_id(inst)
    R = inst["n_rounds"]
    if not isinstance(answer, dict):
        return False, 0
    rounds = answer.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != R:
        return False, 0
    for rnd in rounds:
        if not isinstance(rnd, dict):
            return False, 0
        cycles = rnd.get("cycles")
        if not isinstance(cycles, list):
            return False, 0
        for cyc in cycles:
            if not isinstance(cyc, list) or len(cyc) not in (2, 3):
                return False, 0
            for x in cyc:
                if isinstance(x, bool) or not isinstance(x, int):
                    return False, 0

    matched = set()
    for r in range(R):
        cycles = rounds[r]["cycles"]
        for cyc in cycles:
            ids = cyc
            if len(set(ids)) != len(ids):
                continue
            if any(i not in by_id for i in ids):
                continue
            if any(i in matched for i in ids):
                continue
            k = len(ids)
            ok_chain = True
            for i in range(k):
                a = by_id[ids[i]]
                b = by_id[ids[(i + 1) % k]]
                if not (a["round"] <= r <= a["round"] + a["patience"] - 1):
                    ok_chain = False
                    break
                if not (b["round"] <= r <= b["round"] + b["patience"] - 1):
                    ok_chain = False
                    break
                if a["want"] != b["have"]:
                    ok_chain = False
                    break
            if not ok_chain:
                continue
            for i in ids:
                matched.add(i)
    return True, len(matched)


def _public_view(inst):
    return {"name": inst["name"], "n_types": T, "type_names": list(TYPE_NAMES),
            "n_rounds": inst["n_rounds"],
            "arrivals": [list(rnd) for rnd in inst["arrivals"]]}


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        y_base = baseline(inst)
        y_ub = _upper_bound(inst)
        denom = y_ub - y_base
        if denom < 1e-9:
            denom = 1e-9
        public = _public_view(inst)
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
            obj = 0
        if not ok:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (obj - y_base) / denom
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
