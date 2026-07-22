import sys, json, math, random, isorun

# ==========================================================================
# fsx_B_0630 -- twin-car-shaft-elevators (Format B, isolated candidate)
# Theme: two elevator cars per shaft serving a morning rush.
#
# Each shaft holds TWO cars (role 0 = lower, role 1 = upper) that share one
# hoistway: at EVERY tick the upper car must stay at least G floors above the
# lower car (pos_upper - pos_lower >= G). They can never pass or come within
# the safety gap. Cars move 1 floor/tick (LOOK/scan discipline, infinite
# capacity). A passenger's WAIT is the number of ticks from arrival until a
# car reaches the origin floor and boards them; unserved passengers cost W_cap.
#
# The candidate submits, for each hall call, WHICH car (shaft, role) serves it
# and, for each car, a PARK floor it idles at (its position when it has no
# pending work, and its position at t=0). The evaluator then replays a
# deterministic simulation and scores SUM of squared waits (MINIMIZE).
#
# Mechanisms composed into the objective:
#   * shared-shaft-noncrossing : the gap couples the two cars -- a car cannot
#     reach a floor on the far side of its partner; a badly-assigned car is
#     blocked and its passengers wait.
#   * anticipatory-parking     : each car's t=0 position and idle-rest floor
#     are the candidate's `park`; placing an idle car at its band edge (not in
#     its partner's path, near where its next demand appears) cuts waits.
#   * request-batching          : a car sweeps and boards every arrived passenger
#     it passes in its current direction, so grouping co-directional calls onto
#     one car serves them in a single sweep.
#
# Innovation hook: the non-passing constraint turns dispatch into an online
# interval-partition problem. Committing each car to a demand-tuned floor BAND
# and re-parking the idle car to the band edge beats any nearest-car greedy,
# which drives both cars of a shaft into the same busy zone where the safety
# gap serializes them.
# ==========================================================================


def simulate(F, S, G, T, W_cap, calls, assign, park):
    """Deterministic integer replay. Returns (obj, n_unserved) or None if the
    answer is structurally infeasible (caller rejects)."""
    ncars = 2 * S
    # initial positions = park floors; must respect the gap per shaft
    pos = [int(park[c]) for c in range(ncars)]
    for s in range(S):
        lo, up = pos[2 * s], pos[2 * s + 1]
        if lo < 0 or up < 0 or lo > F - 1 or up > F - 1:
            return None
        if up - lo < G:
            return None
    direction = [1] * ncars
    # per-car assigned call ids
    car_calls = [[] for _ in range(ncars)]
    for i, (sh, ro) in enumerate(assign):
        car_calls[sh * 2 + ro].append(i)

    M = len(calls)
    arr = [calls[i]["t"] for i in range(M)]
    org = [calls[i]["o"] for i in range(M)]
    dst = [calls[i]["d"] for i in range(M)]
    pickup = [None] * M
    onboard = [set() for _ in range(ncars)]     # boarded, not yet dropped
    pending = [set() for _ in range(ncars)]      # assigned, arrived, not boarded

    # index calls per car by arrival tick for admission
    by_tick = {}
    for c in range(ncars):
        for i in car_calls[c]:
            by_tick.setdefault(arr[i], []).append((c, i))

    def stops_for(c):
        st = set()
        for i in pending[c]:
            st.add(org[i])
        for i in onboard[c]:
            st.add(dst[i])
        return st

    def pick_target(c):
        st = stops_for(c)
        p = pos[c]
        if not st:
            return int(park[c]), 0                      # go idle-park
        d = direction[c]
        ahead = [f for f in st if (f - p) * d > 0]
        if ahead:
            tgt = min(ahead, key=lambda f: abs(f - p))
            return tgt, d
        # nothing ahead -> reverse
        d = -d
        cand = [f for f in st if (f - p) * d > 0]
        if cand:
            tgt = min(cand, key=lambda f: abs(f - p))
            return tgt, d
        return p, direction[c]

    def busy(c):
        return bool(pending[c]) or bool(onboard[c])

    def board_drop(c, t):
        p = pos[c]
        for i in sorted(list(onboard[c])):
            if dst[i] == p:
                onboard[c].discard(i)
        for i in sorted(list(pending[c])):
            if org[i] == p:
                pickup[i] = t
                pending[c].discard(i)
                onboard[c].add(i)

    for t in range(T):
        # admit newly-arrived calls into their car's pending set
        for (c, i) in by_tick.get(t, []):
            pending[c].add(i)
        # board/drop at current position BEFORE moving
        for c in range(ncars):
            board_drop(c, t)
        # decide targets/directions
        des = [0] * ncars
        for c in range(ncars):
            tgt, d = pick_target(c)
            # only update the persisted SWEEP direction on a real (non-idle)
            # target pick (d != 0); an idle/park drift must never zero out a
            # car's sweep direction, or it can never resume picking targets
            # via the ahead/reversed-scan logic in pick_target() again.
            if d != 0:
                direction[c] = d
            des[c] = (0 if tgt == pos[c] else (1 if tgt > pos[c] else -1))
        # resolve movement shaft by shaft with the non-crossing coupling.
        # Whenever the two desired moves would violate the gap, the LOWER
        # -priority car yields: it retreats (moves further away) instead of
        # freezing, so the higher-priority car always makes progress. Simply
        # cancelling BOTH moves on conflict (as a naive resolver would) is a
        # correctness bug: if both cars want to approach every tick (each
        # one's next stop sits on the far side of its partner), cancelling
        # both forever reproduces the SAME conflict next tick -> a permanent
        # deadlock that starves most of the fleet. Forcing a yield guarantees
        # the priority car's gap-to-target shrinks every tick it is blocked,
        # so progress is monotel and bounded.
        for s in range(S):
            cl, cu = 2 * s, 2 * s + 1
            pL, pU = pos[cl], pos[cu]
            dL, dU = des[cl], des[cu]
            # priority: busy beats idle; among busy, earliest waiting passenger;
            # tie -> lower car.
            def keyc(c):
                active = list(pending[c]) + list(onboard[c])
                ea = min((arr[i] for i in active), default=10 ** 9)
                return (0 if busy(c) else 1, ea, c)
            order = sorted([cl, cu], key=keyc)
            hiP, loP = order[0], order[1]

            newL = max(0, min(F - 1, pL + dL))
            newU = max(0, min(F - 1, pU + dU))
            if newU - newL < G:
                # The low-priority car ALWAYS yields (retreats away from its
                # partner) on a gap conflict -- regardless of whether ITS OWN
                # desired move was "approaching." An earlier version only
                # yielded when the low-priority car was itself closing the
                # gap, and otherwise froze the high-priority car; that let a
                # low-priority car resting exactly at its (badly chosen) park
                # floor block its partner FOREVER (park never changes, so the
                # freeze condition never lifted) -- a single misplaced park
                # floor could strand a passenger for the whole horizon. A
                # real dispatcher lets an idle car get out of the way, so the
                # low-priority car now always retreats one floor away from
                # the high-priority car's approach (capped at the building
                # boundary); this keeps the safety-gap serialization cost
                # (the whole point of the family) while removing unbounded
                # single-call deadlocks from a stationary idle car.
                if loP == cu:
                    # upper car (low priority) yields: retreat upward
                    newU = max(pU, min(F - 1, pU + 1))
                    newL = max(0, min(F - 1, pL + dL))
                else:
                    # lower car (low priority) yields: retreat downward
                    newL = min(pL, max(0, pL - 1))
                    newU = max(0, min(F - 1, pU + dU))
                if newU - newL < G:
                    newL, newU = pL, pU   # no room anywhere -> hold both
            pos[cl], pos[cu] = newL, newU
        # board/drop again at the new positions (so a car that just arrived acts)
        for c in range(ncars):
            board_drop(c, t)

    obj = 0.0
    unserved = 0
    for i in range(M):
        if pickup[i] is None:
            w = W_cap
            unserved += 1
        else:
            w = pickup[i] - arr[i]
            if w < 0:
                w = 0
        obj += float(w) * float(w)
    return obj, unserved


def _trivial_construction(pub):
    """Baseline / trivial-tier construction: pure round-robin across ALL 2*S
    cars by arrival index -- no floor information used at all (not even a
    fixed midpoint split). Every car gets a fair, even share of calls, but
    which shaft/role serves a call has nothing to do with where it is. Park
    every lower car at the bottom and upper car at the top."""
    F, S = pub["F"], pub["S"]
    calls = pub["calls"]
    ncars = 2 * S
    assign = []
    for i, c in enumerate(calls):
        car = i % ncars
        assign.append([car // 2, car % 2])
    park = []
    for s in range(S):
        park.append(0)          # lower car parks at the bottom
        park.append(F - 1)      # upper car parks at the top
    return assign, park


def make_instances():
    specs = [
        # (F, S, G, T, seed, profile)  profile drives the demand mix.
        (14, 2, 2, 260, 0, "rush_mixed"),
        (16, 2, 3, 300, 1, "rush_lobby"),
        (13, 2, 2, 240, 2, "midcluster"),      # TRAP: mid-building cluster
        (15, 3, 2, 280, 3, "rush_mixed"),
        (17, 2, 3, 320, 4, "midcluster"),      # TRAP
        (12, 2, 2, 220, 5, "banded"),
        (16, 3, 2, 300, 6, "rush_lobby"),
        (15, 2, 2, 280, 7, "midcluster"),      # TRAP
        (18, 2, 3, 320, 8, "banded"),
        (14, 3, 2, 260, 9, "rush_mixed"),
    ]
    out = []
    for (F, S, G, T, seed, profile) in specs:
        rng = random.Random(4200 + seed)
        calls = []
        M = rng.randint(24, 34)
        t = 0
        for k in range(M):
            t += rng.randint(1, 6)               # staggered morning arrivals
            if profile == "rush_lobby":
                # heavy lobby surge going up + a little inter-floor
                if rng.random() < 0.7:
                    o = 0
                    d = rng.randint(2, F - 1)
                else:
                    o = rng.randint(1, F - 2)
                    d = rng.randint(0, F - 1)
            elif profile == "midcluster":
                # origins clustered in the middle band -> nearest-car drags both
                # cars of a shaft into the same zone.
                mid = F // 2
                o = min(F - 1, max(0, mid + rng.randint(-2, 2)))
                d = rng.randint(0, F - 1)
            elif profile == "banded":
                # cleanly separable: half low-origin, half high-origin
                if rng.random() < 0.5:
                    o = rng.randint(0, F // 2 - 1)
                    d = rng.randint(0, F // 2)
                else:
                    o = rng.randint(F // 2, F - 1)
                    d = rng.randint(F // 2 - 1, F - 1)
            else:  # rush_mixed
                r = rng.random()
                if r < 0.45:
                    o = 0
                    d = rng.randint(2, F - 1)
                elif r < 0.75:
                    o = rng.randint(F // 2, F - 1)
                    d = rng.randint(0, F // 2)
                else:
                    o = rng.randint(1, F - 2)
                    d = rng.randint(0, F - 1)
            if d == o:
                d = (o + 1) % F
            calls.append({"id": k, "t": t, "o": int(o), "d": int(d)})
        W_cap = T
        pub = {"F": F, "S": S, "G": G, "T": T, "W_cap": W_cap, "calls": calls}
        out.append({"public": pub, "hidden": {}})
    return out


def baseline(inst):
    pub = inst["public"]
    assign, park = _trivial_construction(pub)
    res = simulate(pub["F"], pub["S"], pub["G"], pub["T"], pub["W_cap"],
                   pub["calls"], assign, park)
    obj, _ = res
    return obj


def score(inst, ans):
    pub = inst["public"]
    F, S, G, T = pub["F"], pub["S"], pub["G"], pub["T"]
    calls = pub["calls"]
    M = len(calls)
    ncars = 2 * S
    if not isinstance(ans, dict):
        return False, 0.0
    assign = ans.get("assign")
    park = ans.get("park")
    if not isinstance(assign, list) or len(assign) != M:
        return False, 0.0
    if not isinstance(park, list) or len(park) != ncars:
        return False, 0.0
    clean_assign = []
    for a in assign:
        if not isinstance(a, (list, tuple)) or len(a) != 2:
            return False, 0.0
        sh, ro = a
        if not isinstance(sh, int) or not isinstance(ro, int):
            return False, 0.0
        if sh < 0 or sh >= S or ro not in (0, 1):
            return False, 0.0
        clean_assign.append([sh, ro])
    clean_park = []
    for p in park:
        if not isinstance(p, int):
            if isinstance(p, float) and p == int(p):
                p = int(p)
            else:
                return False, 0.0
        if p < 0 or p > F - 1:
            return False, 0.0
        clean_park.append(p)
    res = simulate(F, S, G, T, pub["W_cap"], calls, clean_assign, clean_park)
    if res is None:
        return False, 0.0
    obj, _ = res
    if obj != obj or obj < 0 or obj in (float("inf"), float("-inf")):
        return False, 0.0
    # keep the objective strictly positive so normalization is well-defined
    return True, obj + 1.0


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, stt = isorun.run_candidate(cand, inst["public"], timeout=20)
        if stt != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst) + 1.0
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
