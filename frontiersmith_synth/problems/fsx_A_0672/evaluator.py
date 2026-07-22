#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0672 -- "Call Sheet: Rehearsals Before Opening Night"
(family: spaced-rehearsal-call-sheet; format B, quality-metric).

THEME.  A theater company has `n_days` rehearsal days left before opening night
(day n_days+1).  The show has `n_scenes` scenes, each needing a fixed subset of
the `n_actors` cast on stage together.  Each day offers `rooms[d]` parallel
rehearsal rooms (often FEWER near opening -- a tech-week crunch).  Two scenes
can only share a day if they share NO actor (an actor cannot be in two rooms at
once) -- SHARED-ACTOR-CONFLICTS.  A rehearsal held g = n_days+1-day days before
opening is worth boost_s * exp(-decay_s * g) on opening night -- EXPONENTIAL
FORGETTING PER SCENE, decaying at the scene's own rate.  A scene's opening-night
recall is the capped sum of all its rehearsals' decayed contributions --
TERMINAL RECALL SCORING.  Total score = sum_s weight_s * recall_s.

INNOVATION HOOK.  Value depends only on decay-discounted proximity to OPENING
NIGHT, not on how weak a scene currently looks.  The naive "give practice to
whoever's had the fewest rehearsals so far" recipe spreads rehearsals evenly
across the whole calendar and lets early reps for fast-forgetting scenes decay
to nothing by opening -- while also failing to specifically reserve the scarce
late-day rooms (tech week) for the scenes that most need them.  The correct
approach reasons BACKWARD from the deadline: fill days closest to opening first
with whichever conflict-free scenes have the highest decay-discounted marginal
value there, letting the leftover early-day capacity (abundant, low
opportunity cost) naturally absorb bottleneck / low-priority scenes.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md.
  stdout: ONE JSON object {"schedule": [[...], ..., [...]]} of length n_days;
          schedule[d] = list of distinct scene indices rehearsing on day d.
  A schedule is VALID iff it has exactly n_days day-lists, each of length
  <= rooms[d], with distinct scene indices in [0, n_scenes), and no two scenes
  scheduled the same day share an actor.  Any violation, wrong shape,
  non-integer entries, a crash, a timeout, or non-JSON output -> that instance
  scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes,
itself, three references:
    ideal = unconstrained upper bound: every scene may use EVERY day (no room
            or conflict limits) -- an upper bound because any feasible
            schedule only uses a SUBSET of days per scene, and per-day value is
            independent of what else is scheduled, so extra days can only help
            (before the cap truncates it).
    base  = a weak ONE rehearsal per scene, first-fit-forward (scene-index
            order, earliest free & conflict-free day), no lookahead at all.
    cand  = the candidate's achieved total.
  and normalizes with an affine anchor (base -> 0.1, ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (cand - base) / max(1e-9, ideal - base), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance.  All references
(ideal, base) are computed by THIS parent process only.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
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
def _build_instance(seed, name, S, A, D, R0, trap):
    ni = _rng(seed)

    def pick_distinct(k, n, force=None):
        chosen = set()
        if force is not None:
            chosen.add(force)
        tries = 0
        while len(chosen) < k and tries < 5000:
            chosen.add(ni(0, n - 1))
            tries += 1
        return sorted(chosen)

    decay = []
    for _ in range(S):
        if trap:
            if ni(0, 99) < 40:
                lam = ni(55, 90) / 100.0          # fast forgetters
            else:
                lam = ni(8, 25) / 100.0           # slow forgetters
        else:
            lam = ni(10, 60) / 100.0
        decay.append(lam)

    scene_actors = []
    for s in range(S):
        k = ni(2, 4)
        # in trap instances, deliberately wire the star actor (index 0) into
        # MOST scenes, and especially into the fast-forgetting ones -- this is
        # what makes the scarce late-day rooms fiercely contested by exactly
        # the scenes that most need them
        force_star = trap and (ni(0, 99) < 65 or decay[s] >= 0.55)
        if force_star:
            sa = pick_distinct(k, A, force=0)
        else:
            sa = pick_distinct(k, A)
        scene_actors.append(sa)

    weight = [ni(1, 5) for _ in range(S)]
    boost = [1.0] * S
    cap = [ni(15, 30) / 10.0 for _ in range(S)]

    if trap:
        # force EVERY fast-forgetting scene to be high-importance and give it a
        # tight cap -- a single well-timed late rehearsal should nearly
        # saturate it, so WHEN it is rehearsed dominates the score, and the
        # star-actor contention (above) means not all of them can get the
        # best slots -- the schedule must decide who gets threaded early.
        for s in range(S):
            if decay[s] >= 0.55:
                weight[s] = 5
                cap[s] = ni(12, 18) / 10.0

    if trap and D >= 4:
        rooms = [R0] * (D - 3) + [2, 1, 1]        # tech-week room crunch near opening
    elif trap and D >= 3:
        rooms = [R0] * (D - 2) + [1, 1]
    else:
        rooms = [R0] * D

    return {"name": name, "n_scenes": S, "n_actors": A, "n_days": D,
            "rooms": rooms, "scene_actors": scene_actors, "decay": decay,
            "weight": weight, "boost": boost, "cap": cap}


def _build_instances():
    # (seed, name, S, A, D, R0, trap)
    specs = [
        (301, "riverside_troupe",   6,  8, 6, 3, False),
        (302, "harbor_players",     8, 10, 7, 3, False),
        (303, "midtown_ensemble",   7,  9, 6, 2, True),
        (304, "east_end_rep",       9, 11, 8, 3, True),
        (305, "chalkline_co",       6,  7, 5, 2, True),
        (306, "lantern_house",     10, 12, 8, 4, False),
        (307, "brickyard_stage",    8, 10, 7, 3, False),
        (308, "far_north_troupe",  11, 13, 9, 4, True),
        (309, "willow_creek_co",    7,  9, 6, 3, False),
        (310, "grand_finale_rep",  12, 14, 9, 4, True),
    ]
    return [_build_instance(seed, name, S, A, D, R0, trap)
            for seed, name, S, A, D, R0, trap in specs]


# ----------------------------- scoring machinery ----------------------------
def _score_schedule(inst, schedule):
    """schedule: list length n_days of lists of scene indices (already validated)."""
    S = inst["n_scenes"]; D = inst["n_days"]
    boost = inst["boost"]; decay = inst["decay"]; cap = inst["cap"]
    acc = [0.0] * S
    for d in range(D):
        gap = D - d                                # day_no = d+1; gap = n_days+1-day_no
        for s in schedule[d]:
            acc[s] += boost[s] * math.exp(-decay[s] * gap)
    return [min(cap[s], acc[s]) for s in range(S)]


def _total(inst, recall):
    return sum(w * r for w, r in zip(inst["weight"], recall))


def _validate_schedule(inst, answer):
    D = inst["n_days"]; S = inst["n_scenes"]
    rooms = inst["rooms"]; scene_actors = inst["scene_actors"]
    if not isinstance(answer, dict):
        return None
    sched = answer.get("schedule")
    if not isinstance(sched, list) or len(sched) != D:
        return None
    out = []
    for d in range(D):
        day = sched[d]
        if not isinstance(day, list):
            return None
        if len(day) > rooms[d]:
            return None
        seen = set(); used_actors = set(); clean = []
        for x in day:
            if isinstance(x, bool) or not isinstance(x, int):
                return None
            if x < 0 or x >= S:
                return None
            if x in seen:
                return None
            seen.add(x)
            aset = set(scene_actors[x])
            if aset & used_actors:
                return None
            used_actors |= aset
            clean.append(x)
        out.append(clean)
    return out


def _ideal_total(inst):
    """Upper bound: every scene may use every day (rooms & conflicts ignored)."""
    S = inst["n_scenes"]; D = inst["n_days"]
    boost = inst["boost"]; decay = inst["decay"]; cap = inst["cap"]
    tot = 0.0
    for s in range(S):
        raw = sum(boost[s] * math.exp(-decay[s] * gap) for gap in range(1, D + 1))
        tot += inst["weight"][s] * min(cap[s], raw)
    return tot


def _base_total(inst):
    """Weak reference: one rehearsal per scene, earliest free & conflict-free day."""
    S = inst["n_scenes"]; D = inst["n_days"]
    rooms = inst["rooms"]; scene_actors = inst["scene_actors"]
    room_used = [0] * D
    actor_used = [set() for _ in range(D)]
    schedule = [[] for _ in range(D)]
    for s in range(S):
        aset = set(scene_actors[s])
        for d in range(D):
            if room_used[d] < rooms[d] and not (aset & actor_used[d]):
                schedule[d].append(s)
                room_used[d] += 1
                actor_used[d] |= aset
                break
    recall = _score_schedule(inst, schedule)
    return _total(inst, recall)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        ideal = _ideal_total(inst)
        base = _base_total(inst)
        denom = ideal - base
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "n_scenes": inst["n_scenes"],
                  "n_actors": inst["n_actors"], "n_days": inst["n_days"],
                  "rooms": list(inst["rooms"]),
                  "scene_actors": [list(a) for a in inst["scene_actors"]],
                  "decay": list(inst["decay"]), "weight": list(inst["weight"]),
                  "boost": list(inst["boost"]), "cap": list(inst["cap"])}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            sched = _validate_schedule(inst, ans)
        except Exception:
            sched = None
        if sched is None:
            vec.append(0.0); continue
        recall = _score_schedule(inst, sched)
        cand_total = _total(inst, recall)
        r = 0.1 + 0.9 * (cand_total - base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0); continue
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
