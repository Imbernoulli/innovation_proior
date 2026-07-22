# Call Sheet: Rehearsals Before Opening Night

A theater company has `n_days` rehearsal days left before opening night (day
`n_days + 1`). The show has `n_scenes` scenes; each scene needs a fixed subset
of the `n_actors` cast members on stage together. Each day the venue offers
some number of parallel rehearsal rooms (`rooms[d]`), often **fewer near
opening** because the crew is busy with tech / lighting setup. Only one scene
can use a room at a time, and **an actor cannot be in two rooms on the same
day** — so any two scenes rehearsed on the same day must share NO actors.

Each rehearsal of a scene gives it a "recall" boost, but memory fades: a
rehearsal held `g` days before opening (`g = n_days + 1 - day`) is worth only
`boost_s * exp(-decay_s * g)` on opening night — an **exponentially decayed**
fraction of its raw value, decaying at the scene's own forgetting rate
`decay_s`. A scene's opening-night recall is the sum of all its rehearsals'
decayed contributions, **capped** at `cap_s` (an actor can only be so sharp).
The show's total score is the recall-weighted sum over scenes:
`sum_s weight_s * min(cap_s, sum of decayed contributions)`.

Your job: build a full `n_days`-day schedule (which scenes rehearse on which
day) that **maximizes** this total, given the daily room limits and the
actor-sharing conflict rule. A scene may be rehearsed on several different
days (never twice on the *same* day); spacing rehearsals well relative to
opening night — not just rehearsing often — is what earns points.

## Candidate program contract

Standalone program: read ONE JSON public instance from **stdin**, write ONE
JSON answer to **stdout**. Runs isolated, sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a schedule ...
print(json.dumps({"schedule": schedule}))
```

### Public instance (stdin)

```json
{
  "name": "showN",
  "n_scenes": 8, "n_actors": 10, "n_days": 7,
  "rooms": [3, 3, 3, 3, 3, 1, 1],           // rooms available each day, length n_days
  "scene_actors": [[0,2,5], [1,3], ...],    // n_scenes lists of actor indices
  "decay": [0.71, 0.14, ...],               // per-scene forgetting rate (>0)
  "weight": [5, 2, ...],                    // per-scene importance
  "boost": [1.0, 1.0, ...],                 // raw value of one rehearsal
  "cap": [2.3, 1.8, ...]                    // per-scene recall cap
}
```

### Answer (stdout)

```json
{ "schedule": [[0, 3], [1], [0, 2, 4], ...] }   // length n_days; day d's list = scene indices rehearsing that day
```

A schedule is **valid** iff it has exactly `n_days` day-lists, each day-list:
has length `<= rooms[d]`, contains distinct scene indices in
`[0, n_scenes)`, and no two scenes on that day share an actor. Any violation,
wrong shape, non-integer entries, a crash, a timeout, or non-JSON output makes
that instance score `0.0`.

## Scoring (deterministic)

For each of the 10 fixed instances the evaluator computes, itself:

- `ideal` = the (generally unreachable) upper bound obtained by letting every
  scene use *every* day with no room or conflict limits at all,
- `base`  = the total achieved by a weak reference schedule: each scene gets
  **one** rehearsal, placed on the earliest day that still has a free,
  conflict-free room (first-fit, scene-index order, no lookahead),
- `cand`  = the total achieved by your schedule,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (cand - base) / max(1e-9, ideal - base), 0, 1 )
```

Matching the weak one-shot baseline scores ≈ `0.1`; reaching the unconstrained
ideal scores `1.0` (essentially unreachable since it ignores every
constraint). The reported **Ratio** is the mean `r` over all 10 instances; the
**Vector** lists the per-instance scores.

## Notes on the structure

Because value is exponentially decayed *to opening night specifically* (not
to "now"), a rehearsal's worth depends entirely on how close its day is to
`n_days + 1`, not on how "weak" a scene currently looks. Some instances plant
a shared "star" actor across most scenes (forcing heavy serialization) and
shrink the room count sharply in the final days (a tech-week crunch), while
mixing fast-forgetting, high-importance scenes with slow-forgetting ones —
so which scenes get the scarce late-day rooms, and which get threaded into
the roomier early days instead, has a large effect on the total.

## Suggested strategies

1. **One-shot first-fit** (baseline): give every scene a single early
   rehearsal and stop.
2. **Fewest-rehearsals-first**: each day, fill rooms with whichever
   conflict-free scenes have had the fewest rehearsals so far.
3. **Decay-weighted local repair**: rehearse everyone early, then patch the
   biggest shortfalls late.
4. **Backward, value-ranked fill**: walk days from opening night backward;
   each day, greedily seat the conflict-free scenes with the highest
   decay-discounted marginal value for *that* day, capped by remaining need.
