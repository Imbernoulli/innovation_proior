# The Doorman's List

You work the door of a club. You can't memorize everyone's face, only a
fixed list of at most `capacity` regulars at a time — an instant nod-through
for anyone on it. Everyone else still gets in fine, you just don't bother
learning their face.

The night streams in **rounds**. Each round brings a batch of arrivals (a
list of face-ids). For every arrival already on your list, you get an
instant recognition — a **hit**. For every arrival *not* on your list, you
decide: learn their face (adding them, evicting someone if the list is
full) or don't bother. Your score for the whole night is the **total hit
count over every round, maximized**.

Two things make the crowd hard to memorize honestly:

- **Tour buses.** Periodically a busload of one-shot tourists floods the
  door — every face in that round is brand new and will **never appear
  again**, ever. Learning any of them is pure waste (at best one hit,
  forever, for a slot that could serve a regular).
- **Crowd rotation.** The club alternates between `n_eras` distinct,
  *fixed* regular crowds on a schedule (e.g. two eras alternate: crowd 0,
  crowd 1, crowd 0, crowd 1, ...). A face belongs to exactly one era and
  reappears **only** during that era's rounds.

**Both schedules are told to you exactly, every round** — this is not
something you need to infer from behaviour.

## Candidate program contract

Standalone program, invoked **once per round** (many times per test case —
a fresh, isolated subprocess call each time, no memory across calls).

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide this round's admissions/evictions ...
print(json.dumps({"decisions": [...], "state": ...}))
```

### Public input for round `r` (stdin)

```json
{
  "round": 7, "total_rounds": 24, "capacity": 50,
  "scan_period": 7, "scan_span": 1, "scan_phase": 3,
  "inversion_period": 6, "n_eras": 2,
  "floor": ["v7a2b_0_3", "v7a2b_1_0", ...],
  "state": <whatever you returned last round, or null on round 0>,
  "arrivals": ["v7a2b_0_9", "b7a2b_7_0", ...]
}
```

- `floor`: your memorized list **right now** (ground truth, always current).
- A round is a **bus round** iff `scan_period > 0` and
  `(round - scan_phase) mod scan_period < scan_span`.
- Otherwise it's a **crowd round**; the active era is
  `(round // inversion_period) mod n_eras` (or era 0 if `n_eras <= 1`).
- `state`: an opaque JSON value **you** control — nothing here persists in
  process memory between rounds, so any history you want (recency, which
  era a face belongs to, ...) must be carried in this field. Max ~60000
  serialized characters.

### Answer (stdout)

```json
{"decisions": [{"action": "admit"|"skip", "evict": "<face-id>"|null}, ...], "state": ...}
```

Exactly one decision per entry of `arrivals`, **in order**. A decision is
only consulted for arrivals **not currently on your list** (already-listed
arrivals are automatic hits, decision ignored). `"skip"`: `evict` must be
`null`. `"admit"`: if the list has room, `evict` must be `null`; if the
list is full, `evict` must name a face **currently on your list right
now** — accounting for admissions/evictions **earlier in this same
round**. Any malformed decision, wrong list length, an oversized `state`,
a crash, a timeout, or non-JSON output makes the **entire night score 0**.

## Scoring (deterministic)

For each of 10 fixed, seeded nights, the grader also computes `baseline` =
the hit count of "memorize the first faces you ever see, then freeze the
list forever" (always feasible). Per-night score
`r = min(1, 0.1 * hits / baseline)`. Reported **Ratio** is the mean over
all 10 nights; **Vector** lists the per-night scores.

## What makes this hard

Chasing raw recency treats a tour bus's one-shot faces as maximally
"important" (they just arrived) and evicts your real regulars to make
room for them — right before those regulars would have paid off. And once
the crowd rotates, faces from the crowd due back next rotation look just
as "cold" to plain recency as faces that are gone forever, so nothing
protects the one from the other. Both schedules are handed to you exactly
— use them to decide *when* to change how you evict, not just *whom*.

## Constraints

Time limit 5 s per round call, memory 512 MB. `capacity <= 150`,
`total_rounds <= 90`, each round's arrival batch `<= ~250`. Scoring is
fully deterministic (all randomness is seeded by the grader).
