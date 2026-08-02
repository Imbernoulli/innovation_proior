# Mapping a World Through a Keyhole

A cartographer's guild hands you a partially-surveyed dungeon: the **room
topology** is fully drawn, but the true nature of each interactive
**fixture** inside those rooms is not. Old field-notes give you a **cue**
phrase per fixture — a hint, never a certainty — and you must commit to a
full expedition **itinerary in advance**: nobody radios you updates
mid-expedition, you write the whole plan and it runs.

Every fixture is either:
- **Reversible (safe)**: interacting with it always succeeds, banking its
  reward and, sometimes, an item.
- **Irreversible (risky)**: interacting with it succeeds *only if you are
  currently holding a `ward`* (consumed on use). Without a ward in hand, the
  attempt ends the expedition on the spot — every action still left in your
  submitted plan is simply never carried out.

Wards are scarce: they are handed out by a subset of the *safe* fixtures.
Nothing in the fixture list ever says "risky" outright — you must read it
off the fixture's **cue grammar**. Every cue contains exactly one adjective
drawn from one of two fixed pools, identical across every instance:

- `DECAY_WORDS` (fixture is **risky**): crumbling, trembling, creaking,
  hollow-sounding, frayed, splintered, rusted-through, warped
- `STABLE_WORDS` (fixture is **safe**): sturdy, gleaming, freshly-oiled,
  solid, well-anchored, polished, intact, firmly-set

This mapping is 100% reliable, true for every world in this family — it is
part of the task, not a secret to reverse-engineer. Nothing else in the data
(including whether a fixture grants an item) reliably tells you the type;
only the cue does.

## Candidate program contract

Standalone program: read ONE JSON public instance from **stdin**, write ONE
JSON answer to **stdout**. Isolated subprocess, public view only.

### Public instance (stdin)
```json
{
  "name": "collapsed-nave", "rooms": 8, "start": 0, "turn_budget": 32,
  "edges": [[0,1],[1,2], "..."],
  "objects": [
    {"id": "o0", "room": 1, "verb": "cross",
     "cue": "Through the keyhole you note: the stone bridge looks trembling.",
     "reward": 9, "gives_item": null},
    {"id": "o3", "room": 4, "verb": "open",
     "cue": "Through the keyhole you note: the iron-bound chest looks sturdy.",
     "reward": 4, "gives_item": "ward"}
  ]
}
```
`edges` are bidirectional passages. Every fixture in `objects` sits in exactly
one room and can be interacted with at most once. `gives_item` is `"ward"`
for some safe fixtures, and `null` otherwise (never a reliable safe/risky
tell by itself).

### Answer (stdout)
```json
{"actions": [{"type": "goto", "room": 4}, {"type": "interact", "object": "o3"}]}
```
- `{"type":"goto","room":R}` costs `shortest_path_hops(current_room, R)` turns
  (computed on the public room graph) and moves you there.
- `{"type":"interact","object":ID}` costs 1 turn and interacts with the named
  fixture, which must be in your current room and not already used.
- Execution stops the instant the turn budget (`turn_budget`) runs out, or the
  instant you interact with a risky fixture while holding zero wards.
- Malformed or unrecognized steps are ignored (still may cost a turn); a
  non-JSON answer, or an answer whose `"actions"` is not a list, scores this
  instance `0.0`.

## Objective

**Maximize**, per instance: `2 * (distinct rooms visited) + (total reward
collected)`, across a fixed, seeded family of 10 worlds — 3 trap-free
warm-ups (no risky fixtures at all) and 7 trapped worlds where a risky
fixture sits within easy reach of the start, before any ward can plausibly
be found by an itinerary that doesn't plan ahead.

## Scoring (deterministic)

For each instance the evaluator computes, itself:
- `q_base` = the objective of the "explore fast, interact with everything the
  instant you find it, never read a cue" reference itinerary,
- `q_ub`   = an optimistic, generally-unreachable ideal: every room visited,
  every safe reward collected, plus the highest-reward risky fixtures up to
  the world's true ward supply — with travel cost and the turn budget both
  ignored,
- `q_cand` = the objective of your submitted itinerary,

and normalizes with an affine anchor:
```
r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, q_ub - q_base), 0, 1 )
```
Matching the blind-exhaustive reference scores ≈`0.1`; reaching the
(generally unreachable) ideal scores `1.0`; doing worse than blind exhaustion
scores below `0.1`. The reported **Ratio** is the mean of `r` over all
instances; **Vector** lists the per-instance scores.
