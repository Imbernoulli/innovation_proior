# The Clerk Who Redacted Her Own Calendar

A boutique hotel runs a strict privacy audit: before any booking sheet
reaches the night clerk, every guest's actual check-in/check-out dates are
stripped out. All she is handed is a **clash roster** — an anonymised list
of which pairs of bookings *would* overlap on the real calendar — and each
guest's contracted **value**. She has a fixed number of identical **rooms**.
Two guests may only share a room if their (hidden) stays never overlap; the
roster tells her exactly which pairs do. Any guest she cannot seat is turned
away, at a cost equal to that guest's value. Her job: **minimise the total
value of guests turned away.**

What nobody tells her: every roster the hotel ever hands her secretly came
from one real calendar — it is an **interval graph** — with the booking
numbers reshuffled so the calendar order isn't visible from the numbering.
A clerk who notices the roster's structure and reconstructs the hidden
order can seat almost everyone worth seating; one who just reasons about
values, guest by guest, cannot.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from
**stdin**, write ONE JSON object (your answer) to **stdout**. Runs in an
isolated subprocess; sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide who gets which room ...
print(json.dumps({"room": room}))
```

### Public instance (stdin)

```json
{
  "name": "inst201",
  "n": 18,                    // N, number of guests
  "rooms": 3,                 // R, number of identical rooms
  "edges": [[0,3], [0,7], ...],  // clash roster: pairs that MUST NOT share a room
  "value": [12, 40, 8, ...]   // N integer values, one per guest, all >= 1
}
```

The actual check-in/check-out dates are never shown to you — only the
clash roster and the values.

### Answer (stdout)

```json
{ "room": [0, -1, 2, 0, ...] }   // length N; room[i] in {-1, 0, ..., R-1}
```

`room[i] = -1` means guest `i` is turned away. Otherwise `room[i]` is the
room guest `i` is seated in.

A layout is **valid** iff `room` has exactly `N` integer entries, each in
`{-1, ..., R-1}`, and for **every** room, no two guests seated there appear
together in `edges` (that pair would have clashed on the real calendar).
Any invalid output (wrong length, out-of-range index, a clash inside one
room), a crash, a timeout, or non-JSON output makes that instance score
`0.0`.

## Objective

**Minimize** the total value of turned-away guests, summed and normalized
across a fixed, seeded family of 10 instances that vary in guest count,
room count, and clash structure. Several instances are larger, held-out
cases; several are specifically constructed so that reasoning about value
alone (without reconstructing the hidden calendar) performs badly.

## Scoring (deterministic)

For each instance the evaluator computes, itself, from the real hidden
dates (never shown to you):

- `base` — the total value turned away by a **weak reference clerk** who
  seats guests in the roster's own (already scrambled) order, first-fit
  into rooms, using only the clash graph,
- `lb` — a **loose, generally unreachable** lower bound: at every real
  calendar day where more guests are present than there are rooms, *any*
  valid layout must turn away at least the excess count from among the
  guests present that day; `lb` is the worst single day's cheapest way to
  do just that. Because it only ever looks at one day in isolation, it
  never gives credit for one guest's absence fixing several overloaded
  days at once — so it stays strictly below the true achievable optimum on
  most instances, leaving headroom.
- `spill` — the total value your layout turns away.

```
r = clamp( 0.1 + 0.9 * (base - spill) / max(1e-9, base - lb), 0, 1 )
```

Matching the weak clerk scores ≈ `0.1`; doing worse scores below `0.1`;
approaching the (generally unreachable) ideal scores close to, but below,
`1.0`. The reported **Ratio** is the mean of `r` over all instances; the
**Vector** lists the per-instance scores.

## Suggested strategies

1. **Roster-order first-fit** (baseline): seat guests in the order the
   roster lists them, never revisit a decision.
2. **Value-first fit**: sort by value, seat the most valuable guests first
   into the first compatible room.
3. **Detect the hidden structure**: the clash roster is chordal (in fact an
   interval graph). Maximum Cardinality Search recovers a perfect
   elimination ordering; its maximal cliques, joined by a maximum-weight
   spanning tree of clique overlaps, recover a linear calendar order.
4. **Place spills on the reconstructed calendar**: once a linear order is
   available, decide whom to turn away as a resource-constrained
   scheduling problem on that line (e.g. a min-cost flow with one unit of
   flow per room), not by value alone.
