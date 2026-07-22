# Depot Loading Under a Silent Surge

A transfer depot receives a stream of `N` parcels, one at a time. Each truck has
fixed capacity `C`. **The instant a parcel arrives it must be loaded onto some
truck** (an already-open one with room, or a brand-new one) — no looking ahead,
no deferring. Later, during quiet moments, the dispatcher may **re-load** a
bounded number of already-loaded parcels onto a different truck; this re-load
budget is small and strictly capped.

The parcel-size mix is not announced. Some days it stays uniform throughout;
other days it starts small/medium and quietly shifts into a run of oversized
parcels. A dispatcher who packs every truck as tight as possible while the mix
still looks tame ends up with no slack anywhere once the oversized run begins —
and the re-load budget is far too small to undo more than a handful of those
tight loads afterward. Reading the arriving sizes for signs of a shift, holding
some slack back, and timing the re-load budget to consolidate *before* the
shift confirms itself all matter.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**,
write ONE JSON object (your answer) to **stdout**. Runs isolated; sees only the
public instance.

### Public instance (stdin)

```json
{
  "name": "depot306",
  "capacity": 30,
  "n": 40,
  "sizes": [14, 16, 15, 19, ...],   // N integers in arrival order, 1 <= s_i <= C
  "repack_budget": 5                 // B, max total re-load moves allowed
}
```

### Answer (stdout)

```json
{
  "placements": [0, 0, 1, 2, ...],   // length N; truck id parcel i is loaded onto
                                      // AT ARRIVAL (irrevocable at that instant)
  "moves": [{"after": 12, "item": 3, "to": 1}, ...]   // optional, length <= B
}
```

- `placements[i]` is the truck (0..N) parcel `i` is loaded onto the instant it
  arrives; ids need not be contiguous. A truck "exists" once any parcel lands
  on it.
- Each `moves` entry relocates an **already-arrived** parcel `item` (so
  `item <= after`) onto truck `to`, applied right after parcel `after` is
  loaded. `after` values must be **non-decreasing** across the list (moves
  replay in the given order). At most `repack_budget` moves total.
- A truck's load must never exceed `C` — at initial loading or after any move.

Any malformed output, an overfilled truck at any point, a move referencing a
not-yet-arrived parcel, more moves than the budget, non-monotonic `after`
values, a crash, a timeout, or non-JSON output scores that instance `0.0`.

## Objective

**Minimize** the number of trucks holding at least one parcel at the **end** of
the timeline (after all placements and moves), across 10 seeded instances of
varying length, capacity, and arrival pattern — several plain, several with a
mid-stream or ramping shift toward oversized parcels, one a larger held-out case.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `q_lb`   = `ceil(sum(sizes) / C)` — the L1 lower bound (an unreachable ideal),
- `q_base` = trucks used by an internal **next-fit-commit** operator (fills the
  current truck, opens a new one the moment a parcel doesn't fit) — a weak
  online reference,
- `q_cand` = final truck count from replaying **your** `placements` + `moves`,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```

Matching next-fit-commit scores ≈ `0.1`; reaching the L1 bound scores `1.0`;
doing worse than next-fit-commit scores below `0.1`. The L1 bound is loose, so
even strong dispatchers stay below `1.0` — there is real headroom. The reported
**Ratio** is the mean of `r` over all instances; **Vector** lists per-instance
scores.

## Suggested strategies

1. **Next-fit-commit** (baseline): fill the current truck, open a new one the
   moment a parcel doesn't fit, never look back.
2. **Online best-fit**: load each parcel onto whichever open truck leaves the
   least slack; open a new truck only when nothing fits. Reuses gaps well on a
   steady mix, but leaves no slack for a later shift.
3. **Probe, reserve, and time the repack**: compare a short recent window's
   average size against the early stream's average. While they agree, some
   trucks that best-fit would open anyway can be held back from further
   loading instead of topped off, keeping their slack intact. The moment the
   recent average jumps, release those trucks and immediately spend the
   re-load budget pairing lightest with heaviest open trucks that still fit
   together, before the rest of the shift's parcels arrive.
