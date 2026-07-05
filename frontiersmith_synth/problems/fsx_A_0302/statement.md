# Aperture Array: Exposure-Block Scheduling

## Story

An aperture-synthesis radio telescope **array** is grinding through the night's queue
of observation requests. Each request must be observed for an integer **exposure
duration** (tracking minutes). The correlator integrates requests into an **exposure
block**, but a single block can hold at most `capacity` exposure minutes before it must
be flushed to disk and a fresh block opened.

Requests arrive in a **fixed order** (the queue the dynamic scheduler already committed
to). Every request must be assigned to some exposure block, and the total duration
packed into any one block may **never exceed `capacity`**.

Opening a block costs correlator reconfiguration and re-calibration overhead, so the
operator wants to clear the entire queue using the **fewest exposure blocks**. A myopic
"keep one block open, flush it the moment the next request will not fit" policy wastes
capacity: a short request that could have topped off an earlier block instead forces a
brand-new one. Reusing earlier blocks, and reasoning about which durations to pair,
packs far tighter. That is the tension you must resolve.

This is online 1-D bin packing skinned as a telescope-array scheduling contest.

## Input (public instance, one JSON object on stdin)

```json
{
  "name": "queue101",
  "capacity": 100,
  "items": [23, 45, 12, 88, ...]
}
```

- `capacity` (int `C`): exposure-minute budget of a single block.
- `items` (list of ints): request durations **in arrival order**; each `1 <= d <= C`.

## Output (one JSON object on stdout)

```json
{"assign": [b0, b1, b2, ...]}
```

- Exactly `len(items)` entries: `assign[i]` is the exposure-block index for request `i`.
- Each `b_i` is an **integer `>= 0`**. Indices need not be contiguous.
- The number of blocks **used** is the count of **distinct** indices in `assign`.
- For every block, the sum of durations assigned to it must be `<= capacity`.

Any of the following makes the instance score `0.0`: wrong number of entries, a
non-integer or negative index, any block over capacity, a crash, a timeout, or output
that is not the JSON object above.

## Objective and scoring (deterministic)

You **minimize** the number of distinct exposure blocks used. For each instance the
evaluator computes:

- `B_nf` = blocks used by **Next-Fit** (one open block; flush and open a fresh one
  whenever the next request does not fit). This overlap-blind online policy is the weak
  baseline.
- `L1` = `ceil(sum(items) / capacity)` -- the area lower bound. No packing can use fewer
  blocks, and fragmentation usually makes it unreachable.
- `B` = blocks used by your validated assignment.

and normalizes (weak baseline -> `0.1`, area-optimal -> `1.0`):

```
r = clamp( 0.1 + 0.9 * (B_nf - B) / max(1e-9, B_nf - L1), 0, 1 )
```

Reproducing Next-Fit scores about `0.1`; using more blocks scores below `0.1`; packing
tighter scores higher, capped at `1.0`. Because the area bound is generally
unreachable, even excellent schedules stay below `1.0` on the hard queues -- there is
always headroom. Your final score is the mean of `r` over all instances (a mix of
queue lengths, capacities, and duration distributions, including harder held-out queues
whose near-half-capacity requests fragment badly).

## Notes

- Scoring depends only on your emitted `assign`; it never measures wall-clock time.
  Treat the per-instance limit as an operation budget for search-based methods
  (first-fit, best-fit, decreasing-order reordering, local reshuffles, restarts).
- Your program is run in an isolated subprocess and sees only the public instance above.
