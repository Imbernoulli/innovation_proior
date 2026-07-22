# Seed-Swap Network: Heirloom Steward

## Story

A community seed-swap network trades heirloom plant varieties. There are `T`
seed varieties (types) `0..T-1`. Type `0` is **Moon-and-Stars**, a coveted
heirloom that almost everyone wants but almost nobody currently holds. Every
other type is an ordinary "orbit" variety, plentiful on both sides.

Gardeners arrive over `n_rounds` rounds. Each gardener `g` shows up already
holding one packet of variety `have[g]` and wants exactly one packet of
variety `want[g]` (`have[g] != want[g]`). A gardener stays in the pool for
`patience[g]` consecutive rounds starting at `round[g]` (inclusive); if still
unswapped after that window, they leave for good, empty-handed.

Each round, your policy may execute any number of **disjoint swap cycles** of
length 2 or 3 among gardeners **currently present** (arrived, not yet expired,
not yet swapped). A cycle is a list of distinct gardener ids `[a0, a1, ..., ak-1]`
(`k` in `{2,3}`) meaning `a0` receives `a1`'s packet, `a1` receives `a2`'s
packet, ..., `a(k-1)` receives `a0`'s packet -- formally `want[a_i] ==
have[a_{(i+1) mod k}]` for every `i`. Executing a valid cycle satisfies every
gardener in it simultaneously; they leave the pool happy. A packet is used at
most once (whoever offered it leaves with it gone), so a variety that is
`have`-scarce can only ever satisfy **one** cycle per holder.

Your goal: **maximize the total number of gardeners who complete a swap**
over the whole season. Since Moon-and-Stars holders are rare, the moment one
shows up matters: swapping them into the very first available cycle can strand
a same-round trading partner (fine on its own) while a slightly larger cycle,
completable one round later, would have satisfied more gardeners overall using
that same holder. Letting an unmatched gardener sit costs nothing by itself --
it only costs you if their patience runs out before a cycle ever closes.

## Input (public instance, one JSON object on stdin)

```json
{
  "name": "swap07",
  "n_types": 6,
  "type_names": ["Moon-and-Stars Melon", "Cherokee Purple Tomato", ...],
  "n_rounds": 20,
  "arrivals": [
    [ {"id": 0, "have": 1, "want": 0, "round": 0, "patience": 3}, ... ],
    [ ... round 1 ... ],
    ...
  ]
}
```

`arrivals` has exactly `n_rounds` entries (round `r`'s list may be empty).
Every gardener id is a unique non-negative integer. `round[g]` always matches
the index of the list it appears in.

## Output (one JSON object on stdout)

```json
{"rounds": [ {"cycles": [[a0, a1], [b0, b1, b2], ...]}, ... ]}
```

`rounds` must have exactly `n_rounds` entries in order; round `r`'s `cycles`
lists the swap cycles you execute at round `r` (an empty list is fine). Any
malformed top-level shape (wrong length, non-list `cycles`, a cycle whose
length isn't 2 or 3, non-integer ids) makes the **whole answer** score `0.0`
for that instance. A cycle that references an unknown id, an id already used
elsewhere, an absent/expired gardener, or that fails the `want[a_i] ==
have[a_{(i+1) mod k}]` chain is simply **skipped** (not applied, no other
penalty) -- garbage cycles just don't count.

## Objective and scoring (deterministic)

For each instance the evaluator computes, independently of your answer:

- `y_base`: gardeners satisfied by a weak reference policy that greedily
  executes only immediately-available **length-2** cycles, round by round,
  the instant they appear (never waits, never considers length-3 cycles).
- `y_ub`: total number of gardeners who ever arrive (a loose, generally
  unreachable upper bound -- some are always doomed by supply/demand
  imbalance on the scarce variety).
- `y_cand`: gardeners satisfied by your submitted schedule.

```
r = clamp( 0.1 + 0.9 * (y_cand - y_base) / max(1e-9, y_ub - y_base), 0, 1 )
```

Matching the weak reference scores about `0.1`; catching more distinct swaps
scores higher, capped at `1.0`. Because `y_ub` is loose, even a strong policy
stays below `1.0`. Your final score is the mean of `r` over all instances
(several planted scarcity clusters plus harder, more organic held-out
instances).

## Notes

- Scoring never depends on wall-clock time; treat the time limit as a safety
  cap for search-heavy strategies.
- Your program runs in an isolated subprocess and sees only the JSON instance
  above.
