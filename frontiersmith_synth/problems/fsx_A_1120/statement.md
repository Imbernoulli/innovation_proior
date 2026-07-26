# Reading-Hall Carousel Reshelving

A round reading hall has `K` book **carousels** at physical bays `0..K-1`.
Each carousel has `M` **shelf pockets** `0..M-1`. There are `N = K*M` books,
permanently stamped with an id `0..N-1`: book `id`'s **home** is carousel
`id // M`, pocket `id % M`. A "solved" hall has every book at its home.

An overnight prank scrambled the hall using only the two motorised mechanisms
it actually has — carousels here **only rotate in linked groups**, never
individually and never pocket-by-pocket:

- **Hall drive** (`H+` / `H-`): the shared axle turns the whole ring of `K`
  carousels by one bay, rigidly — every carousel keeps its own pocket order
  and just relocates. `H+` moves the carousel at bay `k` to bay `(k+1) mod K`;
  `H-` is the exact inverse.
- **Shelf belt** (`S{k}+` / `S{k}-`, one pair per bay `k = 0..K-1`): the belt
  inside whichever carousel currently sits at bay `k` advances its `M`
  pockets by one step (`S{k}+`: pocket `p`'s book moves to `(p+1) mod M` of
  the *same* carousel); it never touches any other bay.

You are given one scrambled hall and must output a sequence of these moves that
restores every book to its home. **Minimize the number of moves.**

## Candidate program contract

Standalone program: read one JSON object from stdin, write one JSON object to
stdout. Runs isolated, sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a move list ...
print(json.dumps({"moves": moves}))
```

### Public instance (stdin)

```json
{"name": "hall107", "K": 8, "M": 9, "N": 72, "state": [5, 41, 0, ...]}
```

`state[i]` is the id of the book currently sitting at physical slot
`i = bay*M + pocket` (bay `= i // M`, pocket `= i % M`).

### Answer (stdout)

```json
{"moves": ["H+", "S3-", "S3-", "H-", "S0+", ...]}
```

Each entry must be exactly `"H+"`, `"H-"`, or `"S<k><sign>"` with `0 <= k < K`
and `sign` one of `+`/`-` (e.g. `"S12-"`). Apply them to `state`, in order,
using the rules above. The answer is **valid** iff every move string matches
this grammar (with `k` in range) and, after applying all of them, every slot
`i` holds book `i`. Malformed moves, a crash, a timeout, non-JSON output, or a
final state that isn't fully solved make that instance score `0.0`. Lists
longer than 50000 moves are also rejected (score `0.0`) as a sanity cap — no
correct solution ever needs more than a few hundred; pointless moves under
that cap just cost you score.

## Objective

**Minimize** total moves across a fixed, seeded family of 10 halls (varying
`K` and `M`, including larger held-out halls). Every scramble is itself built
from legal moves, so every instance is solvable.

## Scoring (deterministic)

For each instance the evaluator computes, itself, from the given `state`
(never from your answer):

- `q_base` — moves used by an internal weak reference policy that assumes bay
  `k` already holds carousel `k` and spins its shelf forward hoping to match;
  a full revolution that fails nudges the hall once and tries the next bay —
  a full, always-terminating, but naive fixed point.
- `q_ideal` — an optimistic (not always reachable) anchor: half the true
  minimum number of moves the instance actually requires.
- `q_cand` — moves used by **your** answer (only if valid).

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_ideal), 0, 1 )
```

Matching `q_base` scores ≈ `0.1`; reaching `q_ideal` scores `1.0` (generally
unreachable, since it is set below the true optimum); doing worse than
`q_base` scores below `0.1`. The reported **Ratio** is the mean of `r` over
all 10 instances; **Vector** lists the per-instance scores.

## Suggested strategies

1. **Naive fixed point** (baseline, ~`q_base`): assume bay `k` already holds
   carousel `k`; spin forward hoping to match; on failure, nudge the hall.
2. **Read the stamps**: a book's id tells you its true home carousel/pocket
   directly — align the hall once from that, then spin every belt forward.
3. **Stabilizer-chain solve**: same id-based discovery, but pick whichever
   direction is shorter at *every* level — cost = sum of per-level minima,
   each fixed once and never redone.
