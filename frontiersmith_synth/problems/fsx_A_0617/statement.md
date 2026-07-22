# Cold Vault: Place What Can Never Move

A write-once cold-storage vault packs data **blocks** into one linear byte
address space. Each block `i` has a known **size**, a **birth** time and a
**death** time: it occupies memory throughout the half-open interval
`[birth_i, death_i)` and is reclaimed the instant `death_i` arrives. The vault
supports **no compaction and no relocation whatsoever** — once you choose a
block's base offset it is frozen for that block's entire lifetime. Two blocks
whose lifetimes overlap in time must occupy **disjoint** byte ranges; two
blocks whose lifetimes never overlap may reuse the exact same bytes. The
vault's cost is its **high-water mark**: the largest address any block ever
touches. Your job: choose every block's offset to make that high-water mark
as small as possible.

This is offline dynamic storage allocation: the *entire* schedule is given up
front, not revealed one call at a time, so you can plan around future death
times — but every placement is permanent. A block dropped in the wrong spot
can permanently split the free space around it; that fragmentation can never
be compacted away.

## Candidate program contract

Your solution is a **standalone program**: read ONE JSON object (the public
instance) from **stdin**, write ONE JSON object (your answer) to **stdout**.
It runs in an isolated subprocess and sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute an offset for every block ...
print(json.dumps({"offset": offset}))
```

### Public instance (stdin)

```json
{
  "name": "vault_stage_a",
  "n": 96,
  "blocks": [
    {"size": 4, "birth": 0, "death": 220},
    {"size": 4, "birth": 0, "death": 10},
    {"size": 11, "birth": 10, "death": 20},
    ...
  ]
}
```
`blocks[i]` occupies `[birth_i, death_i)`, `size_i >= 1`, `birth_i < death_i`.

### Answer (stdout)

```json
{ "offset": [0, 4, 8, ...] }   // length n; offset[i] = base address of block i
```

- `offset` must be a list of exactly `n` **non-negative integers**.
- A layout is **valid** iff, for every pair of blocks `i, j` whose time
  intervals overlap (`max(birth_i,birth_j) < min(death_i,death_j)`), the byte
  ranges `[offset_i, offset_i+size_i)` and `[offset_j, offset_j+size_j)` are
  **disjoint**.

Any invalid output (wrong length, a non-integer/negative offset, a
time-overlapping byte collision), a crash, a timeout, or non-JSON output makes
that instance score `0.0`.

## Objective

**Minimize** the high-water mark across a fixed, seeded family of 10
instances. Most instances are built in **stages**: each stage introduces a
long-lived "spine" of blocks (alive until the very end) interleaved with a
short-lived cohort (all sharing one death time), followed shortly after by a
cohort of larger blocks that also share a single, slightly later death time.
A few instances are unstructured (random birth/death/size) as generalization
controls. Some instances are larger held-out cases.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `b_fresh` = sum of every block's size — the **fresh-slab** high-water mark
  of a do-nothing allocator that never reuses a freed byte,
- `lb` = the maximum total bytes alive at any single instant — a valid lower
  bound on any feasible high-water mark,
- `lb_anchor` = `floor(0.65 * lb)` — deliberately shrunk **below** that true
  floor, so even a perfect placement cannot saturate the score,
- `q_cand` = the high-water mark of **your** layout,

and normalizes:

```
r = clamp( 0.1 + 0.9 * (b_fresh - q_cand) / (b_fresh - lb_anchor), 0, 1 )
```

Matching the fresh-slab do-nothing layout scores exactly `0.1`. Reusing freed
regions well drives `q_cand` toward `lb` and lifts `r` substantially, but the
shrunk anchor keeps even an optimal placement below `1.0` — there is real
headroom.

The reported **Ratio** is the mean of `r` over all instances; the **Vector**
lists the per-instance scores.

## Suggested strategies

1. **Fresh slab** (baseline): give every block a private, ever-growing range.
2. **Arrival-order first-fit**: reuse holes, but process strictly by birth
   order — blind to which blocks will die together.
3. **Lifetime-keyed placement**: segregate the long-lived spine low, and
   process the rest by death time so an entire equal-death cohort lands in
   one contiguous band that reopens as a single clean hole for later stages.
4. **Search over lifetime-based orders**: try several death-time-keyed
   processing orders and keep whichever yields the lowest high-water mark.
