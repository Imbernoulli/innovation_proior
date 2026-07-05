# Twilight Carnival: Gondola Dispatch Circuit

The Twilight Carnival's flagship attraction is a circular **gondola circuit**. All
evening, thrill-seekers arrive at the loading platform in **parties** — families and
friend groups that refuse to be split up and therefore must all ride in the **same
gondola**. Every gondola is identical, with a fixed **seat capacity** `C`. A single
gondola may carry several parties at once, as long as their combined headcount does
not exceed `C`. Dispatching a gondola around the circuit costs one **dispatch**
(fuel plus a full loop), regardless of how full it is.

Your job: write a heuristic that assigns every waiting party to a gondola so that the
entire queue is cleared using **as few dispatched gondolas as possible**.

This is online-style 1-D bin packing (parties = items, gondola capacity = bin
capacity, dispatches = bins). The queue is fixed and given to you in full; classic
online rules (next-fit, first-fit) are strong baselines, but reordering and smarter
packing can do better.

## Candidate program contract

Your solution is a **standalone program**: read ONE JSON object (the public
instance) from **stdin**, write ONE JSON object (your answer) to **stdout**. It runs
in an isolated subprocess and sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute an assignment ...
print(json.dumps({"assign": assign}))
```

### Public instance (stdin)

```json
{
  "name": "circuit101",
  "capacity": 20,          // C, seats per gondola (positive integer)
  "n": 24,                 // N, number of parties in the queue
  "parties": [7, 12, 3, 5, ...]   // N integer headcounts, each 1 <= s_i <= C
}
```

### Answer (stdout)

```json
{ "assign": [0, 0, 1, 0, 2, ...] }   // length N; assign[i] = gondola index of party i
```

- `assign` must be a list of **exactly N** non-negative integers.
- Gondola indices need not be contiguous. A gondola is "dispatched" iff at least
  one party boards it; the score counts the number of **distinct non-empty
  gondolas**.
- A layout is **valid** iff, for every gondola, the total headcount of the parties
  assigned to it does **not exceed `C`**.

Any invalid output (wrong length, a non-integer or negative index, an overfilled
gondola), a crash, a timeout, or non-JSON output makes that instance score `0.0`.

## Objective

**Minimize** the number of dispatched gondolas across a fixed, seeded family of
12 instances (varying queue length, gondola capacity, and party-size
distribution — uniform, half-gondola "medium", small+large "bimodal", and
large-party "heavy"). Several instances are larger / harder held-out cases.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `q_lb`   = `ceil(sum(parties) / C)` — the **L1 lower bound** (an unreachable ideal),
- `q_base` = dispatches used by an internal **next-fit** operator (a weak baseline),
- `q_cand` = dispatches used by **your** layout,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```

- Matching next-fit scores ≈ `0.1`; reaching the (generally unreachable) L1 bound
  scores `1.0`; doing worse than next-fit scores below `0.1`.
- Because L1 is a loose bound, even excellent packers stay below `1.0` on most
  instances — there is real headroom.

The reported **Ratio** is the mean of `r` over all instances; the **Vector** lists
the per-instance scores.

## Suggested strategies

1. **Next-fit** (baseline): fill the current gondola until a party doesn't fit,
   then open a new one — never look back.
2. **First-fit** in arrival order: reuse gaps in earlier gondolas, but don't
   reorder the queue.
3. **Decreasing-order packing**: seat the largest parties first (first-fit- /
   best-fit-decreasing) so small parties top off partially-filled gondolas.
4. **Local search / metaheuristics**: from a good constructive layout, swap or
   relocate parties (seeded, deterministic) to squeeze out wasted seats.
