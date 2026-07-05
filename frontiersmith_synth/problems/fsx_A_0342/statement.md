# Feeder Consolidation on a Distribution Substation

A distribution substation must energize a queue of customer **load blocks** onto
identical distribution **transformers**. Each transformer has **two independent
limits**:

- a thermal **capacity** `C` (kVA): the summed demand of the blocks it carries must
  never exceed `C`, or the unit overheats;
- a breaker-**channel** count `K`: the protection panel has only `K` feeder
  breakers, so a transformer can carry **at most `K` distinct load blocks**,
  regardless of how small they are.

Each load block must be energized **whole** on a single transformer (you cannot
split one customer's demand across two units). Energizing a transformer costs one
unit (a tap change plus a cooling loop), no matter how lightly it is loaded.

Your job: write a heuristic that assigns every waiting load block to a transformer
so the whole queue is cleared while energizing **as few transformers as possible**.

This is online-style 1-D bin packing with an added **cardinality** constraint
(a.k.a. K-item bin packing): load blocks = items, thermal limit = bin capacity `C`,
breaker count = per-bin item cap `K`, transformers energized = bins. The queue is
fixed and given to you in full; classic online rules (next-fit, first-fit) are
strong baselines, but reordering and smarter packing can do better.

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
  "name": "sub401",
  "capacity": 20,          // C, thermal cap per transformer (positive integer, kVA)
  "channels": 5,           // K, max distinct blocks per transformer (positive integer)
  "n": 24,                 // N, number of load blocks in the queue
  "demands": [7, 12, 3, 5, ...]   // N integer kVA demands, each 1 <= d_i <= C
}
```

### Answer (stdout)

```json
{ "assign": [0, 0, 1, 0, 2, ...] }   // length N; assign[i] = transformer index of block i
```

- `assign` must be a list of **exactly N** non-negative integers.
- Transformer indices need not be contiguous. A transformer is "energized" iff at
  least one block lands on it; the score counts the number of **distinct non-empty
  transformers**.
- A dispatch is **valid** iff, for every transformer, (a) the total demand assigned
  to it does **not exceed `C`** AND (b) the number of blocks assigned to it does
  **not exceed `K`**.

Any invalid output (wrong length, a non-integer or negative index, a thermally
overloaded transformer, an over-full breaker panel), a crash, a timeout, or
non-JSON output makes that instance score `0.0`.

## Objective

**Minimize** the number of energized transformers across a fixed, seeded family of
12 instances (varying queue length `N`, thermal cap `C`, channel cap `K`, and
demand distribution — small "residential", broad "mixed", half-rating
"commercial", large "industrial", and tiny+heavy "bimodal"). Several instances are
larger / harder held-out cases.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `q_lb`   = `max( ceil(sum(demands)/C), ceil(N/K) )` — the **dual L1 lower bound**
  (an unreachable ideal that respects BOTH the thermal and the channel limit),
- `q_base` = transformers used by an internal **next-fit** dispatcher (a weak
  baseline that opens a new unit as soon as a block over-fills the thermal cap or
  fills the breaker panel),
- `q_cand` = transformers used by **your** dispatch,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```

- Matching next-fit scores ≈ `0.1`; reaching the (generally unreachable) dual L1
  bound scores `1.0`; doing worse than next-fit scores below `0.1`.
- Because L1 is a loose bound, even excellent packers stay below `1.0` on most
  instances — there is real headroom.

The reported **Ratio** is the mean of `r` over all instances; the **Vector** lists
the per-instance scores.

## Suggested strategies

1. **Next-fit** (baseline): keep loading the current transformer until a block
   over-fills the thermal cap or the breaker panel is full, then energize a new
   one — never look back.
2. **First-fit** in arrival order: energize each block onto the lowest-index
   transformer with thermal room AND a free channel; reuse earlier gaps but don't
   reorder the queue.
3. **Decreasing-order packing**: steer the largest blocks first (first-fit- /
   best-fit-decreasing under the K-channel cap) so small blocks top off partly
   loaded transformers.
4. **Local search / metaheuristics**: from a good constructive dispatch, relocate
   or swap blocks between transformers (respecting both `C` and `K`, seeded and
   deterministic) to retire near-empty units.
