# The Grand Atrium: Docent Gallery Tours

The Grand Atrium art museum clears its morning queue by grouping visitors into
**guided gallery tours**, each led by a single **docent**. Visitors arrive already
clustered into **tour groups** (families, school cohorts, guided-club parties) that
insist on staying together and therefore must be assigned **entirely to one tour**.

Every tour is constrained by **two independent resources**:

- **Crowd capacity `C`** — the fire-code head limit for one docent-led party moving
  through the galleries. The total number of **people** on a tour may not exceed `C`.
- **Docent-minute budget `T`** — one docent's shift provides `T` minutes of dedicated
  narration / accessibility support. Each group `i` demands `f_i` minutes of that
  attention, and the total demanded by a tour may not exceed `T`.

Your job: assign every waiting group to a tour so the entire queue is cleared using
**as few guided tours (docent shifts) as possible**.

This is **2-D vector bin packing** skinned as a museum: each group is an item with a
two-component demand `(people, docent-minutes)`; each tour is a bin with a two-
component capacity `(C, T)`; "tours used" = bins, which we **minimize**. Unlike plain
1-D packing, a good layout must fill **both** resources — pairing a large-but-quick
group with a small-but-demanding one fills a tour on both axes at once. Online rules
(next-fit, first-fit) are strong baselines, but decreasing-order and best-fit vector
packing can do materially better.

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
  "name": "atrium101",
  "C": 24,                        // crowd capacity, people per tour (positive int)
  "T": 100,                       // docent-minute budget per tour (positive int)
  "n": 40,                        // number of tour groups in the queue
  "people":  [7, 12, 3, 5, ...],  // n integer headcounts,   1 <= p_i <= C
  "minutes": [30, 8, 45, 12, ...] // n integer minute-demands, 1 <= f_i <= T
}
```

### Answer (stdout)

```json
{ "assign": [0, 0, 1, 0, 2, ...] }   // length n; assign[i] = tour index of group i
```

- `assign` must be a list of **exactly n** non-negative integers.
- Tour indices need not be contiguous. A tour is "used" iff at least one group joins
  it; the score counts the number of **distinct non-empty tours**.
- A layout is **valid** iff, for every tour, the total headcount does **not exceed
  `C`** *and* the total docent-minute demand does **not exceed `T`**.

Any invalid output (wrong length, a non-integer or negative index, an over-capacity
or over-time tour), a crash, a timeout, or non-JSON output makes that instance
score `0.0`.

## Objective

**Minimize** the number of tours across a fixed, seeded family of **13** instances
that vary queue length, the two capacities, and the joint demand distribution:
full-range (`unif`), many small groups (`small`), crowd-binding (`crowd_bind`),
time-binding (`time_bind`), and anti-correlated big-quick vs small-demanding groups
(`anti`). Several instances are larger / harder held-out cases (queues up to 90
groups).

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `q_lb`   = `max( ceil(sum(people)/C), ceil(sum(minutes)/T) )` — the **volume lower
  bound** (an unreachable ideal that ignores pairing combinatorics),
- `q_base` = tours used by an internal **2-D next-fit** operator (a weak baseline),
- `q_cand` = tours used by **your** layout,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```

- Matching next-fit scores ≈ `0.1`; reaching the (generally unreachable) volume bound
  scores `1.0`; doing worse than next-fit scores below `0.1`.
- Because the volume bound ignores the combinatorics of co-filling two axes, even
  excellent vector packers stay below `1.0` on most instances — there is real headroom.

The reported **Ratio** is the mean of `r` over all instances; the **Vector** lists
the per-instance scores.

## Suggested strategies

1. **2-D next-fit** (baseline): fill the current tour until a group breaks either
   axis, then open a new one — never look back.
2. **2-D first-fit** in arrival order: reuse room in earlier tours (both axes), but
   don't reorder the queue.
3. **Decreasing-order best-fit vector packing**: sort groups large-first by a
   normalized key (`people/C + minutes/T`, or the max of the two), then place each on
   the tour whose residual capacity best matches its two-component demand (tightest
   leftover / dot-product fit).
4. **Local search / metaheuristics**: from a good constructive layout, relocate or
   swap groups (seeded, deterministic) to co-fill both the crowd and docent-minute
   axes and eliminate half-empty tours.
