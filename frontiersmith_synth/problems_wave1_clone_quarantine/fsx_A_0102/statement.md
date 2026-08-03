# Solstice Festival: Main-Stage Slotting

The Solstice Festival lays out an entire evening lineup on a bank of **identical
main stages**. Acts — bands, DJs, dance troupes — are booked in a **fixed running
order**. Each act reserves a **footprint** of stage resources (backline power draw,
crew, floor area), summarised as one integer. A single stage can host several acts
across the night, but only under **two** limits:

- **Resource capacity `C`:** the combined footprint of the acts on a stage must not
  exceed `C`.
- **Changeover cap `K`:** at most `K` acts may be slotted on a stage — there are
  only `K` soundcheck / changeover windows per stage per night.

Rolling out a stage — trucks, rigging, a full sound system, a licensed crew — costs
one **stage**, no matter how many acts play on it. Your job: write a heuristic that
slots **every booked act** onto a stage so the whole lineup is covered using **as
few stages as possible**.

This is **cardinality-constrained 1-D bin packing** (acts = items, stage capacity =
bin capacity, per-stage act cap `K` = bin cardinality limit, stages = bins). The
extra count limit is the twist: a stage can be "full" either because it is out of
resource capacity **or** because it already holds `K` acts, so a good layout has to
respect **both** constraints at once. The lineup is given to you in full; classic
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
  "name": "fest401",
  "capacity": 30,          // C, resource capacity per stage (positive integer)
  "max_acts": 6,           // K, max acts per stage (positive integer)
  "n": 40,                 // N, number of acts in the lineup
  "acts": [17, 9, 22, 4, ...]   // N integer footprints, each 1 <= s_i <= C
}
```

### Answer (stdout)

```json
{ "assign": [0, 0, 1, 0, 2, ...] }   // length N; assign[i] = stage index of act i
```

- `assign` must be a list of **exactly N** non-negative integers.
- Stage indices need not be contiguous. A stage is "rolled out" iff at least one
  act is slotted on it; the score counts the number of **distinct non-empty
  stages**.
- A layout is **valid** iff, for every stage, (i) the total footprint of its acts
  does **not exceed `C`**, and (ii) it holds **no more than `K` acts**.

Any invalid output (wrong length, a non-integer or negative index, an over-capacity
stage, a stage with more than `K` acts), a crash, a timeout, or non-JSON output
makes that instance score `0.0`.

## Objective

**Minimize** the number of stages rolled out across a fixed, seeded family of
12 instances (varying lineup length, stage capacity `C`, changeover cap `K`, and
act-size distribution — uniform-ish "medium", small-plus-large "bimodal",
big-rig "heavy", and many-small "light"). Several instances are larger / harder
held-out cases where the count cap `K` bites hardest.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `q_lb`   = `max( ceil(sum(acts) / C), ceil(N / K) )` — the **combined L1 lower
  bound** (an unreachable ideal that respects both capacity and the count cap),
- `q_base` = stages used by an internal **next-fit** operator (a weak baseline),
- `q_cand` = stages used by **your** layout,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```

- Matching next-fit scores ≈ `0.1`; reaching the (generally unreachable) combined
  L1 bound scores `1.0`; doing worse than next-fit scores below `0.1`.
- Because L1 is a loose bound, even excellent packers stay below `1.0` on most
  instances — there is real headroom.

The reported **Ratio** is the mean of `r` over all instances; the **Vector** lists
the per-instance scores.

## Suggested strategies

1. **Next-fit** (baseline): load the current stage until an act won't fit (by
   capacity or the K-act cap), then open a new stage — never look back.
2. **First-fit** in booking order: reuse gaps on earlier stages that still have
   both room and a free changeover window, but don't reorder the lineup.
3. **Cardinality-aware decreasing-order packing**: slot the largest acts first
   (first-fit- / best-fit-decreasing) respecting both `C` and `K`, so small acts
   top off partly-filled stages.
4. **Local search / metaheuristics**: from a good constructive layout, relocate or
   swap acts (seeded, deterministic) to empty out whole stages.
