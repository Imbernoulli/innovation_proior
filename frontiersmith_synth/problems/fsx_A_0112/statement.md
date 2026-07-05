# Contact-Net Cohorting: Quarantine Pod Assignment

An outbreak-response team has just flagged a batch of **contacts** from the
contact-tracing net and must house every one of them in a physical **quarantine
pod**. Each contact carries two pieces of information:

- an integer **viral load** `w` (a triage number, `1 <= w <= C`), and
- a **strain lineage** `s` (a small integer label), determined by sequencing.

Every pod is identical and imposes **two** limits:

1. a **load capacity** `C` — the total viral load of a pod's occupants must not
   exceed `C`; and
2. a **cross-contamination limit** `K` — a pod may host occupants from at most `K`
   **distinct** strain lineages (mixing more lineages risks recombination and is
   forbidden).

Opening a pod costs one facility, regardless of how full it is. Your job: assign
every contact to a pod so that the whole batch is housed using **as few pods as
possible**.

This is 1-D bin packing with an added **class/color constraint** (contacts = items
that each have a *size* and a *color*, pod capacity = bin capacity, the `K`-limit
caps the number of distinct colors per bin, pods = bins). The color limit makes
weight-only heuristics strand whole lineages, so it is genuinely harder than plain
bin packing. The full batch is given to you at once; classic online rules
(next-fit, first-fit) are decent baselines, but reordering and class-aware packing
can do better.

## Candidate program contract

Your solution is a **standalone program**: read ONE JSON object (the public
instance) from **stdin**, write ONE JSON object (your answer) to **stdout**. It
runs in an isolated subprocess and sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute an assignment ...
print(json.dumps({"assign": assign}))
```

### Public instance (stdin)

```json
{
  "name": "pods101",
  "capacity": 20,             // C, load capacity per pod (positive integer)
  "max_strains": 2,           // K, max distinct lineages per pod (positive integer)
  "n": 24,                    // N, number of contacts
  "loads":   [7, 12, 3, 5, ...],   // N integer viral loads, each 1 <= w_i <= C
  "strains": [0, 3, 1, 0, ...]     // N integer lineage labels, each >= 0
}
```

### Answer (stdout)

```json
{ "assign": [0, 0, 1, 0, 2, ...] }   // length N; assign[i] = pod index of contact i
```

- `assign` must be a list of **exactly N** non-negative integers.
- Pod indices need not be contiguous. A pod is "opened" iff at least one contact
  lives in it; the score counts the number of **distinct non-empty pods**.
- A layout is **valid** iff **every** non-empty pod satisfies BOTH limits: total
  load `<= C` **and** number of distinct lineages `<= K`.

Any invalid output (wrong length, a non-integer or negative index, an over-capacity
pod, a pod mixing more than `K` lineages), a crash, a timeout, or non-JSON output
makes that instance score `0.0`.

## Objective

**Minimize** the number of pods across a fixed, seeded family of **12 instances**
(varying batch size, pod capacity `C`, lineage limit `K`, number of lineages, and
load distribution — tiny "small" loads, half-pod "medium", broad "mixed", and
large "heavy"). Several instances are larger / harder held-out cases.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `q_lb`   = `ceil(sum(loads) / C)` — the **L1 lower bound** (an unreachable ideal
  that *ignores* the color limit),
- `q_base` = pods used by an internal **color-aware next-fit** operator (a weak
  baseline),
- `q_cand` = pods used by **your** layout,

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```

- Matching next-fit scores ≈ `0.1`; reaching the (generally unreachable) L1 bound
  scores `1.0`; doing worse than next-fit scores below `0.1`.
- Because L1 **ignores** the `K`-lineage limit, even excellent class-aware packers
  stay below `1.0` on the color-dominated instances — there is real headroom.

The reported **Ratio** is the mean of `r` over all 12 instances; the **Vector**
lists the per-instance scores.

## Suggested strategies

1. **Color-aware next-fit** (baseline): fill the current pod until a contact's load
   doesn't fit or would exceed `K` lineages, then open a new pod — never look back.
2. **Color-aware first-fit** in arrival order: reuse gaps in earlier pods that
   still have a free lineage slot, but don't reorder arrivals.
3. **Class-aware decreasing-order packing**: sort by load (and/or by lineage), and
   best-fit each contact into a pod that already holds its lineage when possible so
   whole lineages fill a pod's `K` color slots before spilling over.
4. **Local search / metaheuristics**: from a good constructive layout, relocate or
   swap contacts (seeded, deterministic) — respecting both the load and `K`-lineage
   limits — to reclaim stranded capacity.
