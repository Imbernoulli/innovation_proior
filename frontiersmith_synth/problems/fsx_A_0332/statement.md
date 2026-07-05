# Sump Expedition: Porter Load Planning

A cave-mapping expedition must haul all of its survey gear down a single narrow
shaft to a base camp at the bottom. Gear travels in **porter loads** — one load is
one trip down and back up the shaft. Every trip is expensive, so the expedition
wants to move **all** the gear using as **few porter loads as possible**.

Each porter load must respect **two** limits:

1. **Weight limit `C`** — the total weight of the gear in one load must not exceed
   `C` (the rigging can only take so much on a single descent).
2. **Category limit `K`** — a single load may mix at most `K` **distinct gear
   categories** (ropes, carbide lamps, dye tracers, water samples, …). Each category
   needs its own dedicated stowage and cross-contamination protocol, and a porter
   can manage only `K` of those protocols on one descent. There is **no** limit on
   how many items of the *same* category ride together (beyond the weight limit).

This is **class-constrained 1-D bin packing**: a load can be far below its weight
limit yet still be "full" because it already mixes `K` categories. A good plan must
reason about *which categories share a load*, not just how heavy each load is.

You write a **standalone program** (the candidate) that reads one instance and
emits a loading plan. It runs in an isolated sandbox: it sees only the public
instance below and communicates solely through stdin/stdout.

## Input (stdin): ONE JSON object

```json
{
  "name": "expd101",
  "capacity": 20,          // C: per-load weight limit (int)
  "classes": 3,            // K: per-load distinct-category limit (int)
  "n": 24,                 // number of gear items (int)
  "weights":  [ w_0, ..., w_{n-1} ],   // integer weights, 1 <= w_i <= C
  "category": [ c_0, ..., c_{n-1} ]    // integer category ids, c_i >= 0
}
```

## Output (stdout): ONE JSON object

```json
{ "assign": [ b_0, ..., b_{n-1} ] }
```

`b_i >= 0` is the porter-load index that gear item `i` rides in. Load indices need
not be contiguous; a load "exists" iff at least one item rides it, and the number of
**distinct non-empty loads** is your trip count (to be minimized).

## Validity

A plan is **valid** iff `assign` is a list of exactly `n` non-negative integers and,
for **every** load `b`:

- the total weight of items assigned to `b` is `<= C`, **and**
- the number of distinct categories among items assigned to `b` is `<= K`.

Any violation — wrong length, non-integer / negative index, an over-weight load, a
load mixing more than `K` categories, a crash, a timeout, or non-JSON output — makes
that instance score **0.0**.

## Objective and scoring (deterministic)

Minimize the number of distinct loads `q_cand`. For each instance the evaluator
computes two references it never reveals to the candidate:

- `q_lb  = max( ceil(sum(weights) / C),  ceil(D / K) )` — a lower bound, where `D`
  is the number of distinct categories present. (Total weight forces `>= W/C` loads;
  and with `<= K` classes per load, `B` loads cover at most `B*K` class-slots, so
  `B >= D/K`.) This ideal is usually **unreachable**.
- `q_base` = loads used by an internal **next-fit** operator (a weak baseline).

The per-instance score is an affine anchor (next-fit → 0.1, lower bound → 1.0):

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb),  0, 1 )
```

Matching next-fit scores ≈ 0.1; approaching the (loose) lower bound approaches 1.0;
doing worse than next-fit scores below 0.1. The final score is the mean of `r` over
a fixed, seeded family of 12 instances (weight distributions: light / medium / heavy
/ uniform; varying `n`, `C`, `K`, and the number of categories), including larger
held-out instances for generalization. Because the lower bound is loose, even strong
class-aware packers stay strictly below 1.0 on most instances — there is real
headroom.

## Strategy ladder (rough)

- **trivial** — next-fit loading (reproduces the weak baseline, ≈ 0.1).
- **greedy** — first-fit in arrival order, respecting both limits.
- **strong** — class-aware decreasing packing: keep same-category items together and
  pack heaviest-first with first-fit / best-fit decreasing, preferring loads that
  already carry the item's category so `K` class-slots are conserved.
- Beyond: seeded local search / perturb-and-reoptimize over a class-grouped layout.
