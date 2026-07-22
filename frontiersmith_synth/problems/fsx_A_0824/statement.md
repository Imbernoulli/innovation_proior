# One Question, Four Shuffled Haystacks

## Story

There is a fixed public universe of `N` labeled items, ids `0..N-1`. A
target item is hidden somewhere in it. It was drawn from one of `K=4`
**scenario pools** — each pool is its own probability distribution over the
*same* `N` items, built by clumping most of that pool's mass onto a small
"home" cluster of items scattered arbitrarily through the id space, with a
thin residue spread over everything else. Every pool clumps its items
differently — four differently shuffled haystacks sharing one search space.

You write **one program** that outputs an entire adaptive questioning
**strategy** — a binary decision tree of subset probes ("is the target in
`S`?") — built once, in advance, from the public pool weights. You are
never told which pool the real target came from, and there is no
back-and-forth: the evaluator then walks your tree once per possible
target, using truthful membership answers, to see how well it would have
done under every scenario.

## Input (stdin) — ONE JSON object

```json
{
  "n_items": 12,
  "items": [0, 1, 2, ..., 11],
  "n_pools": 4,
  "pools": [
    {"pool": 0, "weights": [0.014, 0.322, ...]},
    {"pool": 1, "weights": [...]},
    {"pool": 2, "weights": [...]},
    {"pool": 3, "weights": [...]}
  ]
}
```

Each `weights` array has `n_items` non-negative entries summing to 1 — pool
`k`'s belief about where the target is. You are **not** told which items
form which pool's "cluster"; that structure is only implicit in the
weights themselves (a pool's own high-weight items are its clumping).

## Output (stdout) — ONE JSON object

```json
{ "tree": <node> }
```

A `<node>` is either:
- `{"query": [id, id, ...], "yes": <node>, "no": <node>}` — probe "is the
  target in this list of item ids?" (a non-empty, proper subset of the
  remaining universe), then recurse into the matching branch, or
- `{"guess": id}` — a leaf naming the final answer.

**Validity.** For *every* item `i` (as the true target), replaying your
tree with truthful yes/no answers must terminate at a `guess` equal to `i`
within a generous depth cap. A crash, non-JSON output, a malformed node, an
out-of-range or duplicate id in a `query`, or a wrong/missing guess for
**any** item invalidates the **entire instance** (score `0.0` for it).

## Objective & scoring (deterministic)

For a valid tree, let `probes(i)` be the number of `query` nodes visited
before reaching item `i`'s leaf. Each pool's own quality is its own
probes-weighted mean:

```
mean_k = sum_i pool_k.weights[i] * probes(i)
```

Your instance objective is the **maximum over the 4 pools** of `mean_k`
(you must **minimize** this) — a strategy that serves three pools
beautifully but mishandles the fourth is judged on the fourth, not the
average. The evaluator normalizes with a fixed affine anchor, computed
directly (never from your program):

- `weak` = the objective of a fixed "scan items `0,1,2,...` one at a time"
  reference tree that ignores the pools entirely.
- `lb` = `max_k H(pool_k)` (Shannon entropy, bits) — an information floor
  no tree can beat for pool `k` alone; the shared-tree constraint across
  all 4 pools keeps this essentially unreachable in practice, leaving
  headroom above even a strong policy.

```
r = clamp( 0.1 + 0.9 * (weak - your_objective) / max(weak - lb, 1e-6), 0, 1 )
```

Your final score is the mean of `r` over 10 fixed instances (`N` from 12 to
14, always `K=4` pools, differently shuffled clumpings).

## Notes

- Fully deterministic and seeded — no wall-clock or hardware timing
  anywhere in the objective; your program is called exactly once per
  instance, isolated, and only ever sees that instance's public weights.
- A tree that first asks about a *contiguous* id range chosen to balance
  the four pools' *pooled average* is a very reasonable, textbook
  approach — but the split that is best for the average can slice right
  through one pool's own clumping, forcing that one pool to pay for a
  probe that told it almost nothing. Strategies that read structure out of
  the weights themselves and protect it, rather than just balancing raw
  item counts, are worth exploring.
