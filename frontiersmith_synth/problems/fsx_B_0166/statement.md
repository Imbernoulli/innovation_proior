# Glacier Sensor Net: Predicate Prefix-Cache Ordering

## Story

A remote **glacier sensor network** streams monitoring queries into a tiny edge query
engine. Each query is a conjunction of predicates over sensor **channels** (ice temperature,
tilt, GPS drift, meltwater conductivity, crevasse strain, ...). To answer a query the engine
evaluates its predicates **one channel at a time** over the raw sensor logs, and it
**memoizes every partial-evaluation prefix** it ever produces — a *prefix KV-cache*:

- The first time the engine materializes a given leading channel-sequence prefix it is a
  cache **MISS** (real work over the logs).
- Any later query whose leading channels reproduce that exact prefix gets a cache **HIT**
  for free.

Within a single query the predicates commute, so the engine is free to evaluate a query's
channels in **any** order — but it must fix **one global channel order** and canonicalize
every query into that order before caching. A good global order clusters channels that are
frequently queried together into shared leading prefixes, so many queries collapse onto the
same cached path and the total work (the number of **distinct prefixes ever materialized**,
i.e. the number of nodes in the prefix trie) drops sharply. A poor order scatters related
channels and forces the engine to recompute almost everything.

You must choose that global channel order. This is exactly the SQL predicate-column
reordering / prefix-cache hit-rate problem, skinned as an offline sensor-net policy:

> **Pick a permutation of the channels that MINIMIZES the number of distinct cached
> prefixes (equivalently MAXIMIZES the prefix-cache hit rate) on the query stream.**

## Offline decision + generalization

You commit to a single global order after seeing only a **training** stream of queries. You
are scored on a **disjoint held-out** stream drawn from the same sensor-net query
distribution. The order must therefore **generalize** to unseen queries, not memorize the
training log.

## Program contract (isolated stdin → stdout)

Your program reads ONE JSON object from **stdin** — the public instance:

```json
{
  "name": "glacier101",
  "N": 16,
  "train_queries": [[0, 3, 7], [3, 7, 11], [0, 3], ...]
}
```

- `N` — number of sensor channels, with integer ids `0 .. N-1`.
- `train_queries` — the training stream; each query is a sorted list of **distinct** channel
  ids (the channels its predicates touch).

Your program writes ONE JSON object to **stdout**:

```json
{"order": [7, 3, 0, 11, 1, ...]}
```

`order` must be a **permutation of `0 .. N-1`**: exactly `N` integers, each channel id once.
The engine evaluates every query by sorting its channels according to this order, then walks
the shared prefix trie. `order[0]` is evaluated first (the shallowest, most-shared cache
level).

**Invalid output** — wrong length, repeated or out-of-range ids, non-integers, a crash, a
timeout, or non-JSON — scores **0.0** on every instance.

## Scoring (deterministic)

For each instance, on the **held-out** stream, let `q(order)` be the number of distinct
materialized prefixes (trie nodes) under a given channel order. We compute:

- `q_base` = `q(0,1,...,N-1)` — the natural channel order (weak baseline).
- `q_cand` = `q(your order)`.
- `q_ref`  = trie nodes under an internal near-optimal order found by seeded local search on
  the held-out stream — a strong reference you cannot see (it uses the hidden stream).

Your per-instance score is an affine anchor (natural order → 0.1, near-optimal → 1.0):

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_ref), 0, 1 )
```

Matching the natural order scores ≈ 0.1; matching the internal reference scores 1.0; doing
worse than the natural order scores below 0.1. Because the reference is optimized on the
held-out stream you never observe, even strong generalizing orders stay strictly below 1.0
on most instances — there is genuine headroom. The final reported **Ratio** is the mean of
`r` over all 12 instances.

## Objective

**Maximize** the mean normalized score (equivalently, minimize distinct cached prefixes on
the unseen stream).

## Hints / viable strategies

- **Frequency-descending**: lead with globally hot channels so common prefixes are shared.
- **Co-occurrence clustering**: keep channels that are queried together (sensor "stations")
  contiguous in the order — this is what actually merges prefixes.
- **Local search**: hill-climb channel swaps against the training trie-node count, starting
  from a principled order; refine gently so you fit the *stable motif structure* rather than
  the exact training paths.
