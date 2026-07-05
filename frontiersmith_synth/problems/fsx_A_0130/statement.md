# Riverside Freight Yard: Classification Track Plan

## Setting

The Riverside freight yard receives a batch of **N inbound freight cars** that must be
sorted onto **K classification tracks** before outbound trains are assembled. Every car
belongs to a **destination block** (the outbound grouping it is bound for); there are
**B blocks** in total. You are the yardmaster: assign **every car to exactly one track**
to minimize the official classification cost.

This is an *offline* heuristic-contest instance. The yard is fixed and generated
deterministically; the score is a fixed formula; there is no wall-clock component.

## Costs (minimize)

For a plan that assigns each car to a track, the cost is

```
cost = over_pen * overflow_cars
     + mix_pen  * mix_pairs
     + split_pen * split_pairs
```

where, counting **unordered pairs of cars**:

- **split_pairs** — pairs of cars in the *same block* placed on *different tracks*
  (same-block cars split apart must be re-coupled later).
- **mix_pairs** — pairs of cars in *different blocks* placed on the *same track*
  (different-block cars sharing a track must be shuffled apart when the train is pulled).
- **overflow_cars** — for each track, `max(0, cars_on_track - cap)` summed over tracks
  (each car beyond a track's capacity `cap` is heavily penalized).

The instances are built so that there are **more blocks than tracks** (some mixing is
unavoidable) and a **few blocks larger than one track's capacity** (some splitting is
unavoidable). There is no clean optimum — it is a capacity-constrained partitioning
trade-off with multiple viable strategies.

## Input (stdin): ONE JSON object — the public instance

```json
{
  "name": "yard1301",
  "n_cars": 130,
  "n_tracks": 9,
  "cap": 20,
  "n_blocks": 11,
  "block": [3, 7, 0, 3, ...],   // length n_cars; block[i] in [0, n_blocks)
  "mix_pen": 2,
  "split_pen": 3,
  "over_pen": 200
}
```

## Output (stdout): ONE JSON object

```json
{ "assign": [t_0, t_1, ..., t_{N-1}] }
```

`assign[i]` is the track chosen for car `i`, an integer in `[0, n_tracks)`.

A plan is **valid** iff `assign` is a list of exactly `n_cars` integers, each in
`[0, n_tracks)`. Wrong length, an out-of-range track, non-integer entries, a crash, a
timeout, or non-JSON output → that instance scores **0.0**. Exceeding a track's
capacity is *allowed* but pays the heavy `over_pen` cost (it is not an invalidation).

## Scoring

Let `cost(plan)` be the formula above. The evaluator computes two references per
instance: `q_base`, the cost of the round-robin plan (`car i -> track i % K`, a weak
but balanced layout), and `q_lb`, a provable lower bound on cost. Your plan's cost
`q_cand` is normalized with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```

- Matching round-robin → `r ≈ 0.1`.
- Reaching the (generally unreachable) lower bound → `r = 1.0`.
- Doing worse than round-robin → `r < 0.1`.

The final score is the **mean of `r` over all instances** (a mix of medium and harder,
larger held-out yards). Because the lower bound ignores the mixing forced by having
more blocks than tracks, even strong planners stay strictly below `1.0`.

## Isolation

Your program is run in an isolated subprocess: it reads the public instance from stdin
and writes its answer to stdout. It never sees the references or scoring internals.

## Suggested strategies

1. **Round-robin** (`i % K`) — the weak baseline.
2. **Marginal-cost greedy** — place each car on the track with the smallest incremental
   split/mix/overflow cost.
3. **Block-partition** — bin-pack whole blocks onto tracks under capacity, splitting only
   the oversized blocks, to confine mixing to the fewest small blocks.
4. **Local search / metaheuristics** — relocate/swap cars with an O(1) cost delta and use
   seeded perturb-and-re-descend restarts to escape local optima.
