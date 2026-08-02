# Ants That Must Not All Follow the Same Trail

A forager colony sends out `A` ants every round, split across `K` food
**sources**. But ants don't teleport their commitment: a source's usable output
this round is driven by the colony's accumulated **pheromone trail** to it, and
that trail only catches up to a new allocation gradually — it builds when ants
keep arriving and evaporates when they stop:

```
trail_i(t)     = decay_i * trail_i(t-1) + (1 - decay_i) * a_i(t)
potential_i(t) = rate_i * trail_i(t)
harvest_i(t)   = min(stock_i(t), potential_i(t))
stock_i(t+1)   = min(cap_i, stock_i(t) - harvest_i(t) + regen_i)
```

Every source is a finite, regenerating larder: `stock_i` is depleted by
whatever gets harvested and refills by `regen_i` per round, capped at `cap_i`.
Your job: choose how many ants to send to each source, every round, to
**maximize total food harvested** over `T` rounds.

Sending everything to the source with the best raw per-ant `rate` looks
optimal — right up until that source's one-time surplus stock is drained down
to its regen floor. From that point every extra ant camped there is wasted
(harvest is capped by `stock`/`regen`, not by trail), while a never-fed source
has zero trail and needs several rounds of investment before it can absorb
ants productively at all (the `decay_i` ramp-up lag). A policy that reinforces
the best-known trail forever locks the colony onto a depleting source and
starves once it dries up.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**,
write ONE JSON object (your answer) to **stdout**. Runs isolated; sees only the
public instance.

### Public instance (stdin)

```json
{
  "name": "colony3201",
  "K": 4, "T": 20, "A": 55,
  "sources": [
    {"stock0": 480, "cap": 540, "regen": 11, "rate": 1.62, "decay": 0.86},
    {"stock0": 640, "cap": 870, "regen": 33, "rate": 0.71, "decay": 0.83},
    ...
  ]
}
```

`sources` has exactly `K` entries, each with the five fields above (all
non-negative; `0 <= decay < 1`).

### Answer (stdout)

```json
{ "alloc": [[14, 14, 14, 13], [12, 15, 13, 15], ...] }
```

- `alloc` must have exactly `T` rows, each of exactly `K` non-negative
  integers (idle ants — a row summing below `A` — are allowed and simply do
  nothing that round).
- **Invalid** iff: wrong shape, a non-integer/negative/boolean entry, or any
  row's sum exceeds `A`. Also invalid on crash, timeout, or non-JSON output.
  Any of these scores that instance `0.0`.

## Objective

**Maximize** total harvest (`sum` of all `harvest_i(t)`) across a fixed,
seeded family of 10 instances: 3 "abundant" warm-ups where every source has
generous stock/regen (depletion never really binds), and 7 "patchy" layouts
where one source has a deceptively high `rate` but a small, slow-regenerating
stock — a jackpot that looks best at a glance and punishes anyone who commits
to it for the whole horizon. Several instances use a longer horizon or more
sources as held-out generalization cases.

## Scoring (deterministic)

For each instance the evaluator computes, itself, two references never shown
to you:

- `base` = total harvest of an **equal, never-adapting split** of `A` ants
  across all `K` sources every round,
- `ub`   = `1.08 *` the best harvest found by a generic seeded local search
  (many random single-ant transfers from a few plain starting policies) — a
  loosely optimistic, essentially unreachable ceiling.

```
r = clamp( 0.1 + 0.9 * (harvest_cand - base) / max(1e-9, ub - base), 0, 1 )
```

- Matching the equal split scores ≈ `0.1`; reaching the local-search ceiling
  scores close to `1.0` (never exactly, by the 1.08 margin); doing worse than
  equal split scores below `0.1`.

The reported **Ratio** is the mean of `r` over all 10 instances; **Vector**
lists the per-instance scores.

## Suggested strategies

1. **Equal split** (baseline): divide `A` evenly across sources, never adapt.
2. **Static rate-weighted split**: weight the (fixed, whole-horizon) split by
   each source's `rate` — better than equal, but blind to depletion.
3. **Phased exploit + pre-warm**: track each source's *headroom* (stock above
   its sustaining `regen`); spend most ants exploiting the best-headroom
   source while a reserve pre-builds trail on the next one in line, so the
   hand-off lands exactly when headroom runs out.
4. **Tune it**: grid-search the ordering rule, reserve fraction, and switch
   threshold by simulating each config exactly, keeping the best per instance.
