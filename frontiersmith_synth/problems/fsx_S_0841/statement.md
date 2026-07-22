# Hot-Key Repair: Bounded Migration under Power-of-Two Choices

## Story
A key-value store already placed `N` keys onto `S` shards; `shard0[i]` is key `i`'s
**current** shard. Each key carries a real load weight `weight[i]`. Keys are listed in
**creation order** (id 0 = oldest); a handful of the most recently created keys ("viral"
content) carry most of the weight — a Zipfian **skew** on top of a large, boring,
near-uniform majority. Because the initial placement never looked at weight, a few heavy
keys can be stacked on the same shard(s), overloading them.

You may **repair** the placement, but migration is expensive: at most `budget` keys
total may end up somewhere other than `shard0[i]` (a hard cap on the *count* of moved
keys, not on how much weight they carry). A moved key also cannot go anywhere: every key
`i` comes with exactly two precomputed alternative shards `alt[i] = [a, b]` (its
consistent-hash "power-of-two" choices; `a != b`, both `!= shard0[i]`). If you move key
`i`, its final shard must be one of `shard0[i]`, `alt[i][0]`, `alt[i][1]`.

## Task
Write a **standalone program**: read ONE JSON instance from `stdin`, write ONE JSON
answer to `stdout`.

### Public instance (stdin)
```json
{ "name":"skew1", "S":12, "N":260,
  "shard0":[...N ints in [0,S)...],
  "weight":[...N positive numbers...],
  "alt":[[a_0,b_0], ..., [a_{N-1},b_{N-1}]],
  "budget": 9 }
```

### Answer (stdout)
```json
{ "assign": [f_0, ..., f_{N-1}] }
```
`f_i` is key `i`'s final shard.

### Validity
`assign` must be a list of exactly `N` integers (no bool/float/NaN/inf). For every `i`,
`f_i` must be in `{shard0[i], alt[i][0], alt[i][1]}`. The total count of keys with
`f_i != shard0[i]` must be `<= budget`. Any violation — wrong length, an entry outside
the allowed set, too many moves, a crash, a timeout, or non-JSON output — scores **0.0**
on that instance.

## Objective & scoring (deterministic)
For an assignment, `load(s)` is the sum of `weight[i]` over keys with `f_i == s`, and
`M = max_s load(s)` (**lower is better**). Per instance the evaluator computes:

- `M_cand` — your assignment's max shard load,
- `M_base` — the max shard load of doing nothing (`assign == shard0`),
- `LB = (sum of all weight) / S` — the perfectly-balanced ideal (generally unreachable,
  since moves are capped in count and each key can only reach two precomputed shards).

```
r = clamp( 0.1 + 0.9 * (M_base - M_cand) / (M_base - LB), 0, 1 )
```
Doing nothing scores exactly `0.1`. Because of the count cap and the two-choice
restriction, even an optimal repair cannot reach `LB`, so `r < 1` always. The final score
is the mean of `r` over **10** fixed seeded instances: skewed single-hotspot instances,
near-uniform instances with no heavy tail, and held-out instances with **two**
independent hotspots.

## Why it is open-ended
Rebalancing whichever key you meet first is a trap: with a small budget and hundreds of
keys, a blind pass over keys in id (creation) order spends its whole budget on ordinary,
low-weight keys before it ever reaches the handful of viral keys that actually determine
the max load — the count cap makes moving a light key just as "expensive" as moving a
heavy one, while its payoff is tiny. Diagnosing which keys are heavy first, and only then
spending the budget on their two power-of-two alternatives (tracking the running load so
later heavy keys don't all pile onto the same relief shard), reaches a far lower max
load with the same budget. There is no closed-form optimum: with two hotspots, a fixed
per-key candidate pair, and a hard count cap, the right split of a small budget across
multiple overloaded shards is itself a genuine trade-off.

## Isolation
Your program runs in a fresh sandboxed subprocess and only ever sees the public instance
above. The do-nothing and lower-bound references are computed by the evaluator process.
