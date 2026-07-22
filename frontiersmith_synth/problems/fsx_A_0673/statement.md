# One Cache Config Against Five Workload Personalities

You are shipping a cache configuration *before* you know which workload will hit it.
Production will replay five very different fixed access traces — nicknamed **zipf**,
**phase**, **scan**, **bursty**, and **loop** — against your ONE fixed answer, and you
are graded on the **worst** of the five resulting hit rates. A configuration that is
brilliant on four traces and terrible on the fifth scores as if it were terrible
everywhere.

Your answer has two parts:

1. **A pin set** — up to `pin_budget` keys that are permanently resident for the whole
   replay of *every* trace: always a hit, never evicted, and they do not eat into the
   normal cache — they are extra guaranteed capacity.
2. **Eviction weights** — four numbers `w_lru, w_mru, w_lfu, w_scan` (each in `[0, 8]`)
   that parametrize one generic online eviction rule, applied identically and purely
   causally (no lookahead) to all five traces.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**, write
ONE JSON object (your answer) to **stdout**. It runs isolated and sees only the public
instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... analyze inst["traces"] ...
print(json.dumps({"pin": pin, "w_lru": a, "w_mru": b, "w_lfu": c, "w_scan": d}))
```

### Public instance (stdin)

```json
{
  "instance_id": "cache2101",
  "capacity": 24,
  "pin_budget": 6,
  "universe_size": 90,
  "traces": {
    "zipf":   [3, 17, 3, 3, 22, ...],
    "phase":  [0, 9, 9, 2, 11, ...],
    "scan":   [4, 60, 61, 0, 62, ...],
    "bursty": [1, 44, 44, 47, 44, ...],
    "loop":   [0, 1, 2, 3, 60, 61, ...]
  }
}
```

Each trace is a fixed list of integer key ids in `[0, universe_size)`. Nothing is
hidden — the whole trace is given to you; the difficulty is choosing ONE
`(pin, weights)` pair that survives the replay of all five simultaneously.

### Answer (stdout)

`pin`: a list of **at most `pin_budget` distinct** integers in `[0, universe_size)`.
`w_lru, w_mru, w_lfu, w_scan`: finite numbers in `[0, 8]`.

Any wrong type/shape, out-of-range/duplicate pin entry, non-finite or out-of-range
weight, a crash, timeout, or non-JSON output scores that instance `0.0`.

## Replay mechanics (identical for every trace, run independently per trace)

The non-pinned part of the cache has `capacity - len(pin)` slots ("managed region").
At each access to key `k`: pinned or resident → **hit**. Otherwise **miss**: let
`scan_signal` = fraction of the last 24 accesses whose key was that key's first-ever
appearance in this trace (near 1 during a long one-shot sweep of fresh keys, low
during steady reuse). If `w_scan>0`, `k` is first-ever, and
`scan_signal * w_scan >= 1.0`, the miss is **bypassed** (nothing cached, nothing
evicted). Otherwise, if the managed region has a free slot, insert `k`; else evict the
resident `j` maximizing

```
w_lru * recnorm(j) + w_mru * (1 - recnorm(j)) - w_lfu * freqnorm(j)
```

where `recnorm`/`freqnorm` normalize each resident's (time since last use) and
(uses so far) by the current max among residents. `w_lru=1` alone reproduces plain
LRU; `w_mru` alone reproduces MRU (evict the newest — the classic fix for cyclic
sweeps longer than the cache); `w_lfu` protects frequently-reused residents.

## Objective and scoring

**Maximize** the mean, over 10 fixed instances, of a normalized worst-trace hit rate.
Per instance the evaluator computes, itself: `q_base` (worst-trace hit rate of a
pin-nothing, plain-LRU reference) and `q_opt` (worst-trace hit rate of an
*unconstrained*, offline-optimal Belady cache of the same total capacity — a true,
generally unreachable ceiling), then

```
r = clamp(0.1 + 0.9 * (q_cand - q_base) / max(1e-9, 1.5 * (q_opt - q_base)), 0, 1)
```

Matching the plain-LRU baseline scores ≈0.1; even matching the offline-optimal ceiling
leaves headroom below 1.0. The reported **Ratio** is the mean `r`; **Vector** lists the
10 per-instance scores.

## Suggested strategies

1. Pin nothing, plain LRU (the reference).
2. Pin the globally most-frequent keys (summed over all five traces) and keep LRU.
3. Pin whichever keys recur across the *most* traces first (cheap insurance for every
   personality), then spend any leftover budget on whichever single trace is currently
   weakest; pick weights that blend recency, anti-recency, frequency, and a
   scan-admission gate.
4. Local search: swap pinned keys and perturb the weights, keeping only changes that
   raise the worst-trace hit rate.
