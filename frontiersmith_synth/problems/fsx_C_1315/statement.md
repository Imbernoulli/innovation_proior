# Telling People Where To Go When They Will Not Listen

A venue must be cleared through several **exits**. Its occupants are grouped
into **zones**. You are the guidance system: every step you may broadcast one
exit per zone as that zone's directive. People do not fully obey.

## Dynamics (simulated exactly; every constant below is in the input)

Each zone `i` has an egress rate `egress_cap[i]`: at most that many people can
newly *start* moving out per step. At step `t`, the cohort that departs zone
`i` splits: a **compliant** fraction follows this step's broadcast exit
`g_i(t)`; the rest heads for the zone's fixed **default exit** (its nearest)
regardless of what you broadcast — they were never listening.

The compliant fraction is `base_compliance[i] * credibility[i](t)`, where
`credibility[i](t)` starts at `1.0` and evolves once per step, for `t >= 1`:
if `g_i(t) != g_i(t-1)` (directive changed), credibility is **multiplied**
by `credibility_decay[i]` (< 1) — contradicting earlier guidance costs trust,
until rebuilt; if `g_i(t) == g_i(t-1)` (repeated), credibility **increases**
by `credibility_recover[i]`, capped at `1.0`.

People who head for an exit join that exit's **queue** and stay across steps
until served. Each step, exit `e` (capacity `capacity[e]`) serves
`min(queue, capacity[e] * mult)`, where `mult = 1` if `queue <= capacity[e]`,
else `mult = 1 / (1 + congestion_beta[e] * (queue/capacity[e] - 1))`.
Unserved people stay queued — an oversubscribed exit gets *permanently*
slower for as long as the backlog persists (a crush feeds itself).

You may only direct a zone to an exit `e` with `reachable[i][e] == 1` (the
default exit is always reachable). Maximize the total people served by
horizon `T`, over a fixed, seeded family of 10 instances.

## Candidate program contract

Standalone program: read ONE JSON public instance from **stdin**, write ONE
JSON answer to **stdout**. Runs in an isolated subprocess.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a guidance grid ...
print(json.dumps({"guidance": guidance}))
```

### Public instance (stdin)

```json
{
  "name": "ev04_shared_locompl_TRAP",
  "n_zones": 6, "n_exits": 3, "T": 8,
  "population": [92.1, 60.4, ...],        // per zone, float, length n_zones
  "capacity": [14.2, 20.5, 18.0],         // per exit, float, length n_exits
  "egress_cap": [22.0, 15.5, ...],        // per zone
  "base_compliance": [0.18, 0.24, ...],   // per zone, in [0,1]
  "default_exit": [0, 0, 1, 0, 2, 0],     // per zone, nearest exit index
  "credibility_decay": [0.61, 0.7, ...],  // per zone, in (0,1)
  "credibility_recover": [0.07, 0.09, ...], // per zone
  "congestion_beta": [3.1, 4.4, 5.0],     // per exit
  "reachable": [[1,0,1], [1,1,0], ...]    // n_zones x n_exits, 0/1
}
```

### Answer (stdout)

```json
{ "guidance": [ [g_00, g_01, ..., g_0,T-1], [g_10, ...], ... ] }
```

`guidance` must have exactly `n_zones` rows, each of length `T`, each entry an
integer exit index with `reachable[i][g] == 1`. Any shape/type/range/
reachability violation, a crash, a timeout, or non-JSON output scores that
instance `0.0`.

## Scoring (deterministic)

For each instance the evaluator computes, itself, by running the identical
simulator:

- `q_lb`   = `min(sum(population), sum(capacity) * T)` — a congestion-free,
  full-throughput ideal (in practice unreachable once compliance and crush
  losses bite),
- `q_base` = evacuated total under "broadcast every zone's own default exit,
  forever" (guided == default, so compliance/credibility never matter),
- `q_cand` = evacuated total achieved by **your** guidance grid,

```
r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, q_lb - q_base), 0, 1 )
```

Matching the naive "point at the nearest exit" baseline scores `~0.1`;
reaching the ideal scores `1.0`. **Doing worse than that naive baseline
scores below 0.1** — this happens whenever a policy assumes everyone will
comply and keeps re-optimizing directives as conditions drift, because each
contradiction taxes the very compliance the plan is counting on. **Ratio** is
the mean of `r`; **Vector** lists the per-instance scores.

## Suggested strategies

1. **Do nothing clever**: always broadcast the nearest exit.
2. **Reactive full-compliance balancing**: every step, greedily send the
   currently-departing cohort of each zone to whichever reachable exit looks
   least loaded, assuming full compliance.
3. **Forced/controllable decomposition**: only the compliant slice of a zone
   is actually steerable; route it into the exit with the most headroom left
   after the unavoidable default flows, and hold the choice fixed for the
   whole horizon.
4. **Local refinement**: perturb a decomposition-based assignment (still
   respecting reachability and horizon-long consistency) to squeeze out
   remaining crush losses on tight-capacity instances.
