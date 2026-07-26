# The Unseen Tide: Cargo Admission Policy

A freight exchange loads cargo lots onto a single barge of fixed **capacity `C`**.
Lots arrive one at a time; each admission decision is **irrevocable** — accept or
reject the instant a lot is offered, no undo, no reordering. Before today's tide,
you see a **preview manifest**: a historical arrival log drawn from the same
process as today's lots (same size range, same value-density behavior, same rough
sense of how busy things get) — but it is **not** today's actual stream. Today's
real arrivals are held back entirely. You must commit to an **admission policy**
before the first real lot shows up; that policy is then applied automatically,
lot by lot, to the unseen real stream. You never see the sequence you're scored
on, only a rehearsal of it — and how busy today turns out to be relative to the
preview is not guaranteed.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**,
write ONE JSON object (your policy) to **stdout**. Runs in an isolated subprocess.

```python
import sys, json
inst = json.load(sys.stdin)
# ... inspect inst["preview"] ...
print(json.dumps({"policy": {"base": ..., "cap_gain": ..., "drift_gain": ..., "time_gain": ...}}))
```

### Public instance (stdin)

```json
{
  "name": "tide03",
  "capacity": 588,
  "preview_n": 70,
  "preview": [[size_0, value_0], [size_1, value_1], ...]
}
```
`preview` lists `preview_n` integer `[size, value]` pairs (arrival order of the
historical log), `1 <= size <= capacity`, `value >= 1`.

### Answer (stdout) — you output a POLICY, not decisions

```json
{"policy": {"base": 9.5, "cap_gain": 4.2, "drift_gain": 0.1, "time_gain": -1.8}}
```
Four **finite** real numbers. You never see today's real lots — the evaluator
replays your four coefficients against them for you.

## How your policy is replayed (exact, deterministic)

Today's true stream (length `N`, usually a *different* length and total volume
than the preview) is processed lot by lot, `i = 0..N-1`, tracking remaining
capacity `rem` (starts at `C`) and the mean density of lots seen so far:

```
density_i   = value_i / size_i
time_frac   = i / (N - 1)                        # 0 at the first lot, 1 at the last
cap_used    = 1 - rem / C                         # fraction of capacity spent so far
running_avg = mean(density_0 .. density_{i-1})    # 0 before the first lot
threshold_i = base + cap_gain * cap_used**2 + drift_gain * running_avg + time_gain * time_frac
```
**Admit** lot `i` iff `density_i >= threshold_i` **and** `size_i <= rem`; if
admitted, `rem -= size_i` and the lot's value is banked. A missing key,
non-numeric value, or non-finite number (`NaN`/`inf`) in your policy makes that
instance score `0.0`.

## Objective

**Maximize** total banked value across a fixed, seeded family of 10 instances
(varying value-density behavior — flat, drifting up, drifting down, a late
value spike — and varying how today's true volume compares to what the preview
implied, several notably busier). Some instances are larger / held-out.

## Scoring (deterministic)

For each instance the evaluator computes, on the TRUE (hidden) stream only:

- `v_base` = value from admitting **every** lot that still fits (no filtering),
- `v_hi`   = the fractional-knapsack upper bound — sort lots by density
  descending, fill `C`, allowing the boundary lot fractionally. `v_hi` bounds
  *any* feasible selection, causal or not, so no policy can fully reach it,
- `v_cand` = value from replaying **your** policy,

normalized with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (v_cand - v_base) / max(1e-9, v_hi - v_base), 0, 1 )
```

Matching the "admit everything" baseline scores ≈`0.1`; reaching the
(generally unreachable) fractional ideal scores `1.0`. The reported **Ratio**
is the mean of `r`; the **Vector** lists per-instance scores.

## Suggested strategies

1. **Admit everything** (baseline): no filtering, wastes capacity on weak lots.
2. **Static density cutoff**: pick one threshold from the preview (e.g. its
   mean density) and hold it fixed — blind to remaining capacity and to how
   today's stream departs from the preview.
3. **Dual-fitting shadow price**: anchor the threshold to the preview's
   fractional-cutoff density instead of the mean, and let `cap_gain` raise the
   bar as capacity is actually consumed — a busier-than-expected tide meets
   rising selectivity instead of early overcommitment.
4. **Drift-corrected pacing**: also use `time_gain`/`drift_gain` so the policy
   reacts to the *direction* value is moving and to the observed running
   density, instead of assuming today looks exactly like the preview.
