# SOC Sensor Fusion: Budgeting the 1% Alert Allowance

A security operations center watches `K` independent telemetry **sensors**
(netflow entropy, auth timing, DNS query rate, ...). Every event -- ordinary
benign traffic, or an attack drawn from one of several **families** -- produces
one real-valued reading per sensor. You publish, once, for the whole SOC:

- a per-sensor alert **threshold** `theta_c`,
- a per-sensor fusion **weight** `w_c >= 0`,
- a fusion vote **threshold** `tau`.

An event is **flagged** iff `sum_c w_c * [reading_c > theta_c] >= tau`. A fixed
SLA caps the **false-positive rate** on a fixed benign event set at `fp_cap`
(1%): if your layout flags more than that fraction of benign events, the
entire instance scores **0**.

Each attack family is drawn at several **stealth levels**, from overt
(level 0, loud on its fingerprint sensor) to near-benign (the highest level,
barely above background noise). Every family has its own fingerprint sensor
where its signal survives deepest into stealth; on every other sensor an
attack from that family looks exactly like background noise. The 1%
false-positive allowance is **one shared, divisible resource** across all `K`
sensors -- making one sensor more sensitive to catch a stealthy family spends
part of it; making another sensor stricter frees part of it back. Some
families are intrinsically easier to separate from noise than others, so an
equal split of the allowance across sensors is rarely the best split.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**,
write ONE JSON object (your answer) to **stdout**. Runs isolated; sees only
the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute theta, w, tau ...
print(json.dumps({"theta": theta, "w": w, "tau": tau}))
```

### Public instance (stdin)

```json
{
  "channels": 6,                 // K, number of sensors
  "fp_cap": 0.01,                 // shared false-positive budget
  "n_benign": 2000,               // size of the fixed benign event set
  "benign": [[..K floats..], ...],// n_benign x K matrix of sensor readings
  "families": 6,                  // F, number of attack families
  "levels": 5,                    // L, stealth levels 0 (overt) .. L-1 (stealthiest)
  "family_names": ["c2-beacon", ...],
  "channel_names": ["netflow-entropy", ...],
  "attacks": [ /* [F][L] */ [ /* L */ [ /* 120 x K matrix per level */ ] ] ]
}
```
`attacks[f][l]` is a list of sensor-reading vectors (each length `K`) for
family `f`'s attack samples at stealth level `l`.

### Answer (stdout)

```json
{ "theta": [.. K floats ..], "w": [.. K floats >= 0 ..], "tau": 1.0 }
```
Any wrong length, non-finite value, negative weight, crash, timeout, or
non-JSON output scores that instance `0.0`.

## Objective

**Maximize**, per instance: for each family `f`, its stealth-weighted
detection quality
```
q_f = sum_{l=0}^{L-1} (l+1) * det(f, l)  /  sum_{l=0}^{L-1} (l+1)
```
where `det(f, l)` is the fraction of family `f`'s level-`l` attack samples your
layout flags (deeper stealth levels count for more). The instance's raw
quality is `q = min_f q_f` -- the **worst-served family**, so a layout that
looks good on average while abandoning one family scores poorly. This is
computed over a fixed, seeded family of 10 instances varying in sensor count,
family count (some with more families than sensors, forcing two families to
share a fingerprint sensor; some with idle sensors nobody needs), and each
family's intrinsic separability. Several instances are larger, held-out cases.

## Scoring (deterministic)

If your false-positive rate exceeds `fp_cap`, the instance scores `0`.
Otherwise the evaluator computes, itself, two references: `q_base` (using only
sensor 0, at the full private 1% allowance) and `q_ub` (giving **every**
sensor its own full private 1% allowance simultaneously, ignoring that the
allowance must be shared -- an unreachable ceiling):
```
r = clamp( 0.1 + 0.9 * (q - q_base) / max(1e-9, q_ub - q_base), 0, 1 )
```
Matching the weak single-sensor baseline scores `~0.1`; real headroom remains
below the unreachable ceiling. **Ratio** is the mean of `r` over all 10
instances; **Vector** lists the per-instance scores.

## Suggested strategies

1. **Single fixed sensor** (baseline): threshold sensor 0 alone at the full 1%.
2. **Equal-share global operating point**: split the 1% evenly across every
   sensor, OR-fuse them (`w=1`, `tau=1`) -- the obvious "maximize average
   detection" recipe.
3. **Per-family private-budget probe**: for each family, find which sensor is
   its fingerprint and how much allowance it alone would need.
4. **Max-min budget reallocation**: starting from the equal split, repeatedly
   move slices of the shared 1% from a sensor serving an already-comfortable
   family to the sensor serving the currently worst-off family, verifying the
   exact fused false-positive rate after every transfer.
