# Rationed Sprays, Standing Guard

You manage integrated pest management for one field over a **T-day season**.
Each day the field holds a pest population **P** (crop-eating insects) and a
beneficial-predator population **Q** (their natural enemy), plus a hidden
resistance level that only ever increases. Before the season starts you must
commit, in one shot, a **full-season plan**: for every day `t` a pesticide
dose `spray[t]` in `[0,1]` and a predator release amount `release[t]` in
`[0, release_cap]`. There is no feedback during the season — write your plan
against the dynamics below (every constant you need is in the input), then
the evaluator replays it.

## Coupled dynamics

Every day the populations update (resistance `R` starts at `0.02` and is
never sent to you, but you can track it yourself since it evolves
deterministically from your own plan):
```
eff_t        = spray_eff0 * spray[t] * (1 - R_t)      # chemical kill on pests decays with resistance
pred_kill_t  = pred_kill0 * spray[t]                   # chemical kill on predators does NOT decay
P_{t+1} = max(0, P_t + r_pest*P_t*(1 - P_t/K_pest) - attack_rate*P_t*Q_t - eff_t*P_t)
Q_{t+1} = max(0, Q_t*(1 - pred_death) + convert_eff*attack_rate*P_t*Q_t - pred_kill_t*Q_t + release[t])
R_{t+1} = min(1, R_t + resist_gain*spray[t]*(1 - R_t))
```
`R` is a ratchet: it never decreases, and it only depends on your own past
`spray` values, so you can always reconstruct it from your own plan.
Pesticide resistance is pest-specific — it degrades `eff_t` but never
`pred_kill_t` — so repeated spraying gets weaker against the pest while
staying just as harmful to the predators that would otherwise help you.

## Objective and scoring

Daily cost is crop loss above a damage threshold plus intervention cost,
charged against the pest level **before** that day's action takes effect:
```
cost = sum_t [ loss_coeff*max(0, P_t - damage_thresh) + spray_cost*spray[t] + release_cost*release[t] ]
```
**Minimize** total season cost, summed (as a normalized ratio) across a
fixed, seeded family of 10 instances (six engineered so reflexive spraying
is self-defeating, four where chemical control is comparatively cheap and
safe). The evaluator replays your plan against the true dynamics and
normalizes:
```
base = cost of the "never spray, never release" plan on this instance   (computed by the evaluator)
r    = clamp(0.1 + 0.9 * (base - cost) / base, 0, 1)
```
Doing nothing exactly reproduces `base`, anchoring `r = 0.1`; `cost >= 0`
always, so `r` cannot reach `1.0` — headroom remains above any reference
solution. **Ratio** is the mean of `r` over the 10 instances; **Vector**
lists the per-instance scores.

**The trap**: reflexively spraying full dose whenever `P_t` exceeds
`damage_thresh` never releases predators and re-sprays every rebound. On the
instances where this field's chemistry punishes repeated dosing, this
grinds the predator population toward zero and resistance toward 1,
entering a cycle where damage recurs and each spray buys less and less.

## Candidate program contract
Standalone program: read one JSON object from stdin, write one JSON object
to stdout.
```python
import sys, json
inst = json.load(sys.stdin)
# ... decide the whole season's plan ...
print(json.dumps({"spray": spray, "release": release}))
```
### Public instance (stdin)
```json
{"name": "field-01", "T": 45, "P0": 400.0, "Q0": 15.0,
 "r_pest": 0.22, "K_pest": 1200.0, "attack_rate": 0.0012,
 "convert_eff": 0.35, "pred_death": 0.08, "resist_gain": 0.15,
 "spray_eff0": 0.75, "pred_kill0": 0.55, "loss_coeff": 2.0,
 "damage_thresh": 180.0, "spray_cost": 90.0, "release_cost": 1.2,
 "release_cap": 30.0}
```
### Answer (stdout)
```json
{"spray": [s_0, ..., s_{T-1}], "release": [u_0, ..., u_{T-1}]}
```
Both arrays must have length exactly `T`. Each `s_t` must be a finite number
in `[0,1]`; each `u_t` a finite number in `[0, release_cap]` (a tolerance of
`1e-9` absorbs harmless floating-point round-trip noise at the boundary;
anything beyond that is out-of-range). Any malformed output (wrong
length/type, a value outside its range by more than that tolerance,
non-finite value, a crash, a timeout, or non-JSON) scores that instance
`0.0`.

## Suggested strategies
1. Never spray, never release (baseline; reproduces the evaluator's own
   reference, `r = 0.1`).
2. Spray full dose whenever `P_t` exceeds `damage_thresh`, otherwise do
   nothing; never release predators.
3. Compute the critical predator density `q* = r_pest / attack_rate` (the
   level at which predation alone cancels pest growth at low `P`). Judge,
   from `spray_eff0`, `pred_kill0`, and `resist_gain`, whether a short
   calibrated opening spray is worth its resistance/predator cost on this
   field; then release predators up to `release_cap` until `Q` reaches
   `q*`, hold it there with light maintenance releases, and ration chemical
   use to genuine runaway emergencies — letting the coupled dynamics
   suppress the pest for free the rest of the season.
4. Full trajectory optimization over the plan space respecting the
   per-day bounds, using the known dynamics as an internal simulator to
   refine strategy 3 toward the season-minimal cost.
