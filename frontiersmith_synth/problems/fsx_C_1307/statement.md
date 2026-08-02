# Cull Schedule for a Delay-Coupled Vegetation-Prey-Predator Cascade

## Story
A reserve has three coupled populations, each tracked as a fraction of its own
carrying capacity: vegetation `V`, a herbivore `H` ("the prey" you are trying
to save), and a predator `Pr` that is currently overabundant and holding `H`
down. Wildlife managers may cull the predator each step; your program submits
the **entire cull schedule up front** (the whole horizon's parameters are
public), and it is replayed causally, step by step, against the fixed
recurrence below. Your goal: maximize prey persistence and abundance over the
horizon — but culling too hard, too fast, backfires.

## Dynamics (deterministic; identical formulas every instance, coefficients vary)
Given state `(V_t, H_t, Pr_t)` and your chosen cull fraction `c_t in [0, cull_max]`:
```
Vd = V[t - tauV]      (or V0 if t < tauV)     # vegetation felt by H's cohort tauV steps ago
Hd = H[t - tauH]      (or H0 if t < tauH)     # prey felt by Pr's numeric response tauH steps ago

graze     = gH * H_t * V_t
V_{t+1}   = V_t + rV * V_t * (1 - V_t) - graze

predation = aPred * Pr_t * H_t / (H_t + hHalf)
H_{t+1}   = H_t + rH * H_t * Vd * (1 - H_t) - mH * H_t * (1 - Vd) - predation
Pr_{t+1}  = Pr_t + rPr * Pr_t * Hd * (1 - Pr_t) - c_t * Pr_t
```
Grazing and predation act on **current** state (immediate ecological
interactions); reproduction/mortality of `H` and `Pr` act on **delayed** state
(`tauV`, `tauH` steps back — maturation/gestation lag: this generation's
births and starvation depend on conditions felt when it was conceived).

After each step, every value is clamped to `[0, 2]`, then **the population
floor applies**: if `V_{t+1} < floor.V` (resp. `H`, `Pr`) that population is
set to exactly `0` and stays `0` forever after (it is now locally extinct —
the recurrence is multiplicative in each species, so `0` is an absorbing
state). This is permanent and irreversible.

## Task
Standalone program, stdin JSON in -> stdout JSON out, isolated subprocess.

**Input** (all fields; nothing is hidden):
```json
{"T":45,"V0":0.85,"H0":0.20,"Pr0":0.55,
 "rV":0.28,"gH":0.80,"rH":0.55,"aPred":0.25,"hHalf":0.45,"mH":0.35,"rPr":0.35,
 "tauV":3,"tauH":4,"floor":{"V":0.05,"H":0.04,"Pr":0.03},"cull_max":0.90}
```
**Output:**
```json
{"cull":[0.10, 0.10, ..., 0.10]}
```
Exactly `T` numbers, each in `[0, cull_max]`.

## Replay & scoring (deterministic, no wall-time)
The evaluator replays your `cull` array against the recurrence above from
`(V0,H0,Pr0)`. A malformed answer (wrong length, non-numeric, non-finite,
out of `[0,cull_max]`, wrong JSON shape, a crash, or a timeout) scores the
whole instance `0`. Otherwise:
```
avgH        = mean over t=1..T of min(1, H_t)
persistence = (#{V,H,Pr} alive at t=T) / 3
score       = 0.7 * avgH + 0.3 * persistence        # already in [0,1]
```
Final score is the mean over 10 fixed instances.

## Why it is open-ended
Doing nothing leaves the predator dominant and prey chronically suppressed —
a mediocre, non-zero score. The obvious fix is a feedback controller: cull
hard when `H` is low, ease off once it recovers. This looks right, but the
system's delay structure punishes it: predator relief is felt by `H`
**immediately** (less predation), so `H` booms past what current vegetation
regrowth (`rV`) can support; the resulting overgrazing crash in `V` is only
felt by `H`'s own growth/mortality term `tauV` steps later — by which point
the controller has long since stopped culling (it saw `H` "doing fine") and
cannot prevent the delayed starvation crash through the floor. The longer
`tauV` is, the further the correction and its consequence are separated in
time, and the worse a reactive policy performs. The genuine insight is to
never chase the trajectory at all: commit to a small, constant, sustained
cull rate, calibrated down as `tauV` grows, that relieves predation gently
enough that vegetation's own regrowth keeps pace — trading a slower recovery
for one that never triggers the delayed cascade.

## Isolation
Your program runs in a fresh sandboxed subprocess and only ever sees the
public instance above; all replay and grading happen in the evaluator
process.
