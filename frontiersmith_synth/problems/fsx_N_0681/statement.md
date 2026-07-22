# The Gatekeeper's Ledger

You run the seller-approval desk of a marketplace. Applicants arrive in **T
rounds** of up to **B** applicants each. Every applicant carries a public
**application score** in `[0, 1]` (documents, off-platform reviews, ...).
For each applicant you must **approve or reject**. An approved applicant is
secretly either a genuinely **legitimate seller** (earns revenue) or a
**bad actor** (fraud, costs you). Rejected applicants contribute nothing.

Your program reads the whole instance, and outputs one approve/reject
decision for **every** applicant slot across the whole horizon (there is no
back-and-forth: you must commit to a full plan, but you know the exact public
formulas below so you can simulate the platform's own bookkeeping yourself).

## State that your decisions shape

Three state variables evolve round by round, using **only your decisions and
the true (hidden) outcomes of your approvals**:

- **Mix health `h`** (`h0` given) — the expected fraction of legitimate
  sellers *in the round currently arriving*. Applicant `i`'s TRUE legitimacy
  is a coin flip with `P(legit) = sigmoid((score_i - (1-h)) / temp)`: the
  same score is backed by a real business more often when `h` is high, and
  more often fronts a bad actor when `h` has decayed. This coin is **hidden**
  — you never observe it, only `score_i`.
- **Trust `r`** (`r0` given) — approved-legit revenue is `v_good * r`;
  approved-bad cost is `c_bad_eff * (1 + bad_amp*(1-r))` (fraud hurts more
  when trust is already damaged).
- **Volume multiplier** — the round's *active* applicant count is
  `round(B * vol_mult)` (only the first that many of the round's B slots are
  real; extra slots are ignored, decide them arbitrarily). If the round's
  approval rate falls below `vol_shrink_threshold`, `vol_mult` shrinks by
  `vol_shrink_step` next round (floor `vol_min_frac`); otherwise it grows by
  `vol_grow_step` (cap `1.0`) — over-rejecting shrinks future volume.

At the end of each round, given `bad_frac` = (bad approvals / active count)
and `approval_rate` = (approvals / active count):
```
h_recover = h_recover_param * (spiral_h_recover_mult if in_spiral else 1)
r_recover = r_recover_param * (spiral_r_recover_mult if in_spiral else 1)
h = clamp(h + h_recover*(1-h)*(1-bad_frac) - h_decay_bad*bad_frac, h_floor, h_ceiling)
r = clamp(r + r_recover*(1-r)*(1-bad_frac) - r_decay_bad*bad_frac,  r_floor, 1.0)
```
**The trap.** If `h` ever drops below `cascade_threshold`, the platform
enters a **permanent reputation spiral** for the rest of the run:
`c_bad_eff` is multiplied by `spiral_c_bad_mult`, and `h`/`r` recovery rates
are multiplied down by `spiral_h_recover_mult` / `spiral_r_recover_mult`.
Nothing in a single round's immediate math warns you this is imminent —
only looking a few rounds ahead does.

## Candidate program contract

Standalone program: read one JSON object from **stdin**, write one JSON
object to **stdout**.
```python
import sys, json
inst = json.load(sys.stdin)
# ... decide ...
print(json.dumps({"decisions": decisions}))
```
### Public instance (stdin)
```json
{"name": "desk1003", "T": 14, "B": 30, "N": 420,
 "score": [0.41, 0.77, ...],
 "h0": 0.55, "r0": 0.65,
 "params": {"h_decay_bad":0.5,"h_recover":0.05,"h_floor":0.05,"h_ceiling":1.0,
            "cascade_threshold":0.50,
            "r_decay_bad":0.3,"r_recover":0.05,"r_floor":0.15,
            "v_good":1.0,"c_bad":0.75,"bad_amp":0.5,"temp":0.16,
            "vol_shrink_threshold":0.3,"vol_shrink_step":0.1,
            "vol_grow_step":0.06,"vol_min_frac":0.45,
            "spiral_c_bad_mult":2.4,"spiral_h_recover_mult":0.15,
            "spiral_r_recover_mult":0.2}}
```
### Answer (stdout)
```json
{"decisions": [0, 1, 1, 0, ...]}   // length N, each 0 or 1
```
Any malformed output (wrong length, non-0/1 element, a crash, a timeout, or
non-JSON) scores that instance `0.0`.

## Objective and scoring

**Maximize** total net platform value (sum of realized revenue minus fraud
cost) across a fixed, seeded family of 10 instances (some calm, several
engineered so the mix health starts near the cascade threshold). The
evaluator replays your decisions against the true hidden outcomes and
normalizes:
```
r = clamp(0.1 + 0.9 * total_value / (T*B*v_good), 0, 1)
```
Rejecting everyone (`total_value = 0`) scores `0.1`; the denominator is a
loose, generally unreachable ceiling, so real headroom remains above any
reference solution. The reported **Ratio** is the mean of `r`; **Vector**
lists the per-instance scores.

## Suggested strategies

1. Reject everyone (baseline).
2. Per-applicant break-even rule using the current `h`/`r` state only.
3. Short-horizon rollout: project a few candidate acceptance bars forward,
   tighten preemptively when the projection nears `cascade_threshold`.
4. Full state-space dynamic programming for a provably optimal policy.
