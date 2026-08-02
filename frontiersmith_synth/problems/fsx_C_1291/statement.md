# Reinvest or Cash Out, Ten Thousand Times

You run a small operation with **capital** `K` that produces **output** every
turn: `output = K * base_rate * multiplier(tier)`. The multiplier is a **step
function of capital** — it starts at `1.0`, and each time cumulative capital
`K` crosses one of a fixed list of **thresholds** it permanently jumps to the
next, higher value. Crossing a threshold is an *unlock*: the new, higher
multiplier applies to every turn from then on, including turns where crossing
several thresholds at once jumps the multiplier by more than one step.

Each turn you decide how to split that turn's output between:
- **Reinvest** — added to `K`, so it compounds: it raises future turns'
  output, and can eventually push `K` across a threshold for a permanent
  multiplier jump.
- **Harvest** — banked as realized score. Banked money never grows further,
  but it is never discounted either.

The game runs for a **fixed, finite** number of turns `N`. When it ends,
banked cash counts at full value, but any capital still sitting in `K`
(never harvested) is only worth a discounted **salvage fraction** `s < 1` of
its face value — idle capital you never converted to real returns.

Always reinvesting (never harvesting) maximizes long-run compounding growth
and would be optimal if the game never ended — but on a *finite* horizon it
strands value in `K` at the discounted endgame. You must decide, turn by
turn, how much to harvest versus reinvest, using the full, exact schedule of
thresholds and multipliers, the current turn, and the turns remaining.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from
**stdin**, write ONE JSON object (your answer) to **stdout**. Runs in an
isolated subprocess.

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide a fraction to reinvest each turn ...
print(json.dumps({"invest": fractions}))
```

### Public instance (stdin)

```json
{
  "name": "forge_sprint",
  "n_turns": 18,             // N, number of turns (positive integer)
  "capital0": 10.0,          // K, starting capital
  "base_rate": 0.15,         // per-turn output rate at tier 0
  "thresholds": [40.0],      // increasing capital levels that unlock a jump
  "multipliers": [1.0, 4.0], // len == len(thresholds)+1, strictly increasing
  "salvage": 0.15            // s in (0,1): discount on capital left at the end
}
```

### Answer (stdout)

```json
{ "invest": [1.0, 1.0, 0.0, 0.0, ...] }   // length N, each in [0,1]
```

`invest[t]` is the fraction of turn `t`'s output you reinvest (the rest is
harvested that turn). Any invalid output (wrong length, a non-numeric or
out-of-[0,1] entry, non-finite value), a crash, a timeout, or non-JSON output
makes that instance score `0.0`.

## Simulation (this is exactly what the evaluator runs)

Starting `K = capital0`, `B = 0`, for each turn `t = 0..N-1`:
`tier = ` number of thresholds already `<= K`; `output = K * base_rate *
multipliers[tier]`; `invest = f_t * output`; `K += invest`; `B += output -
invest`. Final score = `B + salvage * K`.

## Objective

**Maximize** the final score, averaged over a fixed, seeded family of 10
instances (short aggressive-multiplier games, long gentle games, multi-
threshold ladders, and larger held-out cases).

## Scoring (deterministic)

For each instance the evaluator computes, itself, two references from the
*same* simulation:
- `q_base` = final score of the **never-reinvest** trajectory (`f_t=0`
  always) — a weak baseline with no compounding at all.
- `q_ideal` = the ending capital of the **always-reinvest** trajectory
  (`f_t=1` always) *with the salvage discount waived* — an unreachable ideal
  (provably an upper bound on any real, discounted trajectory).

```
r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, q_ideal - q_base), 0, 1 )
```

Matching the never-reinvest baseline scores ≈ `0.1`; reaching the (generally
unreachable) ideal scores `1.0`; doing worse than never-reinvesting scores
below `0.1`. The ideal is provably loose (it waives a real discount), so
even a strong policy stays below `1.0` — there is real headroom.

The reported **Ratio** is the mean of `r` over all instances; the **Vector**
lists the per-instance scores.

## Suggested strategies

1. **Never reinvest** (baseline): harvest everything, every turn.
2. **A single fixed reinvest fraction** for the whole game, tuned per
   instance.
3. **A single switch point**: reinvest fully for some prefix of turns, then
   harvest fully for the rest — search over where to switch.
4. **Multi-stage plans**: sequence several reinvest/harvest phases around
   more than one threshold, adapting to the exact thresholds and turns left.
