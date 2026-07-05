# Terraced Vineyard: Drip-Emitter Placement

A large terraced vineyard is modelled as an `N x N` grid of vine plots. A long dry
spell has left every plot with an integer **moisture deficit** `w[r][c] >= 0` — how
badly that plot needs water. Sun-baked slopes and low hollows form clustered
**hot zones** of high deficit over a mild background.

The estate can install at most `K` **drip emitters**. An emitter dropped on plot
`(r, c)` wets every plot within **Chebyshev radius `R`** of it — the
`(2R+1) x (2R+1)` square block centred on `(r, c)`, clipped to the grid edges. A
plot wet by **at least one** emitter has its deficit fully relieved and contributes
its `w` to the **recovered deficit**; a plot no emitter reaches contributes nothing.
Wetting a plot twice helps no more than wetting it once.

Your job: write a heuristic that chooses where to drop the (up to) `K` emitters so
as to **recover as much total deficit as possible**.

This is weighted **maximum coverage** with square footprints (NP-hard, submodular):
marginal-gain greedy is a strong baseline, but relocation / swap local search does
better. The instance is fixed and given to you in full.

## Candidate program contract

Your solution is a **standalone program**: read ONE JSON object (the public
instance) from **stdin**, write ONE JSON object (your answer) to **stdout**. It runs
in an isolated subprocess and sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... choose emitter plots ...
print(json.dumps({"emitters": emitters}))
```

### Public instance (stdin)

```json
{
  "name": "vineyard101",
  "N": 30,                 // grid side length (N x N plots)
  "R": 2,                  // emitter Chebyshev wetting radius
  "K": 10,                 // maximum number of emitters you may place
  "grid": [[3, 1, 40, ...], ...]   // N rows x N cols of integer deficits w >= 0
}
```

### Answer (stdout)

```json
{ "emitters": [[r0, c0], [r1, c1], ...] }
```

- `emitters` is a list of **at most `K`** plots. Each element is a `[row, col]`
  pair of integers with `0 <= row < N` and `0 <= col < N`.
- Placing fewer than `K` emitters is allowed (but usually wasteful). Duplicate
  positions are allowed (but wasteful — they wet the same block).

Any invalid output (not a list, more than `K` emitters, a non-integer or
out-of-range coordinate, a malformed pair), a crash, a timeout, or non-JSON output
makes that instance score `0.0`.

## Objective

**Maximize** the recovered deficit (union of the wetted blocks, each plot counted
once) across a fixed, seeded family of **12 instances** that vary the grid size,
emitter radius `R`, budget `K`, and the number / sharpness of hot zones
(`fewhot`, `midhot`, `manyhot`). Several instances are larger / harder held-out
cases.

## Scoring (deterministic)

For each instance the evaluator computes, itself, from the full grid:

- `win(r,c)` = total deficit of the `(2R+1)^2` block centred at `(r,c)`;
- `UB`   = the sum of the `K` **largest** `win` values — a loose upper bound on the
  best achievable union (any `K` blocks' union `<=` their summed weight `<=` the
  `K` largest block weights);
- `weak` = the **union** deficit recovered by placing emitters on the `K` plots
  with the largest `win` (deterministic tie-break) — these pile onto the hottest
  zone and overlap heavily, a deliberately weak reference;
- `cand` = the **union** deficit recovered by **your** emitters.

and normalizes with an affine anchor (`weak -> 0.1`, `UB -> 1.0`):

```
r = clamp( 0.1 + 0.9 * (cand - weak) / max(1e-9, UB - weak), 0, 1 )
```

- Reproducing the top-block pile-up scores `~0.1`; spreading emitters to cover many
  hot zones scores higher.
- Because `UB` double-counts overlaps it is unreachable, so even good spreaders stay
  **well below `1.0`** — there is real headroom.

The reported **Ratio** is the mean of `r` over all instances; the **Vector** lists
the per-instance scores.

## Suggested strategies

1. **Top-block pile-up** (baseline): drop emitters on the `K` highest-`win` plots,
   ignoring overlap — reproduces the weak reference (`~0.1`).
2. **Marginal-gain greedy**: place emitters one at a time on the plot adding the
   most currently-uncovered deficit; spreads across hot zones.
3. **Relocation local search**: from the greedy layout, pull each emitter out and
   best-respond it to the plot adding the most uncovered deficit given the others;
   iterate under a fixed pass budget.
4. **Metaheuristics**: multi-emitter swaps or seeded perturb-and-reoptimize
   restarts to escape single-move local optima.
