# Moonshine Still: Choose Your Cuts

## Problem
A feedstock mixes `K` chemical components. Each component's mass is spread over `G`
boiling-point *bins* (`0..G-1`), given as a per-component, per-bin mass table. These
distributions **overlap**: several components can have real mass in the same bin, and
one component can even re-condense in more than one place along the range, so a bin's
identity is a blend, not a pure label.

You run the feedstock through **exactly two distillation passes**. In **pass 1** you
choose cut points partitioning `[0,G)` into contiguous fractions; each cut point must
be a multiple of `S` (a coarse valve granularity given in the input). Each pass-1
fraction gets one action:
- `S` (sell) — cash it in now,
- `D` (dump) — discard it (no cost, no revenue),
- `R` (recycle) — forward its exact per-component, per-bin mass into pass 2's input,
  paying an energy cost proportional to the mass moved.

Masses of every `R` fraction are **summed bin-by-bin** (non-recycled bins contribute
zero) to form pass 2's input over the same `G` bins. In **pass 2** you again choose cut
points — any bin boundary, no coarse restriction — and each resulting fraction gets `S`
or `D` (no third pass exists, so `R` is invalid there).

**Pricing.** For a sold fraction over bins `[a,b)`: `mass_c` = its mass of component
`c`, `total = sum_c mass_c`, `dominant` = the component with largest `mass_c`,
`purity = mass_dominant / total`. Look up the price-band table (input, ascending purity
thresholds) for the multiplier of the highest band whose threshold is `<= purity`.
**Batch-size qualifier:** if `total < M_min`, the multiplier is capped at `cap_small`
*regardless of purity* — a small lot never clears certification for the good bands,
however pure; only a properly-sized batch does. Revenue = `total * value[dominant] *
multiplier`. Selling ANY fraction (either pass) costs a fixed handling fee `H`.
Recycling a pass-1 fraction costs `energyCost * mass` of that fraction.

## Input (stdin)
```
K G S
v_1 ... v_K                      (value per unit mass at 100% purity, per component)
m_1[0] ... m_1[G-1]              (component 1's mass in each bin)
...
m_K[0] ... m_K[G-1]
H energyCost
M_min cap_small
B
lo_1 mult_1
...
lo_B mult_B
```

## Output (stdout)
```
c1 x_1 ... x_c1                  (pass-1 cuts, strictly increasing, each in [1,G-1], multiple of S)
a_1 ... a_{c1+1}                 (pass-1 actions, each S/D/R, one per fraction)
[c2 y_1 ... y_c2]                (pass-2 cuts -- ONLY if some pass-1 action was R)
[b_1 ... b_{c2+1}]                (pass-2 actions, each S/D)
```
Omit the pass-2 lines entirely if no pass-1 fraction was recycled.

## Feasibility
- Pass-1 cuts: strictly increasing integers in `[1,G-1]`, each a multiple of `S`.
- Pass-2 cuts: strictly increasing integers in `[1,G-1]` (no multiple-of-S rule).
- Each action list's length must equal (cut count + 1); pass-2 actions may not be `R`.
- No trailing tokens beyond what the format requires.
Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F` = (sum of sold-fraction revenue, both passes) − `H` × (number of sold
fractions, both passes) − `energyCost` × (total mass recycled in pass 1).

## Scoring
Let `B_ref` be the checker's trivial construction: cut the feedstock into `K`
equal-width bin slices (snapped to multiples of `S`), sell all `K` in pass 1, no
recycling; `B_ref` is that construction's `F` (always positive). Then
```
sc = min(1000.0, 100.0 * F / max(1e-9, B_ref))
Ratio = max(0.0, sc) / 1000.0
```
Reproducing the baseline scores `Ratio ~= 0.1`; `10x` better caps at `1.0`.

## Constraints
- `3 <= K <= 5`, `12 <= G <= 50`, `S >= 1`.
- Time limit 5s, memory 512m.

## Example
Suppose component 3 shows up as several small, separate pockets scattered across the
bin range, each fairly pure but under `M_min` in mass. Selling each where it sits earns
only `cap_small`, however pure it looks. Marking all of them `R` (they need not be
adjacent — the recycle mask is just "which bins were marked `R`") pools their mass into
one pass-2 stream; if the pool clears `M_min` and stays dominated by component 3, pass
2 can re-cut it into one properly-sized, high-purity lot at a far better price band.
Whether this beats leaving each pocket alone, and which components are worth pooling,
depends on the mass table and cost constants in the input; the checker computes `F`
exactly as defined above from your submitted cuts and actions.
