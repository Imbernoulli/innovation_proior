# Waste-Fed Row: Cross-Feeding Biofilm Layout

A biofilm grows along a 1-D row of `L` cells. A single **primary nutrient**
(metabolite type `0`) diffuses in from the two open edges of the row, where its
concentration is continuously held at `boundary_conc`. You are given a small
catalogue of candidate **species**; each consumes one metabolite type and, as a
side effect of growth, may excrete a **waste metabolite** (type `1`) that a
*different* species could use as its own food. Your job: choose which species (if
any) to inoculate at every cell so that the row's **total biomass** after a fixed
number of growth steps is as large as possible.

## Dynamics (run by the evaluator, exactly as follows)

Two metabolite fields live on the row: `conc[0]` (primary nutrient) and `conc[1]`
(waste). Both start at `0` everywhere except `conc[0][0] = conc[0][L-1] =
boundary_conc`. For each of `T` steps, in order:

1. **Diffusion.** For each metabolite `k` and cell `i`: `new[k][i] = conc[k][i] +
   diffusion[k] * (left + right - 2*conc[k][i])`, where `left`/`right` are the
   neighboring cells (a missing neighbor at the row's ends is replaced by the edge
   cell itself -- no-flux boundary). Then `conc[0][0]` and `conc[0][L-1]` are reset
   to `boundary_conc` (continuous inflow at the open edges).
2. **Reaction.** At every cell `i` inoculated with species `s` (your `assign[i]`,
   skip if `-1`): let `c = conc[s.consumes][i]`, `monod = c / (s.Km + c)`. If
   `s.produces != -1`, let `inhib = s.Ki / (s.Ki + conc[s.produces][i])`
   (self-inhibition by the species' OWN waste piling up locally); otherwise
   `inhib = 1`. Uptake `consumed = min(s.vmax * monod * inhib, c)`. Then
   `conc[s.consumes][i] -= consumed`; `biomass[i] += consumed * s.yield_biomass`;
   if `s.produces != -1`, `conc[s.produces][i] += consumed * s.yield_byproduct`.

Total yield = `sum(biomass)` over all cells after `T` steps. **Maximize** it.

The trap: the species with the single highest `vmax` looks best, but a species
that produces waste (`produces != -1`) and has a **small** `Ki` gets throttled by
its own accumulating waste if nothing nearby consumes that waste type. A different
species placed close enough (waste diffuses slowly -- distance matters) to eat that
waste relieves the inhibition and grows itself, so an interleaved layout can beat
any single-species monoculture. Read the numbers; do not assume any position or
ordering is meaningful beyond the fields given.

## Candidate program contract

Standalone program: read ONE JSON object from **stdin**, write ONE JSON object to
**stdout**. Runs isolated, sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute a layout ...
print(json.dumps({"assign": assign}))
```

### Public instance (stdin)

```json
{
  "name": "strip101", "L": 18, "T": 28, "boundary_conc": 5.0,
  "diffusion": [0.40, 0.15],
  "species": [
    {"consumes": 0, "produces": -1, "vmax": 0.18, "Km": 2.0, "Ki": 1e9,
     "yield_biomass": 0.30, "yield_byproduct": 0.0},
    {"consumes": 0, "produces": 1, "vmax": 1.35, "Km": 1.05, "Ki": 1.6,
     "yield_biomass": 0.44, "yield_byproduct": 0.52},
    ...
  ]
}
```

### Answer (stdout)

```json
{ "assign": [1, 2, 1, 2, 0, ...] }   // length L; each entry -1 or a species index
```

Any invalid output (wrong length, an out-of-range or non-integer entry), a crash,
a timeout, or non-JSON output makes that instance score `0.0`.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `obj_base` = total biomass from a fixed reference layout (species index `0`
  inoculated in every cell),
- `obj_ceil` = `(2 * diffusion[0] * boundary_conc * T) * best_chain_yield`, where
  `best_chain_yield` is the best achievable biomass-per-unit-of-consumed-primary-
  nutrient over any simple metabolite hand-off chain among the given species
  (computed from the species table alone, ignoring transport limits -- an
  idealized, generally unreachable reference),
- `obj_cand` = total biomass from **your** simulated layout,

and normalizes:

```
r = clamp( 0.1 + 0.9 * (obj_cand - obj_base) / max(1e-9, obj_ceil - obj_base), 0, 1 )
```

Matching the fixed reference monoculture scores ~`0.1`; the idealized ceiling
assumes unlimited transport, so it stays out of reach and real layouts land below
`1.0`. The reported **Ratio** is the mean of `r` over 10 seeded instances (some
larger / harder held-out); **Vector** lists the per-instance scores.

## Suggested strategies

1. **Fixed monoculture** (reference): ignore the instance, always use species 0.
2. **Fastest consumer**: fill the row with `argmax(vmax)` -- tempting, but
   self-poisoning if that species also produces waste with a small `Ki`.
3. **Avoid inhibition**: fill with a slower species that produces no waste.
4. **Cross-feeding layout**: interleave a waste-producing species with the
   waste-consuming species that best exploits it (compare `vmax * yield_biomass`
   across all species whose `consumes` matches the producer's `produces`), placed
   close enough for slow byproduct diffusion to reach.
