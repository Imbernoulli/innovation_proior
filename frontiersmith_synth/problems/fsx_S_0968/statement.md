# The Hollow-Span: Ballast Against the Canyon Choir

A rope bridge is strung across a haunted canyon, deck planks `0..N-1` hanging
between two fixed stone anchors, joined to neighbours (and the nearest anchor)
by rope segments that behave as springs. Every night the canyon "sings": a
fixed, KNOWN spectrum of ghostly gust tones blows through, each a steady
frequency with its own spatial push-shape across the deck. Before a crossing
you may bolt integer sacks of ballast onto the planks (total mass capped by a
budget), then choose how heavy a cargo to carry across the resting stone at
the deck's centre. A crossing at a given storm intensity is **safe** only if
the resting stone's peak sway stays under a fixed threshold; an unsafe
crossing loses the cargo entirely.

Plank `i` carries mass `base_mass[i] + ballast[i]`, plus `cargo` added only at
the resting-stone node (cargo's own weight detunes the bridge while carried).
Mass, with the fixed spring stiffnesses, determines the structure's vibration
**modes**: frequencies `omega_k` and shapes `phi_k`. A gust tone `(omega_g,
shape, amp)` excites mode `k` via the **resonance overlap integral** `p_k =
phi_k . shape` (how much the gust's push matches that mode's shape); its
contribution to peak sway is `amp * |p_k * phi_k(node)| / sqrt((omega_k^2 -
omega_g^2)^2 + (2*zeta*omega_k*omega_g)^2)` -- large when the mode sits close
to the gust tone, damped by `zeta`. Modes and tones combine by summing
absolute contributions (a standard, conservative worst-case rule). A storm
night scales every tone's amplitude by a fixed multiplier; the cargo scores on
a storm only if the scaled peak clears the threshold. The objective is `cargo
* (fraction of seeded storm nights that are safe)`, averaged over 10 bridges.

**The trap.** Bolting ballast onto whichever plank swings hardest looks
obviously correct -- it dampens that one mode. But mass added anywhere pulls
*every* mode's frequency down together. On several of the 10 bridges, doing
exactly this drags a second mode -- safely clear at baseline -- straight into
a second gust tone, making things worse. Planks and springs are randomized,
but always shaped so this danger is real: the design variable is the
structure's whole spectrum, not any one plank's raw sway.

## Candidate program contract

Standalone program: read ONE JSON public instance from stdin, write ONE JSON
answer to stdout. Runs isolated; sees only the public instance.

### Public instance (stdin)

```json
{
  "name": "span04", "n_nodes": 7,
  "base_mass": [14.0, 9.0, ...],           // N floats, per-plank base mass
  "stiffness": [31.2, 28.4, ...],          // N+1 floats, spring constants
  "ballast_budget": 30, "cargo_cap": 38,
  "cargo_node": 3,                          // resting-stone plank index
  "damping_zeta": 0.032,
  "storm_scales": [0.55, 0.75, 0.95, 1.15, 1.4, 1.65],
  "amp_threshold": 0.343,
  "gust_components": [
    {"omega": 5.1, "shape": [0.2, 0.5, ...], "amp": 1.0},
    {"omega": 8.7, "shape": [0.1, 0.4, ...], "amp": 0.85}
  ]
}
```

### Answer (stdout)

```json
{"ballast": [b_0, ..., b_{N-1}], "cargo": c}
```

`ballast[i] >= 0` integers with `sum(ballast) <= ballast_budget`; `c` a single
integer with `0 <= c <= cargo_cap`. Wrong shape, non-integers, negatives,
over-budget ballast, an out-of-range cargo, a crash, a timeout, or non-JSON
output makes that instance score `0.0`.

## Scoring (deterministic)

For each instance the evaluator computes, itself, two references: `q_base`
(best-reachable objective using the naive "biggest swinger" placement above)
and `q_ub` (the best of a small shortlist of placements: the naive one, a
two-way split, a uniform spread, and a bounded local search on the peak).
Your objective `obj` is normalized:

```
r = clamp(0.1 + 0.82 * (obj - q_base) / max(1e-9, q_ub - q_base), 0, 1)
```

Matching the naive placement scores ~0.1; matching the shortlist reference
scores up to 0.92, never 1.0 -- `q_ub` is a bounded shortlist, not a proven
optimum, so genuine spectral reasoning against the actual gust spectrum can
match or beat it. The reported **Ratio** is the mean of `r` over all 10
instances; **Vector** lists the per-instance scores.

## Suggested strategies

1. **Spread thin**: divide the budget evenly across planks, scan cargo.
2. **Biggest swinger**: modal-analyze the baseline, dump all ballast on the
   most-excited plank, scan cargo -- the obvious first move.
3. **Spectral exchange search**: build your own mass/stiffness model, solve
   the true modes, and locally search ballast layouts (moving units between
   planks) that minimize the resonance overlap with the ACTUAL gust spectrum.
4. **Joint refinement**: alternate tuning ballast for a given cargo and
   re-picking cargo for the tuned ballast, since cargo itself joins the mass
   matrix while it's being carried.
