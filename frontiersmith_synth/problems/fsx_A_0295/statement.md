# Reservoir Spectral-Load Constant

## Problem

A flood-control **reservoir dam network** is a chain of `n` reservoirs built
along one river, indexed `0 .. n-1`. You choose a **storage-capacity profile**
`f = (f_0, f_1, ..., f_{n-1})` with each `f_i >= 0`: the buffering capacity you
install at reservoir `i`.

During a storm, two independent inflow surges enter the chain at reservoirs `i`
and `j`. Their released water re-converges downstream at the **confluence index**
`k = i + j`, and the hydraulic stress deposited there is proportional to the
product `f_i * f_j` of the two capacities. Summing over every surge pair that
converges at the same place gives the **stress spectrum** — the autoconvolution
of the profile with itself:

```
c_k = sum over all i,j with i+j=k of  f_i * f_j        (k = 0 .. 2n-2)
```

A network that concentrates its stress into a few sharp confluence peaks is
brittle. A robust design spreads the stress spectrum as *flat and low* as
possible. We measure brittleness by the **total spectral energy** (the squared
L2 norm) of the stress spectrum, made scale-invariant:

```
L(f) = n * ( sum_k c_k^2 )  /  ( sum_i f_i )^4
```

Multiplying `f` by any positive constant leaves `L(f)` unchanged, so only the
*shape* of the profile matters. **Minimize `L(f)`.**

This is a step-function instance of the L2 autoconvolution inequality. Unlike
the peak (max) version, the L2 energy rewards a genuinely spread-out spectrum:
cramming capacity into one block is terrible, the flat/uniform profile is only
mediocre, and driving the energy down to its (unknown) floor requires a
carefully shaped, tapered profile. There is a hard positive lower bound and no
known closed-form optimum, so the problem stays genuinely open-ended.

## Input (stdin)

A single integer `n` (`12 <= n <= 240`): the number of reservoirs in the chain.

## Output (stdout)

`n` real numbers separated by whitespace: the capacities `f_0 .. f_{n-1}`.

## Feasibility

- Exactly `n` values.
- Each value finite (no `nan`/`inf`) with `0 <= f_i <= 1e6`.
- `sum_i f_i > 0`.

Any violation scores `Ratio: 0.0`.

## Objective

Minimize the normalized spectral-load index `L(f)` defined above.

## Scoring

The checker computes your `F = L(f)` and an internal baseline `B = L` of the
**half-block** construction it builds itself (unit capacity in the first
`floor(n/2)` reservoirs, zero elsewhere — all the buffering crammed into one
contiguous half of the river). With a minimization normalization:

```
sc    = min(1000, 100 * B / F)
Ratio = sc / 1000
```

Reproducing the baseline scores `~0.1`; a 10x lower index caps the ratio at
`1.0`. Scoring is exact and deterministic.

## Constraints

- `12 <= n <= 240`.
- Exact `O(n^2)` autoconvolution; runs well within the time limit.

## Example

For `n = 12`:

- **Half-block** `f = (1,1,1,1,1,1,0,0,0,0,0,0)`: the stress spectrum is a
  triangle peaking at `c_10 = 6`; `sum_k c_k^2 = 146`, `sum f = 6`, so
  `L = 12 * 146 / 6^4 = 1.3519` -> `Ratio = 0.1` (this is the baseline).
- **Uniform** `f = (1,...,1)`: `L = 0.6712` -> `Ratio ~= 0.20` (a flat spectrum
  already halves the energy).
- A **shaped, tapered** profile can reach `L ~= 0.60`, giving `Ratio ~= 0.23`.
