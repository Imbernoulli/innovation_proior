# Carving a Radio Filter from Integer Gears

## Problem
A circulant radio filter of length `N` is a ring of `N` integer "gear teeth"
`h_0, ..., h_{N-1}` (each a signed integer) applied cyclically to a signal. Its
discrete frequency response at bin `k = 0, ..., N-1` is

```
H(k) = sum_{j=0}^{N-1} h_j * exp(-2*pi*i*j*k / N)
```

Because every `h_j` is real, `|H(k)| = |H(N-k mod N)|`, so only the
**canonical** bins `k = 0, 1, ..., floor(N/2)` need to be specified and scored.

For every canonical bin `k` you are told one of three roles:
- **PASS(k, T)** — the design wants `|H(k)|` close to a positive target gain `T`.
- **STOP(k)** — the design wants `|H(k)|` close to `0`.
- **FREE(k)** — a don't-care bin: nothing is asked of `|H(k)|` here, and it is
  never scored.

The PASS bins recur periodically around the ring (a "comb" of teeth in the
spectrum, giving the family its name); FREE bins sit at the transition around
each tooth.

You must output the `N` integer gear-teeth values, subject to a total material
budget: `sum_j |h_j| <= B`.

## Input (stdin)
```
line 1:             N B M        (M = floor(N/2) + 1)
next M lines:        k type T    (type in {P, S, D}; T is meaningful only for P)
```

## Output (stdout)
`N` whitespace-separated integers `h_0 ... h_{N-1}`.

## Feasibility
The output must contain exactly `N` tokens, each parsing as a base-10 integer
with `|h_j| <= 10^9`, and `sum_j |h_j| <= B`. Anything else scores `0`.

## Objective (minimise)
```
Dev = max over PASS/STOP bins k of | |H(k)| - T_k |     (T_k = 0 for STOP)
```
FREE bins never enter `Dev`.

## Scoring
The checker rejects any infeasible output. Otherwise let `B_base` be the `Dev`
of the do-nothing stencil (`h_j = 0` for all `j`) — i.e. `B_base` is the
largest PASS target. Then

```
Ratio = min(1000, 100 * B_base / max(1e-9, Dev)) / 1000
```

The do-nothing stencil reproduces `Ratio ~= 0.1`. No stencil within the given
budget reaches `Dev = 0` (the PASS targets are irrational-looking decimals, not
exactly reachable by any small integer combination), so the score never
saturates.

## Why rounding is the whole game
`B` is generous enough that the *exact* real-valued filter matching every
PASS/STOP bin (with FREE bins pinned to zero) already fits the budget — so
there is no need to shrink the design's amplitude. The only thing standing
between that real-valued design and a feasible answer is rounding every
coefficient to an integer, and that rounding is not free: it perturbs `H(k)`
at *every* bin, scored or not. Rounding each coefficient independently treats
that disturbance as pointwise noise with nowhere in particular to go. But a
bin's frequency-domain value never depends on any other bin's value, so the
FREE bins are spare degrees of freedom you can move by any amount without
ever touching a PASS or STOP response — including using them to *choose*
where the unavoidable rounding disturbance lands, instead of letting it fall
wherever pointwise rounding happens to put it.

## Constraints
- `N` between `17` and `97`; `B` a positive integer, always large enough to
  afford the exact real-valued PASS/STOP match.
- `T` given to 6 decimal places.
- The PASS teeth recur with a period between `4` and `9` canonical bins
  (implicit in the bin roles, not stated separately).

## Example (worked, illustrative form only — not one of the real tests)
`N=8`, canonical bins `k=0..4`: `k=0` PASS `T=4.000000`, `k=1` FREE, `k=2` STOP,
`k=3` FREE, `k=4` PASS `T=4.000000`, budget `B=6`. The all-zero stencil gives
`Dev=4` and `Ratio ~= 0.1`. Any stencil that gets both PASS bins near `4` and
`k=2` near `0` while letting `k=1, k=3` land wherever they may will beat the
baseline; a stencil that additionally *aims* the leftover rounding residue at
`k=1` and `k=3` — instead of at `k=0`, `k=2`, `k=4` — does better still, for the
same budget.
