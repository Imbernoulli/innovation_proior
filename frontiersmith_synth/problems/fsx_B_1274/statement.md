# Warranty Reserves From a Mix-Shifting Claims Triangle

## Problem
An insurer's warranty book has `C` cohorts (launch batches), numbered
oldest to newest. Each unit in a cohort belongs to one of `P` hidden
product **types**. A type has its own **failure-hazard shape**: a
distribution over development age `d = 0..H-1` (time since the cohort's own
origin) describing *when* a unit that will eventually fail actually fails.
Whatever fails at age `d` is only **reported** (recorded as a claim) some
extra delay later, governed by one shared **reporting-lag** distribution
over `l = 0..Lmax`. So a claim from a unit that failed at age `d` is
reported at development age `d+l`.

Each cohort has an `EXPOSURE` (units in service) and a known `MIX` over the
`P` types (fractions summing to 1) — underwriting already knows what it
sold. Older cohorts have been observed for longer (more of the triangle is
filled in); recent cohorts have barely started. Crucially, **recent
cohorts tend to lean toward slower-developing types than older cohorts
did** — the book's mix has been drifting. You do not see the hazard
shapes, the lag distribution, or each type's ultimate value rate directly;
you only see cohort-level exposure, mix, and the partially-filled
cumulative-claims triangle.

## Input (stdin)
```
C P H Lmax A_full t
EXPOSURE_1 AGE_1 MIX_1_1 ... MIX_1_P K_1 R_1(0) ... R_1(K_1-1)
...
EXPOSURE_C AGE_C MIX_C_1 ... MIX_C_P K_C R_C(0) ... R_C(K_C-1)
```
`A_full = H+Lmax-1` is the development age at which everything that will
ever be reported has been reported. Row `c` (in cohort order, oldest
first): `EXPOSURE_c` (units), `AGE_c` (periods since origin, i.e. "now"),
`MIX_c` (`P` fractions summing to 1), then `K_c = min(AGE_c, A_full)+1`
cumulative reported-claims-value figures `R_c(0..K_c-1)` — `R_c(a)` is the
total dollar value of claims from cohort `c` reported by development age
`a`. `t` is the test id (informational only). All values are decimals.

## Output (stdout)
Exactly `C` non-negative numbers, whitespace-separated, in cohort order:
your estimate of each cohort's **reserve** — the dollar value still to be
reported for claims that already exist in the population but haven't
surfaced yet.

## Feasibility
Score `0.0` if: token count `!= C`; any token fails to parse as a finite
number; or any value is negative (a reserve can't be negative — money
already reported is money already reported).

## Objective
For each cohort `c`, let `U_c` be its true ultimate claims value (what
`R_c` would reach once fully developed) and `reserve_c = max(0, U_c -
R_c(K_c-1))`. Let `scale_c = max(reserve_c, 0.04*U_c)`. Your per-cohort
accuracy is `acc_c = max(0, 1 - |your_estimate_c - reserve_c| / scale_c)`.
The total objective `F` is the `EXPOSURE`-weighted mean of `acc_c` over all
cohorts. Higher is better; a cohort you already got exactly right (or
close, relative to its own scale) contributes close to `1`.

## Scoring
The checker also computes an internal baseline `B`: the same accuracy
formula evaluated at the naive "nothing more will be reported" guess
(reserve `= 0` for every cohort). Then:
```
sc = min(1000.0, 100.0 * F / B)
Ratio = sc / 1000.0
```
Matching the naive baseline scores `0.1`; meaningfully better reserving
scores higher, with headroom left above any reference solution.

## Constraints
`8 <= C <= 24`, `2 <= P <= 3`, `3 <= H <= 7`, `1 <= Lmax <= 3`,
`300 <= EXPOSURE_c <= 3000`. Time limit 5s, memory 512MB.

## Example
Two cohorts, one type each side of a shift, `A_full=6`. Cohort A is fully
developed: reported total `1000`, so `reserve_A = 0` — guessing `0` there
is free accuracy. Cohort B is brand new: reported so far is only `40`, but
because B's mix leans toward the slow-developing type, its true ultimate is
`900`, so `reserve_B = 860`. A naive guess of `0` for B gets `acc_B = 0`
(it is off by more than its own scale). A chain-ladder guess from *aggregate* factors dominated by fast-developing
older cohorts might project only `reserve_B ~= 150` — still badly short.
Recognizing that B's mix, not B's thin history, should set its ultimate
rate (read off older, fully-developed, similarly-typed cohorts) gets much
closer to `860`. (Illustrative numbers only — real instances differ.)
