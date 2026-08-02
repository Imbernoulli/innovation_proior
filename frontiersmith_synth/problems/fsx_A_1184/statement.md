# Column Tailing: Untangling a Chromatogram into True Compounds

## Problem
A lab has a **reference retention-index table**: `M` candidate compound slots at fixed,
known nominal retention times `r_1 < r_2 < ... < r_M` (an evenly spaced grid). In any
single chromatography run, some *unknown* subset of these slots is truly present, each
with an *unknown* positive area (abundance). Every peak observed in this run travels
down the **same physical column**, so every true peak has the **same tailing shape** — it
is not a symmetric bump but an **exponentially-modified Gaussian (EMG)**:
```
EMG(t; mu, sigma, tau) = 1/(2*tau) * exp( sigma^2/(2*tau^2) - (t-mu)/tau )
                          * erfc( sigma/(tau*sqrt(2)) - (t-mu)/(sigma*sqrt(2)) )
```
`sigma` (peak width) is known and identical for every compound. `tau >= 0` (the column's
tailing/asymmetry time constant) is the **same single value for every true peak in this
run** — a property of the column, not of the compound — but its numeric value is **not
given**; it must be inferred from the data. As `tau -> 0`, EMG reduces to a plain
symmetric Gaussian. The observed trace is
```
intensity(t) = sum_{i present} Area_i * EMG(t; r_i, sigma, tau) + noise(t),   t = 0..T-1
```
with small measurement noise. A large `tau` gives peaks a long right-hand tail that can
carry substantial mass into a neighboring slot's window — fitting each slot as if it
were symmetric can then "explain" that spillover as a second, entirely fictitious,
compound.

## Input (stdin)
```
testId T M
sigma thr lam
r_1 r_2 ... r_M
I_0 I_1 ... I_{T-1}
```
`testId` may be ignored. `sigma` is the shared peak width. `thr` is the area threshold
above which a slot counts as "declared present" for scoring. `lam` is the area-tolerance
decay used in scoring (see below). `r_1..r_M` are the integer nominal retention times.
`I_0..I_{T-1}` is the observed intensity trace on the integer time grid `t=0..T-1`.

## Output (stdout)
`M` non-negative real numbers `x_1 x_2 ... x_M`, whitespace-separated: your estimated
area for reference slot `i` (use `0` to declare it absent).

## Feasibility
Valid iff **all** hold: exactly `M` numbers are printed; every value is finite
(`nan`/`inf` reject); every value is `>= 0`; every value is `<= 1e9`. Any violation
scores `Ratio: 0.0`.

## Objective
Let `present` be the hidden true set of slots and `a_i` their hidden true areas. A slot
`i` is **predicted present** iff `x_i >= thr`. With `TP`/`FP`/`FN` the usual
true/false-positive/false-negative counts against `present`:
```
F1 = 2*precision*recall / (precision+recall)          (0 if TP=0)
AreaAcc = mean over i in TP of exp( -|x_i - a_i| / a_i / lam )    (0 if TP empty)
F = F1 * AreaAcc
```
Maximize `F`. A phantom compound (false positive) drags down `F1`; a real compound
whose area you get badly wrong drags down `AreaAcc` even if you found it.

## Scoring
```
Ratio = min(1.0, 0.90 * F)
```
printed as `Ratio: <value>`.

## Constraints
- `8 <= M <= 17`, `T <= 500`.
- `sigma`, `thr`, `lam` fixed positive constants given verbatim in the input.
- `tau` (unknown, to be inferred) lies in `[0, 10]` and is identical for every true
  peak in a given instance.
- Time limit 5s, memory 512m.

## Example (worked score, illustrative numbers only)
Suppose `M=3`, `thr=10`, `lam=0.15`, hidden `present={1,3}` with `a_1=40, a_3=25`
(slot 2 is truly absent). You output `x = (38, 6, 20)`. Slot 2's `6 < thr` so it is
correctly predicted absent. `TP={1,3}`, `FP=FN={}`, so `precision=recall=1`, `F1=1`.
`AreaAcc = mean( exp(-|38-40|/40/0.15), exp(-|20-25|/25/0.15) ) = mean(exp(-0.333),
exp(-1.333)) = mean(0.717, 0.264) = 0.490`. `F = 1 * 0.490 = 0.490`, so
`Ratio = min(1, 0.90*0.490) = 0.441`.
