# Alloy Composition Search

You are designing a metal alloy by dissolving **K** candidate solute elements
into a base metal. Element `i` has a **strengthening coefficient** `s_i` and
an **intermetallic-forming tendency** `b_i` (positive integers, `b_i >= 1`).
Choose nonnegative integer amounts `x_1, ..., x_K` (atomic parts of each
solute) to dissolve. Two things happen as you add solute:

1. **Solid-solution strengthening.** Each element raises yield strength by
   the standard sub-linear (Labusch-type) law: element `i` alone contributes
   `s_i * sqrt(x_i)`, additive across elements: `F = sum_i s_i * sqrt(x_i)`.
   Strength keeps rising the more you dissolve — no local optimum from this
   term alone.
2. **Phase-diagram boundary.** The total solute load `X = sum_i x_i` must not
   exceed a hard cap `MAXX`, and the composition must stay inside the
   **solid-solution phase field**: the diagram is discretized into `numBins`
   equal-width bands of width `W` along the total-solute axis (band `k`
   covers `X` in `[k*W, (k+1)*W - 1]`). Each band has its own budget `T[k]`
   for the alloy's **intermetallic score** `IM = sum_i b_i * x_i`. If the
   band `k` your `X` falls into has `IM > T[k]`, the alloy forms brittle
   intermetallic phases and is **worthless** — not degraded, worthless.

The band budgets `T[k]` are **not guaranteed to be monotonic** in `X`. A
generous band can be followed by a narrow brittle band, itself followed by an
even more generous band further out (real alloys show this: a two-phase
field can pinch the safe composition range before a different single-phase
field opens up beyond it). "More total solute means more room" is not
promised — read the actual band table and reason about where the true
optimum sits, including regions past a tight band.

## Input (stdin)

```
K  W  numBins
s_1 s_2 ... s_K
b_1 b_2 ... b_K
T_0 T_1 ... T_{numBins-1}
```

`MAXX = numBins*W - 1`. All values are positive integers, `b_i >= 1` for
every `i`. `T[k]` is not required to relate to `k` in any particular way: a
below-average-cost mix can push deep into a band, and a generous band's
budget can exceed what an average-cost mix needs for its own top edge.

## Output (stdout)

One line: `K` nonnegative integers `x_1 ... x_K`, the amount of each solute
element to dissolve.

## Feasibility

* Exactly `K` integers, each `>= 0`.
* `X = sum(x_i) <= MAXX`.
* `IM = sum(b_i * x_i) <= T[X // W]`.

Any violation (wrong token count, non-integer/non-finite token, negative
amount, `X` too large, or the brittleness check failing) scores `0`.

## Objective and scoring

Maximize `F = sum_i s_i * sqrt(x_i)`. The grader also builds an internal
baseline `B`: try every element alone (others at zero), each at its largest
amount that is feasible while confined to the first three bands
(`X <= 3*W-1`); `B` is the best such single-element strength. Then

```
Ratio = min(1000, 100 * F / max(1e-9, B)) / 1000
```

Reproducing the baseline scores `Ratio ≈ 0.1`; the score rises with `F` and
saturates at `1.0` once `F >= 10*B`.

**Worked example (illustrative shape only).** `K=2, W=5, numBins=1`
(`MAXX=4`), `s=[10,10]`, `b=[1,2]`, `T=[4]`. Baseline: element 0 alone reaches
`Bx=4`, `B=10*sqrt(4)=20`; element 1 alone reaches only `Bx=2`
(`10*sqrt(2)≈14.1`); so `B=20`. Submission `x=[4,0]`: `F=20`, `Ratio=0.1`.
Submission `x=[2,1]` (`IM=2+2=4<=4`): `F=10*sqrt(2)+10*sqrt(1)≈24.1`,
`Ratio≈0.121`. Submission `x=[3,1]` (`IM=3+2=5>4`): brittle, `Ratio=0`.

## Why the obvious approach struggles

Only the final submitted composition is graded — nothing requires that some
incremental path to it stayed feasible the whole way. Growing the
composition one unit at a time, always adding to whichever element most
improves strength per unit of brittleness cost, and **stopping the instant
the band currently being passed through would be breached**, throws that
freedom away: it can never discover that a narrow brittle band is followed
by a far more generous one, because it treats the first breach as fatal
instead of simply not submitting a composition inside the brittle band.

## Constraints

`3 <= K <= 6`, `W = 20`, `numBins = 10` (`MAXX = 199`), `18 <= s_i <= 87`,
`1 <= b_i <= 9`, band budgets `T[k]` up to roughly `1300` (scaled around the
midpoint of the `b_i` range, so typical mixes reach meaningfully into most
bands). Time limit 5 s, memory 512 MB. Fully deterministic; 10 test cases,
several with a deliberately non-monotonic band table.
