# Shadowed Film Growth

## Setting

You grow a thin film over a 1D substrate of `L` columns (indices `0..L-1`,
all starting at height `0`) by aiming incoming flux over `T` discrete time
steps. Growth is **ballistic deposition with shadowing**: a particle aimed
at column `a` does not necessarily land there.

**Shadowing (capture).** Within the capture radius `R` of the aim,
`[a-R, a+R]` (clipped to `[0, L-1]`), aim column `a` catches its own
particle **unless** some other column in the window is **more than a
margin `M` taller** than `a` — a genuinely tall neighbor casts a
capturing shadow and steals the flux; a merely slightly-taller one does
not. Among columns clearing the margin, the tallest wins; ties break
toward the column nearest the aim, then the smaller index. So ordinary,
mildly-varying relief grows right where aimed — the danger is a column
that becomes *much* taller than a nearby low neighbor.

**Lateral sticking.** Once the landing column `c` is determined, the
particle sticks by the standard ballistic-deposition rule:
`h[c] = max(h[c] + 1, h[c-1], h[c+1])` (a missing off-substrate neighbor
is ignored). A particle landing next to a taller neighbor sticks to that
neighbor's shoulder instead of adding only one unit — this is what lets
shadowing compound: once a column pulls ahead it both catches more flux
*and* grows faster per catch.

This composition is an intrinsic **roughening instability**: naively-
targeted flux does not relax toward any fixed shape once a lead opens —
taller columns capture disproportionately more of whatever is aimed
nearby. Hitting a target relief exactly is not just a matter of aiming at
whichever column still needs height; you must anticipate where flux will
actually land.

## Input (stdin)
```
L R M T
target_0 target_1 ... target_{L-1}
```
`target_i >= 0` is the desired final height of column `i`. `M` is the
shadowing margin described above. `T` is the number of flux shots you
get, and it is **less than `sum(target)`** — there is never enough flux
to hit every column's target exactly, so deciding where the unavoidable
shortfall lands is itself part of the problem, on top of the shadowing
dynamics.

## Output (stdout)
Exactly `T` whitespace-separated integers `a_1 ... a_T`, each in
`[0, L-1]`: the aim column for time steps `1..T`, in order. The film grows
by processing these shots one at a time, in order, under the shadowing +
lateral-sticking rule above, starting from all-zero heights.

## Feasibility
Output is worth `Ratio: 0.0` if: the token count is not exactly `T`; any
token is not a finite integer; or any aim column is outside `[0, L-1]`.

## Objective

Let `h[0..L-1]` be the final heights after processing your `T` shots.
Minimize the squared error to the target relief:
```
F = sum_i (h[i] - target[i])^2
```

## Scoring
The checker builds its own reference `B`: a simple round-robin schedule
(`aim_t = t mod L`) that is oblivious to both the target and the
shadowing dynamics, run through the identical simulation. Then
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Matching `B` scores `~0.1`; cutting the squared error to a tenth of `B`
caps the score at `1.0`.

## Constraints
`10 <= L <= 44`, `1 <= R <= 3`, `0 <= M <= 5`, `0 <= target_i <= 15`,
`1 <= T < sum(target)`. Time limit 5s, memory 512MB.

## Example
`L=5, R=1, M=3`, target `[0, 6, 0, 4, 0]` (`sum=10`), `T=9`. Suppose the
schedule first spends 6 shots building column 2 up to height 6 (landing
directly — nothing nearby is taller yet), then aims its 3 remaining
shots at column 3. Column 2 is within `R=1` of column 3 and is
`6 > M=3` taller than it, so every one of those 3 shots is captured by
column 2 instead: it overshoots to height 9 while column 3 never moves,
giving `F=(9-6)^2+(0-4)^2=25`. This is the general trap: once a taller
neighbor's lead exceeds `M` within radius `R`, naively continuing to aim
at its shorter neighbor only feeds the leader; reaching a target with a
tall feature next to a valley means respecting that dynamic (e.g.
finishing the valley before the nearby peak opens too large a lead, or
diverting the doomed shots elsewhere) rather than fighting it head-on
after the fact.
