# Chord of Masses

## Problem

You are loading a single physical string until it plays a chord.

The string has `n` beads at evenly spaced positions `1..n` along a massless,
fixed-tension string with both ends clamped (displacement zero at position
`0` and at position `n+1`). Bead `i` naturally carries mass `1`, but you may
glue extra integer **mass units** onto it: after loading, bead `i` has mass
`m_i = 1 + e_i` with `e_i >= 0` an integer. You have a total budget of `B`
mass units to distribute (`sum(e_i) = B` **exactly**), and no single bead
may receive more than `CAP` units (`0 <= e_i <= CAP`).

For small transverse displacements `y_i`, Newton's law for bead `i` under
unit tension/spacing gives the standard discrete-string dynamics
`m_i * y_i'' = -(2*y_i - y_{i-1} - y_{i+1})` (with `y_0 = y_{n+1} = 0`).
The normal-mode frequencies `omega_1 < omega_2 < ... < omega_n` are the
square roots of the generalized eigenvalues of `K v = lambda * M * v`,
where `K` is the fixed `n x n` tridiagonal matrix with `2` on the diagonal
and `-1` on the two off-diagonals, and `M = diag(m_1,...,m_n)`.

You are given a **target chord**: rational target ratios `T_1=1, T_2, ...,
T_r` (`r <= n`) that the first `r` mode frequencies should approximate,
i.e. you want `omega_k / omega_1 ~= T_k` for `k = 1..r` (mode 1, the
fundamental, is always the chord's root). Loading a string with more mass
always slows every mode down — never speeds one up — so shaping the
*relative* pitches of a chord out of an unloaded string (whose natural
overtones sit close to the harmonic series `1,2,3,...`) requires choosing
**where**, not just **how much**, mass to add.

## Input (stdin)

```
n B CAP r
p_1 q_1 p_2 q_2 ... p_r q_r
```
`n` beads, budget `B`, per-bead cap `CAP`, `r` target modes, and `r`
rational target ratios `T_k = p_k/q_k` (always `p_1=q_1=1`).

## Output (stdout)

`n` integers `e_1 ... e_n` (whitespace/newline separated), the mass units
assigned to each bead.

## Feasibility

* Exactly `n` integer tokens.
* `0 <= e_i <= CAP` for every bead.
* `sum(e_i) == B` exactly.

Any violation scores `Ratio: 0.0`.

## Objective (minimize)

Let `omega_1 <= ... <= omega_r` be the `r` smallest eigenfrequencies
achieved by your mass vector. Your cost is
```
F = sum_{k=2}^{r} ( ln(omega_k/omega_1) - ln(T_k/T_1) )^2
```
— the summed squared error, in log-frequency space, between the chord you
actually built and the chord you were asked for (mode 1 is always its own
perfect match). **Lower `F` is better.**

## Scoring

The checker also builds its own baseline `e_i` by spreading `B` as evenly
as possible across all `n` beads (no attempt to shape the chord), giving
baseline cost `F_base`. Your printed score is
```
ratio = min(900, 100 * F_base / max(1e-6, F)) / 1000
```
so an output that ties the uniform baseline scores `~0.1`; a genuinely
well-shaped chord scores much higher, capped at `0.9` (real headroom is
intentionally left above any reference solution).

## Example (illustrative shape only — not a generated test case)

`n=6, B=5, CAP=4, r=2`, target `T = (1, 3/2)` (input: `6 5 4 2` then
`1 1 3 2`).

The uniform baseline spreads `B=5` over `6` beads as `e=(1,1,1,1,1,0)`,
which achieves `omega_2/omega_1 = 1.9983` (close to the unloaded string's
natural near-octave) — far from the target `1.5`, giving `F_base=0.08228`.

Loading the two ends more heavily instead, `e=(2,0,0,0,0,3)` (feasible:
each `<=4`, sum `=5`), achieves `omega_2/omega_1 = 1.5336`, giving
`F = 0.000492`. Score: `min(900, 100*0.08228/0.000492)/1000 = 0.9`.

## Constraints

`8 <= n <= 60`, `2 <= r <= 5`, `r <= n/3`, `1 <= B <= 300`,
`1 <= CAP <= B`, `1 <= p_k, q_k <= 2000`. Time limit 5s, memory 512MB.
10 test cases span this range, small/loose to large/adversarial.
