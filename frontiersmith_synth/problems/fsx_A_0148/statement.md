# Festival Vibe Tensor: Minimum-Act Rank Decomposition

## Problem
A music festival's "vibe" is measured on a three-way grid: for each **stage** `i`
(`0..a-1`), **timeslot** `j` (`0..b-1`) and musical **genre** `k` (`0..c-1`) the
producers have recorded an integer intensity `T[i][j][k]`.

The lineup is built out of **acts**. A single act is a *separable* pattern: it has
a stage profile `u ∈ ℚ^a`, a timeslot profile `v ∈ ℚ^b` and a genre profile
`w ∈ ℚ^c`, and it contributes intensity `u[i]·v[j]·w[k]` to every cell. Stacking
`R` acts reproduces the whole grid when

```
T[i][j][k] = Σ_{r=1..R} u_r[i] · v_r[j] · w_r[k]      for all i,j,k.
```

You must reproduce the recorded tensor **exactly** using **as few acts as
possible**. In algebraic terms: output a rank-`R` CP (canonical polyadic)
decomposition of `T` and minimise `R`.

## Input (stdin)
```
a b c
<a·b·c integers>        # index order: i outer, then j, then k
```
The integers are laid out as `a` blocks, each of `b` lines, each line holding `c`
values (`T[i][j][0..c-1]`). Whitespace between tokens is arbitrary; read by token.
Dimensions satisfy `1 ≤ a,b,c ≤ 5`.

## Output (stdout)
```
R
u_1[0] .. u_1[a-1]  v_1[0] .. v_1[b-1]  w_1[0] .. w_1[c-1]
...
u_R[0] .. u_R[a-1]  v_R[0] .. v_R[b-1]  w_R[0] .. w_R[c-1]
```
The first token is the number of acts `R ≥ 1`. Then `R` rows follow, each with
exactly `a+b+c` numbers: the stage profile, then the timeslot profile, then the
genre profile of that act. Entries may be integers, exact fractions `p/q`, or
decimals; they are parsed as exact rationals. `nan`/`inf` are rejected.

## Feasibility
The submission is feasible only if it parses to the exact schema above and its
reconstruction equals `T` **exactly** (exact rational arithmetic). Any schema
violation, non-finite value, out-of-range `R`, or reconstruction mismatch scores
`0`.

## Objective
Minimise the number of acts `R` (fewer scalar multiplications to realise the grid).

## Scoring
Let `B = a·b` be the rank of the canonical *mode-3 slab* decomposition (one act
per stage×timeslot pair). With a feasible submission of `R` acts,
```
Ratio = min(1.0, 0.1 · B / R).
```
Reproducing the baseline (`R = B`) scores `0.1`; a 10×-smaller decomposition
caps at `1.0`. Lower `R` is strictly better.

## Constraints
- `1 ≤ a,b,c ≤ 5`, integer `T`.
- `1 ≤ R ≤ 5000`.
- Deterministic exact-rational grading; no time or randomness in the score.

## Notes / difficulty
The multilinear (Tucker) ranks of `T` are small, so slab-style constructions and
Tucker compression give easy upper bounds — but the true CP rank is **open**: it
can lie strictly below the product of the two smallest multilinear ranks, and no
polynomial method recovers it in general (Jennrich-style simultaneous
diagonalisation needs rank ≤ a dimension, which is violated here). Multiple
distinct strategies exist and there is no known closed-form optimum.

## Example (worked score — illustrative shape only, not a test)
Input (`a=b=c=2`):
```
2 2 2
2 1
2 1
1 3
0 0
```
i.e. `T = [[[2,1],[2,1]], [[1,3],[0,0]]]`. A valid **2-act** answer:
```
2
1 0   1 1   2 1
0 1   1 0   1 3
```
Check: act 1 = (1,0)⊗(1,1)⊗(2,1) fills the `i=0` slab with `[[2,1],[2,1]]`; act 2
= (0,1)⊗(1,0)⊗(1,3) fills `i=1` with `[[1,3],[0,0]]`; together they equal `T`.
Here `B = a·b = 4`, so `Ratio = min(1, 0.1·4/2) = 0.200`. The trivial mode-3 slab
would use `R = 4` acts and score `0.100`.
