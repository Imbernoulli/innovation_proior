# Vein Harvest in a Crystal Lattice

A crystal is described by a full-rank integer lattice
`L = { x_1 b_1 + ... + x_n b_n : x_i in Z }` in `Z^n`, where the rows `b_1,...,b_n`
of an `n x n` integer matrix `B` are a basis. Somewhere inside this crystal a **dense
vein** has been planted: a whole residue class of lattice points that is unusually rich in
*short* vectors. The basis you are handed has been scrambled, so the vein is not visible in
the rows themselves — you must dig it out. Harvest `k` independent short lattice vectors.

## Input (stdin)
```
n p k
B[0][0] B[0][1] ... B[0][n-1]
...
B[n-1][0] ...     B[n-1][n-1]
```
- `n` (7 or 8) is the dimension, `p` is a small prime, `k = n - 2`.
- The next `n` lines are the basis `B` (integer entries). `det B != 0`.

The vein is the set `{ v in L : a . v == 0 (mod p) }` for a **hidden** linear form
`a in (Z/pZ)^n`. Neither `a` nor the vein is given — only that one such planted residue
class exists and is anomalously dense in short vectors.

## Output (stdout)
Exactly `k` lines, each `n` integers: the ambient coordinates of `k` lattice vectors
`v_1, ..., v_k`.

## Feasibility (all must hold, else score 0)
1. Each `v_i` is a lattice vector: `v_i = x_i B` for some integer row `x_i` (i.e. an
   integer combination of the rows of `B`).
2. Each `v_i` is nonzero.
3. The `k` vectors are linearly independent over the rationals.
Non-integer / `nan` / `inf` tokens, wrong token count, or any violation ⇒ `Ratio: 0.0`.

## Objective (MINIMIZE)
`F = sum_{i=1..k} || v_i ||_2^2`  (sum of squared Euclidean norms). Shorter, denser
harvests score higher.

## Scoring
The checker builds an internal baseline `Bbase` = the sum of the `k` smallest squared
row-norms of `B` (a trivial feasible harvest: `k` independent basis rows). Then
```
sc    = min(1000, 100 * Bbase / F)
Ratio = sc / 1000
```
So submitting the `k` shortest basis rows scores `Ratio ≈ 0.1`, and a harvest whose total
squared length is `10x` smaller than the baseline caps at `1.0`. The score is deterministic
and reproducible.

## Why this is hard
Running a generic basis reduction on the whole scrambled `B` chases short vectors at the
crystal's *average* density and levels off well short of the vein. The planted density lives
in exactly one of the many congruence classes `{v : a . v == 0 (mod p)}`; the short vectors
you want are concentrated there. Intersecting `L` with a congruence sublattice and reducing
*that* is far more productive than reducing all of `L` — but only for the right residue rule,
which you must discover. There are on the order of `p^n` candidate classes.

## Constraints
- `7 <= n <= 8`, `p = 2`, `k = n - 2`. Time limit 5 s, memory 512 MB.
- Entries of `B` fit in 64-bit integers.

## Example (illustrative, not a real test)
Suppose `n = 3`, `k = 1`, and after your search you find the lattice vector `v = (1, -1, 0)`
with `||v||^2 = 2`. If the checker's baseline shortest row had squared norm `50`, then
`F = 2`, `Bbase = 50`, `sc = min(1000, 100 * 50 / 2) = 1000`, `Ratio = 1.0`. A weaker harvest
`v = (3, 1, -2)` with `||v||^2 = 14` would give `sc = 100 * 50 / 14 ≈ 357`, `Ratio ≈ 0.357`.
(Numbers illustrate the formula only; real tests have `k > 1` and a planted vein.)
