# High-Tone Sieve: Cheapest Circuit for a Spectral Cutoff

## Problem
You are given a sparse, symmetric `n x n` matrix `A` whose eigenvalues are guaranteed
to avoid a band around a fixed cutoff `theta = 0.5`: every eigenvalue lies at distance
at least `gap` from `theta` (this `gap` is revealed to you exactly). Think of the
eigenvalues as "tones" -- your job is to build the cheapest possible **sieve**: a
straight-line arithmetic program that, given any of several probe vectors `v`, computes
(to within tolerance `epsilon`) the vector `y = P v`, where `P` is the exact spectral
projector onto the eigenspace of eigenvalues **above** `theta` (the "high tones").

Your program is a fixed sequence of instructions over a small register file (register
`0` always holds the current probe vector; other registers start at zero). Allowed
instructions:

```
NVEC m                    declare m registers (0 <= all indices < m)
MATVEC d s                reg[d] = A @ reg[s]
AXPBY  d a s1 b s2        reg[d] = a*reg[s1] + b*reg[s2]
SCALE  d a s              reg[d] = a*reg[s]
COPY   d s                reg[d] = reg[s]
CSOLVE d a b s             reg[d] = Re[ (A - (a+b*i) I)^-1 @ reg[s] ]
OUTPUT s                  declare reg[s] the answer (exactly one, must be last)
```
`a`, `b` are arbitrary real literals (decimal, no `nan`/`inf`). Register `0` may never
be written. At most 64 registers and 20000 instructions. The SAME program is replayed
once per probe vector; its op cost is charged once.

`CSOLVE` is real: it internally solves the complex shifted linear system and returns
the real part -- for a real symmetric `A` this is a genuine, well-defined linear map.

## Input (stdin)
```
n
theta epsilon bandwidth
gap
nnz
nnz lines: i j value      (0-indexed, i<=j, symmetric, includes the diagonal)
k
k lines of n floats        (unit-norm probe vectors)
```

## Output (stdout)
Your program, as described above.

## Feasibility
Let `y_ref = P v` be computed exactly from `A`'s true eigendecomposition. Your program
is feasible iff, replayed on **every** probe `v`, it produces a finite output with
`||y - y_ref||_2 <= epsilon`. Any parse error, out-of-range register, non-finite value,
missing/duplicate `OUTPUT`, or accuracy violation scores `0`.

## Objective
Minimize the total scalar-operation cost `F` of your program:
- `MATVEC` costs `2 * nnz_full` (mult+add per stored nonzero, both matrix triangles);
- `AXPBY` costs `3n`; `SCALE` costs `n`; `COPY`/`OUTPUT` are free;
- `CSOLVE` costs `8*n*bandwidth^2 + 4*n*bandwidth` (banded complex factor+solve).

## Scoring
The checker builds its own reference cost `B = ceil(6/gap) * (2*nnz_full + 9n)` (the cost
of a generously-safe flat filter sized off the revealed gap). With your cost `F`:
```
Ratio = min(1, 0.1 * B / F)
```
Fewer ops score higher; there is no known optimal circuit, so real headroom remains
above what any of the reference strategies below reach.

## Constraints
`30 <= n <= 150`, `1e-3 <= gap <= 0.4`, `epsilon = 5e-3`, `bandwidth = 3`, `4` probes
per case, `1 <= nnz <= 5n`. Deterministic; no timing in the score.

## Example
Suppose a tiny instance needs `F = 40000` ops and the checker's own baseline is
`B = 60000`. Then `Ratio = min(1, 0.1 * 60000 / 40000) = min(1, 0.15) = 0.15`: beating
the safe baseline by 1.5x nets a modest score; a submission reaching `F = 6000` (10x
below `B`) would cap at `Ratio = 1.0`. A single flat polynomial filter must grow its
degree roughly like `1/gap` to resolve a narrow gap -- but a program that instead
composes a `MATVEC` with a handful of `CSOLVE` shifted-resolvent terms, chosen from the
revealed `gap`, needs only about `log(1/gap)` such terms to reach the same accuracy,
at several-fold lower total cost on tight-gap instances.
