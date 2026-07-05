# Cooling-Mode Rank of a Thermal-Coupling Tensor

## Problem

A hyperscale data center is cooled by `I` coolant pumps feeding `J` heat
exchangers that serve `K` server zones. The three-way thermal coupling of the
facility is captured by a positive integer tensor `T` of shape `I x J x K`,
where `T[i][j][k]` is the heat drawn from zone `k` when pump `i` drives
exchanger `j` at unit load.

The cooling controller must recompute the full heat map on every control tick.
It does so by superposing **cooling modes**. A single cooling mode is a rank-1
outer product

```
mode = a (x) b (x) c ,   a in Q^I,  b in Q^J,  c in Q^K
```

i.e. its contribution to the heat map is `a[i] * b[j] * c[k]`. Evaluating a mode
costs one scalar multiply per coupled triple; a controller built from `R` modes
performs `R` scalar multiplications per tick. **Fewer modes = a cheaper, faster
controller.**

Your task: express the entire coupling tensor as a sum of as few cooling modes
as possible.

## Input (stdin)

```
I J K
```
followed by `I*J` lines, each with `K` integers: the block for pump/exchanger
pair `(i,j)` in row-major order (`i` outer, `j` inner) gives
`T[i][j][0], T[i][j][1], ..., T[i][j][K-1]`.

All entries are positive integers, so the tensor is fully dense.

## Output (stdout)

```
R
<mode 1: I rationals>          (vector a_1)
<mode 1: J rationals>          (vector b_1)
<mode 1: K rationals>          (vector c_1)
<mode 2: ...>
...
<mode R>
```

The first line is the number of modes `R`. Each mode is then given on three
lines: the `a` vector (`I` numbers), the `b` vector (`J` numbers), the `c`
vector (`K` numbers). Numbers may be integers, exact fractions `p/q`, or
terminating decimals (e.g. `-3`, `7/2`, `1.5`).

## Feasibility

The decomposition is **valid** iff it reconstructs the tensor exactly over the
rationals:

```
sum_{r=1..R}  a_r[i] * b_r[j] * c_r[k]  ==  T[i][j][k]   for every (i,j,k).
```

Any violation (wrong token count, unparsable number, `R <= 0`, or a single
mismatched entry) scores `Ratio: 0.0`.

## Objective

Minimize `R`, the number of cooling modes (scalar multiplications).

## Scoring

Let `B = nnz(T)` be the number of non-zero entries (the schoolbook baseline:
one mode per coupled triple). With `R` your mode count,

```
Ratio = min(1.0, 0.1 * B / R).
```

The schoolbook decomposition scores `0.1`; a controller using `10x` fewer modes
caps at `1.0`. The tensors are generated with an **overcomplete planted rank**
(more hidden modes than any dimension), so the true minimum is not recoverable
by known polynomial factorization methods and remains an open target well below
any slicing-based construction.

## Constraints

- `2 <= I <= J < K <= 8`.
- `0 <= R <= 2*I*J*K`.
- Deterministic scoring: exact rational arithmetic; no randomness, no timing.

## Example

For a `2 x 2 x 2` tensor with `nnz = 8`, the schoolbook decomposition uses
`R = 8` modes and scores `0.1 * 8 / 8 = 0.1`. Slicing along the last axis into
two `2 x 2` matrices of rank 2 gives `R = 4` modes and scores
`0.1 * 8 / 4 = 0.2`. Any decomposition that finds fewer modes scores higher.
