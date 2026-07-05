# Bilinear Limb-Kernel Splits: Minimizing Scalar Multiplications

## Problem
A big-number library must implement a fixed **limb-combination kernel**. Given two
short integer limb vectors `x = (x_0, ..., x_{p-1})` and `y = (y_0, ..., y_{q-1})`,
the kernel evaluates `r` prescribed **bilinear forms** simultaneously:

```
b_k(x, y) = sum_{i=0..p-1} sum_{j=0..q-1} T[k][i][j] * x_i * y_j      (k = 0 .. r-1)
```

The coefficient tensor `T` is fixed and published. On limb hardware the expensive
operation is a **scalar multiplication** of two limb-sized quantities; additions and
multiplications by fixed integer constants are cheap (shifts/adds) and are *not*
charged. Following Karatsuba/Toom accounting, you may precompute any linear
combinations of the `x_i` and of the `y_j` for free; you pay only for the number of
**scalar products** you form.

A **split** is a list of `R` scalar products (terms). Term `t` is
```
m_t = ( a_t . x ) * ( c_t . y )      where  a_t in Q^p,  c_t in Q^q,
```
and each output is recovered by a free linear mix `b_k = sum_t d_t[k] * m_t` with
`d_t in Q^r`. A split is **valid** iff it computes every form exactly, i.e.
```
T[k][i][j] == sum_t d_t[k] * a_t[i] * c_t[j]      for all k, i, j.
```

Design a valid split using as **few** scalar products `R` as possible.

## Input (stdin)
```
line 1:  p q r
then r blocks, each p lines of q integers  =  T[k][i][0..q-1]
```
So the `k`-th block (0-indexed) is the `p x q` coefficient matrix of output `b_k`.
Bounds: `4 <= p,q <= 7`, `3 <= r <= 5`; coefficients are small integers.

## Output (stdout)
```
line 1:  R                                   (number of scalar products, R >= 1)
next R lines: a_t[0..p-1]  c_t[0..q-1]  d_t[0..r-1]     (p+q+r tokens per line)
```
Each token is an integer or a rational `n/d` (`|n|, d <= 1e7`). No `nan`/`inf`/floats.

## Feasibility
The submitted split must reconstruct `T` **exactly** over the rationals (checked with
exact arithmetic). Any identity mismatch, malformed line, out-of-range `R`
(`1 <= R <= 4*p*q`), or non-integer/rational/non-finite token scores `0`.

## Objective
Minimize `R`, the number of scalar products.

## Scoring
Let `B = p*q` be the schoolbook baseline (form every product `x_i*y_j`). With a valid
split of size `R`:
```
sc    = min(1000, 100 * B / R)
Ratio = sc / 1000
```
A schoolbook split (`R = p*q`) scores `0.1`; halving the products doubles the score.
The tensor is planted with an **overcomplete** internal structure, so its true minimal
`R` is not known in closed form -- there is no reachable optimum, only better splits.

## Constraints
Deterministic exact-rational scoring; no timing. `p,q <= 7`, `r <= 5`.

## Example (worked score)
Suppose `p=q=4, r=3`, so `B = 16`. A schoolbook split forms all `16` products and
scores `Ratio = 100*16/16 / 1000 = 0.1`. A split that reuses linear structure to
compute all three forms with only `8` scalar products scores
`Ratio = 100*16/8 / 1000 = 0.2`. Reaching `4` products would score `0.4`.
(Illustrative arithmetic only -- the actual tensor and its attainable `R` differ.)
