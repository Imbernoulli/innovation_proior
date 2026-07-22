# Recompressed Scroll: Minimal-Product Circuit for Shared Hidden Factors

An alchemist's scroll once held a few compact formulae. A careless apprentice
*expanded* every product, leaving a bloated list of monomials. Recompress the
scroll: build one arithmetic circuit that recomputes all the formulae using as
few genuine multiplications as possible.

## Input (stdin)
```
p n k
```
then, for each of the `k` target polynomials `P_0..P_{k-1}` over variables
`x_0..x_{n-1}` (all arithmetic modulo the prime `p`):
```
M                      # number of monomials of this target
M lines, each:  c e_0 e_1 ... e_{n-1}
```
Line `c e_0 ... e_{n-1}` is the monomial with coefficient `c` (in `[0,p)`) and
exponent `e_j` on `x_j`, so the target is `sum over its monomials of
c * prod_j x_j^{e_j}`, reduced mod `p`. The targets are given fully expanded;
they were secretly generated from a small pool of shared low-degree factors, but
that structure is not given to you.

## Output (stdout): a straight-line program (SLP)
```
L
L lines, each one of:
    const v            # a constant v in [0,p)
    add i j            # node[i] + node[j]  (mod p)
    sub i j            # node[i] - node[j]  (mod p)
    mul i j            # node[i] * node[j]  (mod p)
out o_0 o_1 ... o_{k-1}
```
Nodes `0..n-1` are the input variables `x_0..x_{n-1}` (implicit). Instruction `t`
(0-based) defines node `n+t` and may reference only **earlier** nodes, so the
program is a DAG. The final `out` line names, for each target `P_i`, the node
that must equal it.

## Feasibility
Every `add/sub/mul` operand and every `out` index must reference an already
defined node; every `const` value must lie in `[0,p)`; the `out` line must list
exactly `k` indices. The program is valid only if, for every `i`, the value at
node `o_i` equals `P_i` **as a polynomial** (verified by exact evaluation at many
seeded random points in `F_p^n`). Any parse error, out-of-range value, or a
single mismatch scores `Ratio: 0.0`.

## Objective (minimize) — the cost model
Cost = number of **non-scalar multiplications**: a `mul i j` counts as `1`
**only when both operands are non-constant** (each depends on at least one input
variable). Multiplication by a constant, as well as every `add`, `sub` and
`const`, is **free**. Thus any linear form `c_0 x_0 + ... + c_{n-1} x_{n-1}`
costs `0`; the whole difficulty is how few genuine products rebuild all `k`
targets, reusing shared work across them.

## Scoring
Let `cost` be your non-scalar multiplication count. The checker builds a baseline
`B` = the cost of the canonical *expanded* evaluation (shared univariate powers,
then one product-chain per distinct monomial). Your score is
```
Ratio = min(1.0, 0.1 * B / cost)
```
Reproducing the expanded baseline scores about `0.1`; genuinely fewer products
score higher, and the ceiling stays open above any reference solution. Larger
tests use more variables, more factors and more targets.

## Constraints
`p` is prime near `2^31`; `1 <= n <= ~60`; `1 <= k <= ~40`; time limit 5 s,
memory 512 MB. Scoring is exact and deterministic.

## Example (illustrative FORM only — small, not from the test set)
Two targets over `x_0..x_3`, `p = 101`:
```
101 4 2
4
1 1 1 0 0
1 1 0 1 0
1 0 1 1 0      # wrong shape on purpose -- see below
...
```
The point of the example is the *shape*, not real data: a target such as
`x_0 x_2 + x_0 x_3 + x_1 x_2 + x_1 x_3` has four monomials, yet it equals
`(x_0 + x_1)(x_2 + x_3)` — one non-scalar multiplication instead of four. When
several targets share such hidden factors, computing each shared factor once and
reusing it slashes the product count far below the monomial baseline. The bonus
coefficients that tie the factors together live in the input, not here.
