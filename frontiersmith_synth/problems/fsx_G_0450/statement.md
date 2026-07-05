# Reversible Circuit Synthesis for a Modular-Arithmetic Permutation

## Problem
A quantum-arithmetic unit acts on an `n`-bit register as a fixed **permutation**
`pi` of the `2^n` computational basis states. The permutation is the modular
polynomial map

```
pi(x) = ( c0 + c1*x + c2*x^2 + c3*x^3 )  mod 2^n
```

(a Rivest permutation polynomial; the coefficients are given implicitly through the
full truth table in the input). Your job is to **synthesize a reversible logic
circuit** that realizes `pi` exactly, using as **few gates as possible**.

The only gates allowed are **multiple-controlled Toffoli (MCT) gates**, the standard
reversible / quantum library:
- a gate has one **target** bit `t` and a set of **control** bits `C` (`t not in C`);
- applied to a basis state, it **flips bit `t` iff every control bit in `C` is 1**;
- `|C| = 0` is a NOT gate, `|C| = 1` a CNOT, `|C| = 2` a Toffoli, and larger `|C|`
  a generalized Toffoli.

Bits are indexed `0 .. n-1`, with bit `0` the least significant.

## Input (stdin)
```
n
p[0] p[1] ... p[2^n - 1]
```
`n` (`4 <= n <= 8`) is the register width; the second line is the truth table of the
target permutation: `p[x] = pi(x)` for `x = 0 .. 2^n-1`. It is guaranteed to be a
genuine permutation of `0 .. 2^n-1`.

## Output (stdout)
A circuit as a straight-line list of MCT gates, applied **in the order printed**
(top to bottom) to the input register:
```
G
k_1 t_1 <c_1 ... c_{k_1}>
k_2 t_2 <c_1 ... c_{k_2}>
...
k_G t_G <c_1 ... c_{k_G}>
```
- `G` is the number of gates (`0 <= G <= 50000`).
- Each gate line lists its control count `k` (`0 <= k <= n-1`), then the target bit
  `t` (`0 <= t < n`), then exactly `k` distinct control bits (each in `0..n-1`,
  none equal to `t`).

## Feasibility
The circuit is feasible iff it parses under the schema above, every bit index is in
range, controls are distinct and exclude the target, `G <= 50000`, and — crucially —
applying the whole circuit to every input `x in {0,..,2^n-1}` yields exactly `p[x]`.
Any violation (bad schema, out-of-range index, non-integer / non-finite token, or a
single mismatched output) scores `0`.

## Objective
**Minimize** the gate count `G`.

## Scoring
Let `F = G` be your gate count and let `B` be the gate count of a fixed reference
construction (basic transformation-based / MMD synthesis) that the checker builds
itself for this exact permutation. The score is
```
Ratio = min(1.0, 0.1 * B / F)
```
Reproducing the reference cost scores `0.1`; halving it scores `0.2`; a 10x smaller
circuit saturates at `1.0`. Minimal reversible circuit size for a general
permutation is not known in closed form, so the ceiling is genuinely open.

## Constraints
- `4 <= n <= 8`, so `16 <= 2^n <= 256`.
- Deterministic exact scoring: the checker simulates all `2^n` inputs over the
  integers; no timing is involved.

## Example (worked score)
Suppose for some instance the checker's reference construction needs `B = 60` gates
and you submit a circuit with `F = 20` gates that reproduces the truth table exactly.
Then `Ratio = min(1, 0.1 * 60 / 20) = 0.30`. If instead your circuit gets even one
input wrong, `Ratio = 0.0`.
