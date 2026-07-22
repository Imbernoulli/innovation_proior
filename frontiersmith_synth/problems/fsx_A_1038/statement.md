# Lighthouse Lattice: Harbors Lit, Harbors Dark

## Problem
A coastal lighthouse authority has `N` candidate lamp positions on a straight
seawall, addressed `0, 1, ..., N-1`. You may activate **at most `K`** of them as
emitters. Each activated emitter at position `i` carries unit amplitude and an
independently chosen **discrete phase** `p in {0, ..., P-1}`, contributing the
phasor `exp(2*pi*i*(p/P))` (a `P`-th root of unity).

Bearings out at sea are indexed the same way, `0, ..., N-1`: bearing `j`
corresponds to the direction `z_j = exp(2*pi*i*j/N)`. The combined field
sensed at bearing `j` from your chosen emitters `S` (positions `i`, phases
`p_i`) is the phased-array sum
```
E(j) = sum_{i in S} exp( 2*pi*i * ( p_i/P + i*j/N ) )
```
(complex arithmetic; `i` here plays double duty as the imaginary unit and the
loop variable, as usual for this kind of sum) and the **intensity** there is
`I(j) = |E(j)|^2`. Note `E` is exactly the array polynomial `A(z) = sum x_i
z^i` (with `x_i` your emitter's phasor) evaluated at the `N`-th roots of
unity `z_j` — the bearings and the lamp positions live on the same lattice.

There are `T` **harbor** bearings that must be illuminated, and `Q`
**protected** bearings (fishing grounds, rival ports, ...) that must stay
below a given per-bearing intensity threshold.

## Input (stdin)
```
N K P
T
t_1 t_2 ... t_T
Q
q_1 thr_1
q_2 thr_2
...
q_Q thr_Q
```
`t_1..t_T` are the harbor bearings; each `q_k thr_k` is a protected bearing
and its maximum allowed intensity.

## Output (stdout)
```
m
i_1 p_1
i_2 p_2
...
i_m p_m
```
Print `m` (`1 <= m <= K`, you need not spend the whole budget), then `m`
distinct emitter positions `i_k in [0, N)` with phases `p_k in [0, P)`.

## Feasibility
Invalid (scores `Ratio: 0.0`) if: `m` is outside `[1, K]`; any `i_k` or `p_k`
is out of range; positions repeat; or **any** protected bearing exceeds its
threshold (checked with tolerance `1e-6`). Feasibility is a hard gate —
gain at the harbors never buys back a lit protected bearing.

## Objective
Maximize `F`, the **worst-illuminated harbor**:
```
F = min_{j in targets} I(j)
```

## Scoring
The checker builds its own reference `B`: the largest divisor `d` of `N`
with `d <= K` such that `d` divides every harbor bearing but no protected
bearing (such a `d` always exists for the generated instances). `B = d^2`
— the intensity a single "comb" of `d` equally spaced, equal-phase emitters
achieves at any bearing that is a multiple of `d` (and it is *exactly* zero
everywhere else, by a finite-geometric-series identity, so it never risks a
protected bearing). Then:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching the reference exactly scores `0.1`; `10x` better caps at `1.0`.

## Constraints
- `10 <= N <= 42`, `4 <= K <= 32`, `P = 8`.
- `2 <= T <= 3`, `0 <= Q <= 5`.
- Time limit 5s, memory 512m.

## Example (illustrative shape only — not a real test case)
Suppose `N = 8, K = 4, P = 8`, one harbor `t_1 = 2` and one protected bearing
`q_1 = 1` with `thr_1 = 3`. Emitters at positions `0` and `4`, both phase `0`,
give `E(2) = exp(0) + exp(2*pi*i*4*2/8) = 1 + 1 = 2`, so `I(2) = 4` there,
while `E(1) = exp(0) + exp(2*pi*i*4*1/8) = 1 + (-1) = 0`, so the protected
bearing is exactly dark regardless of its threshold. Spending only 2 of the 4
available emitters this way is a valid (if modest) submission.
