# Resonance-Free Substation Deployment

## Problem
A regional power grid has **n** high-voltage transmission lines. Each candidate
**substation** is installed with a *phase configuration*: an assignment of a
three-phase offset in `{0, 1, 2}` to every line. A configuration is therefore a
vector in `F_3^n` (integers mod 3), and there are `3^n` possible configurations.

The grid operator has learned the hard way that certain balanced triples of
substations excite a destructive **resonance cascade**. Concretely, three
*distinct* deployed configurations `a, b, c` trigger a cascade exactly when

```
a[i] + b[i] + c[i] ≡ 0  (mod 3)   for every line i = 0 .. n-1,
```

i.e. when the three configurations form a line in `F_3^n` (a 3-term progression).
A deployment that contains no such triple is **resonance-free** (a *cap set*).

Some configurations are already **reserved** for maintenance / spares and may not
be deployed. You must choose a resonance-free deployment, using no reserved
configuration, that powers **as many substations as possible**.

## Input (stdin)
```
n k
r_1[0] r_1[1] ... r_1[n-1]
...
r_k[0] r_k[1] ... r_k[n-1]
```
- `n` — number of transmission lines (`3 ≤ n ≤ 7`).
- `k` — number of reserved configurations.
- Each of the next `k` lines is one reserved configuration: `n` phases in `{0,1,2}`.

## Output (stdout)
```
m
c_1[0] c_1[1] ... c_1[n-1]
...
c_m[0] c_m[1] ... c_m[n-1]
```
- `m` — number of substations you deploy.
- Each of the next `m` lines is one deployed configuration: `n` phases in `{0,1,2}`.

## Feasibility
An output is valid iff **all** hold:
1. every phase is in `{0, 1, 2}`;
2. all `m` deployed configurations are distinct;
3. none is a reserved configuration;
4. the deployment is resonance-free: no three distinct deployed configurations
   sum to `0` on every line (no line / 3-term progression in `F_3^n`).

Any violation scores `Ratio: 0.0`.

## Objective
Maximise `m`, the number of deployed substations.

## Scoring
Deterministic. Let `F = m` for a valid deployment and let `B` be an internal
audit baseline the checker builds itself: the resonance-free grid of
configurations with phase in `{0,1}` on lines `0..n-2` and phase `0` on the last
line, minus any reserved config (`|B| = 2^(n-1)`; reserved configs carry phase 2
on the last line, so they never intersect it). The reported score is

```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```

Reproducing the audit baseline scores `≈ 0.1`; a deployment `10×` the baseline
caps at `1.0`. The final grade averages `Ratio` over all test cases.

## Constraints
- `3 ≤ n ≤ 7`, `k = n`.
- Configurations are vectors over `F_3`. Scoring uses exact integer arithmetic
  only — no timing, no randomness in the grade.

## Example
Suppose `n = 3`, `k = 3`, with reserved configs `(0,0,2), (1,0,2), (0,1,2)`.
The audit baseline is `{(0,0,0),(1,0,0),(0,1,0),(1,1,0)}`, size `B = 4`, so a
4-substation deployment matching it scores `100*4/4/1000 = 0.1`. The full
`{0,1}^3` cap `{(x,y,z): x,y,z ∈ {0,1}}` has 8 configurations, none reserved and
resonance-free, giving `F = 8` and `Ratio = 100*8/4/1000 = 0.2`. A smarter search
finds a 9-configuration resonance-free deployment in `F_3^3`, scoring `0.225`.
