# Resonance-Free Dam Discharge Scheduling

## Problem
You operate a reservoir dam network of **n rivers**. On any given day a dam is set to
a single **operating profile**: a vector of discharge levels, one per river, each level
drawn from `{0, 1, 2}` (`0 = low`, `1 = mid`, `2 = high`). There are `3^n` possible
profiles.

Hydraulic engineers have discovered a destructive **standing-wave resonance**: three
*distinct* profiles `a`, `b`, `c` trigger a resonance cascade whenever, on **every**
river `i`, their levels sum to a multiple of three:

```
(a[i] + b[i] + c[i]) mod 3 == 0    for all i = 1..n
```

You must publish a **catalogue** of approved profiles such that **no three of them
resonate**. Because more approved profiles means more scheduling flexibility, you want
the catalogue to be **as large as possible**.

(Mathematically: the catalogue must be a *cap set* in `F_3^n` — a set of points with no
three collinear. This problem is open-ended: the maximum size is unknown in general and
many different construction strategies are viable.)

## Input (stdin)
A single integer `n` (`4 <= n <= 8`) — the number of rivers.

## Output (stdout)
First line: an integer `M` — the number of approved profiles.
Then `M` lines, each with `n` integers in `{0,1,2}` — one profile per line.

## Feasibility
The output is rejected (score `0`) unless ALL hold:
- each of the `M` rows is a length-`n` vector with every entry in `{0,1,2}`;
- all `M` profiles are pairwise distinct;
- no three distinct profiles resonate (sum to `0 mod 3` on every river).

## Objective
Maximize `F = M`, the number of approved profiles.

## Scoring
Let `B = 2^(n-1)` be the size of the built-in conservative catalogue (fix the last
river to `low`, and use only `low`/`mid` on the remaining `n-1` rivers — always
resonance-free). The score is

```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```

so the conservative catalogue scores `~0.1`, and a catalogue `10x` larger caps at `1.0`.

## Constraints
- `4 <= n <= 8`
- `0 <= M <= 3^n`
- deterministic scoring; exact integer arithmetic only.

## Example
For `n = 4`, the conservative catalogue `{0,1}^3 x {0}` has `B = 8` profiles → `Ratio 0.1`.
A catalogue of `20` profiles (the maximum for `n = 4`) scores
`sc = 100 * 20 / 8 = 250`, i.e. `Ratio 0.250`.
