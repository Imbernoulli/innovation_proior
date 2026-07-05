# Orbital Debris Cleanup Manifest (Resonance-Free Cap)

## Problem
A debris survey has catalogued objects in low orbit. Each object carries an **orbital
signature**: a vector of `n` ternary parameters `d[0..n-1]`, each in `{0,1,2}` (three
discretized buckets of inclination, eccentricity, RAAN, ...). The full catalogue is
every signature in `F_3^n` (there are `3^n` of them), EXCEPT a set of `m` **protected**
signatures that belong to active satellites and may never be de-orbited.

You must produce a **cleanup manifest**: a set `S` of distinct, non-protected signatures
to remove. The mission-safety rule forbids a **resonance triple**: three *distinct*
manifest objects `a, b, c` whose signatures are collinear in `F_3^n`, i.e.

```
(a[i] + b[i] + c[i]) mod 3 == 0   for every coordinate i.
```

De-orbiting a resonance triple simultaneously would seed a Kessler cascade, so no such
triple may appear in `S`. (Equivalently, `S` must be a **cap set** that avoids the
protected signatures.)

Your goal is to clear as much debris as possible: **maximize `|S|`**.

## Input (stdin)
```
n m
<protected signature 1: n integers in {0,1,2}>
...
<protected signature m>
```
`n` is the signature length, `m` the number of protected signatures.

## Output (stdout)
One line per selected debris object, each line `n` integers in `{0,1,2}` separated by
spaces — the signature of an object you remove. Print as many lines as you like (possibly
zero). Order does not matter.

## Feasibility
The manifest is accepted only if ALL hold:
- every line has exactly `n` integers, each in `{0,1,2}`;
- no signature is repeated;
- no signature is one of the `m` protected signatures;
- `S` contains **no resonance triple** (no three distinct collinear signatures).
Any violation scores `0`.

## Objective
Maximize the number of removed objects `|S|`.

## Scoring
Let `F = |S|` for a feasible manifest, and let `B` be the size of a fixed internal
reference manifest (the "one-axis / low-bucket" sweep the checker builds itself). The
score is
```
Ratio = min(1000, 100 * F / max(1e-9, B)) / 1000
```
so the reference construction scores `~0.1` and you must roughly `10x` it to saturate.
Reproducing the reference scores about `0.1`; smarter orderings score strictly higher.

## Constraints
- `4 <= n <= 7`, so `81 <= 3^n <= 2187`.
- `0 <= m <= 3^n`. Time limit 5s, memory 512MB.

## Example (worked score)
Suppose `n = 4`, and the reference manifest has `B = 8` objects. If your feasible manifest
removes `F = 18` objects with no resonance triple, then
`Ratio = 100 * 18 / 8 / 1000 = 0.225`. A manifest that repeats a signature, includes a
protected one, or contains three collinear signatures scores `0.0`.
