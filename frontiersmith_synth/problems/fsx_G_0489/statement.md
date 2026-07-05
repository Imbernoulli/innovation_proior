# Dense Admissible Set in a Window (Prime-Sieve Design)

## Problem
An integer set `H = {h_1, ..., h_k}` is called **admissible** if, for every prime `p`,
the residues `{h mod p : h in H}` do **not** cover all `p` residue classes modulo `p`
(equivalently, for every prime there is at least one residue class that no element of `H`
occupies). Admissible sets are exactly the shapes that the Hardy–Littlewood prime
`k`-tuples conjecture predicts can be translated to contain infinitely many all-prime
constellations, so building large admissible sets inside a short window is the core
combinatorial design problem behind prime-gap and bounded-gap records.

You are given a window width `W`. Construct an admissible set of integers whose diameter
(largest minus smallest element) is at most `W`. Make the set **as large as possible**.

## Input (stdin)
A single line with one integer `W` (the window width).

## Output (stdout)
Whitespace-separated **distinct non-negative integers**, the elements of your set `H`.
They may span multiple lines. Requirements:
- every element satisfies `0 <= h <= 3*W`;
- all elements are distinct;
- `max(H) - min(H) <= W` (the set fits in a window of width `W`);
- `H` is admissible (see above).

## Feasibility
Your output is feasible iff it is non-empty, all elements are distinct integers in
`[0, 3*W]`, the diameter is at most `W`, and the set is admissible. Only primes
`p <= |H|` can ever be fully covered, so admissibility is decided by checking those
primes. Any violation scores `0`.

## Objective (maximize)
The score is the cardinality `|H|`, normalized by an internal reference construction
`B` built by the grader (a coarse double-residue sieve). With `F = |H|`:
`Ratio = min(1000, 100 * F / B) / 1000`. Reproducing the reference gives about `0.1`;
a genuinely dense admissible set scores much higher, and the true maximum is unknown,
so the top of the scale is left open.

## Constraints
- `100 <= W <= 6000`.
- Deterministic scoring; no randomness, timing, or hardware is involved.

## Example (worked score)
Suppose `W = 10` and you output `2 6 8 12` (an *illustrative shape only*).
Diameter `12 - 2 = 10 <= 10`, elements distinct and in range. Admissibility: mod 2 the
residues are `{0}` (class 1 uncovered — ok); mod 3 the residues are `{2,0,2,0}={0,2}`
(class 1 uncovered — ok). No prime `p <= 4` is fully covered, so the set is admissible
with `F = 4`. If the grader's reference for this window had size `B = 4`, the ratio
would be `min(1000, 100*4/4)/1000 = 0.1`.
