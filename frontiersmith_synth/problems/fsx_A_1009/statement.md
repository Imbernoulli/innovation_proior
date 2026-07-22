# Deep-Space Array: Probing Against the Fault Sweep

## Problem
A deep-space probe carries `n` components (numbered `0..n-1`). At most one thing has
failed, but mission control cannot see inside the hull. You may run a batch of
diagnostic **probes**: each probe wires together a chosen subset of components and
reports `1` if **at least one** of the components it touches is faulty, `0` otherwise
(a probe cannot isolate which member fired — only whether any did).

You must design the probe matrix — which components go into which probe — using a
budget of at most `m` probes, each touching at most `cap` components (a probe's wiring
harness has limited pins).

Mission control ships you their **fault sweep**: every possible single-component fault
(`n` of them), plus a published list of `K` specific **double faults** — pairs of
components known (from historical failure correlation) to sometimes fail together.
Your matrix will be graded against exactly this sweep, never against faults outside it.

For a true fault `F` (a single component, or one of the `K` published pairs), the
probes you ran produce an observed outcome pattern (which probes fired). Given that
pattern, the **candidate fault set** is every fault in the sweep whose own outcome
pattern would look identical — the possibilities mission control cannot yet rule out.
A matrix that gives every fault in the sweep its own unique pattern makes every
candidate set a singleton (perfect diagnosis); a matrix that aliases patterns together
leaves ambiguity.

Minimize the **total ambiguity**: the sum, over every fault `F` in the sweep, of
`|candidate set of F|`.

## Input (stdin)
```
n m cap
K
a_1 b_1
a_2 b_2
...
a_K b_K
```
`n` components, probe budget `m`, per-probe capacity `cap`, then `K` published double
faults as component-index pairs (`0 <= a_i < b_i < n`, all pairs distinct).

## Output (stdout)
```
r
row_1
row_2
...
row_r
```
`r` (`0 <= r <= m`) is the number of probes you use; each `row_i` is a string of
exactly `n` characters over `{0,1}`, where character `j` is `1` iff probe `i` touches
component `j`.

## Feasibility
Valid iff **all** hold: `r` parses as an integer in `[0, m]`; exactly `r` further rows
follow; every row has length exactly `n` with characters only `0`/`1`; every row has at
most `cap` ones. Any violation scores `Ratio: 0.0`.

## Objective
For a fault `F`, its **fingerprint** is the bitwise OR, over its member component(s),
of that component's probe-membership bitmask. Two sweep faults are **aliased** iff
their fingerprints are identical. Minimize
`F = sum over faults f in sweep of |{f' in sweep : fingerprint(f') == fingerprint(f)}|`
(a fault always counts itself, so `F >= n + K`, achieved only when every fingerprint in
the sweep is unique).

## Scoring
The checker builds its own reference matrix: split the `n` components into `min(m, n)`
contiguous blocks and use one probe per block (block `i` = all components whose index
falls in that block's range). This never targets the published pairs — it is a coarse,
always-valid construction. Let `B` be this reference matrix's total ambiguity `F`.
With minimization normalization:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Matching the reference scores `Ratio = 0.1`; cutting ambiguity to a tenth of it caps at
`1.0`.

## Constraints
- `16 <= n <= 210`, `1 <= K <= 46` (all `< n*(n-1)/2`), `cap >= ceil(n/2) + 5`.
- `m` is set so that a full `ceil(log2 n)`-bit binary-splitting core always fits, with a
  small number of probes left over.
- Time limit 5s, memory 512m.

## Example
`n=8, m=6, cap=8, K=2`, pairs `(1,6)` and `(2,5)` (each pair is bit-complementary in
3-bit binary: `1=001,6=110` and `2=010,5=101`, so a plain 3-bit counting-code matrix
gives both pairs the fingerprint `111` and cannot tell them apart, nor from any other
component-pair whose codes also OR to `111`). The checker's own block-reference splits
`n=8` into `min(m,n)=6` size-`ceil(8/6)=2` blocks (`{0,1},{2,3},{4,5},{6,7}` plus two
empty probes); working out its ambiguity sum gives `B=18` (`Ratio 0.1` if matched
exactly, e.g. by reproducing that same block matrix).

A matrix using just the 3-bit counting core (3 probes, `r=3`, one probe per bit) keeps
every single fault unique but leaves both published pairs aliased to fingerprint `111`
and to each other: `F=16`, `Ratio=0.1125` — barely above the reference, because the
published doubles are still muddled. Keeping that same 3-bit core and spending 2 of
the 3 leftover probes on single-component pivot probes that separate `(1,6)` and
`(2,5)` from each other and from everything else drives every sweep fault to a unique
fingerprint: `F=n+K=10` (the minimum possible), `Ratio=0.18`.
