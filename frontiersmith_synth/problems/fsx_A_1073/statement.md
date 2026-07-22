# Worst-Class Probe Panel

## Problem
You are designing a diagnostic probe panel for a reference genome of length `L` over
the alphabet `ACGT`, carrying `M` marker loci, each a window of length `W` starting at
a published position `pos_i`. A laboratory has released a panel of `n_mutants`
structured mutant genomes. Each mutant applies exactly ONE edit-pattern class to a
published subset of loci; loci outside that subset are byte-identical to the reference:

- **SNP-cluster** (class `0`): at every affected locus, flips the bases at a fixed,
  published set of offsets inside its `W`-window (each flip changes the base to a
  different one — never a no-op).
- **Segment-inversion** (class `1`): replaces an affected locus's whole `W`-window with
  its reverse complement.
- **Tandem-retile** (class `2`): replaces an affected locus's whole `W`-window with a
  published short repeat unit (taken from inside that same window) tiled to fill it.

You have a budget of `K` probes. Each probe is specified by a source coordinate `pos`
and an orientation: its *sequence* is `ref[pos..pos+P)` if orientation `F`, or the
reverse complement of that substring if orientation `R`. A probe is **anchored** to
locus `i` if `pos_i <= pos <= pos_i + (W-P)` (its source window lies fully inside that
locus's window); a probe anchored to no locus detects nothing. A probe anchored to
locus `i` **detects** that locus in a mutant genome if its sequence differs, in at most
`D` positions (Hamming distance), from the mutant's `[q, q+P)` window for *some*
`q` with `pos_i <= q <= pos_i + (W-P)` — i.e. it may bind anywhere inside that locus's
own window, not only at its own source coordinate (a segment-inversion moves where a
reverse-complement sequence actually occurs inside the window).

For every published mutant, its *coverage* is the fraction of the `M` loci detected by
your panel. Your score is the **minimum coverage over all published mutants** — one
badly-covered locus/edit-class combination sinks the whole panel, however well the rest
is covered.

## Input (stdin)
```
L M K P W D
<reference string, length L>
pos_1 pos_2 ... pos_M
snp_count
o_1 ... o_{snp_count}        SNP-cluster offsets, each in [0, W)
tandem_len tandem_start
n_mutants
class subset_size idx_1 ... idx_{subset_size}     one line per mutant, n_mutants lines
```
`class` is `0` (SNP-cluster), `1` (inversion) or `2` (tandem-retile); each `idx_j` is a
0-indexed marker-locus index in the subset that mutant stresses.

## Output (stdout)
```
T
pos_1 orient_1
...
pos_T orient_T
```
`1 <= T <= K`; each `orient` is the literal character `F` or `R`; each `pos_t` is the
reference start coordinate the probe is drawn from.

## Feasibility
An output is valid iff **all** hold:
- `T` parses as an integer with `1 <= T <= K`;
- exactly `T` `(pos, orient)` pairs follow, each `pos` an integer with `0 <= pos <= L-P`;
- each `orient` is exactly `F` or `R`.
Any violation (bad token, wrong count, out-of-range value, non-finite token) scores
`Ratio: 0.0`.

## Objective
Maximize `F = min` over the `n_mutants` published mutants of `(detected loci)/M`.

## Scoring
The checker also builds its own minimal panel `B`: one forward (`F`) probe at shift `0`
for only the first `C = max(2, M div 4)` marker loci, scored by the identical rule
above. Then, with maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the checker's minimal panel scores `Ratio = 0.1`.

## Constraints
`9 <= M <= 18`, `P = 20`, `W = 50`, `D = 2`. Time limit 5s, memory 512m.

## Example
With `P=20, W=50, D=2`: a locus untouched by a given mutant is detected (0 mismatches at
`q=pos`) by any probe anchored to it. A locus hit by segment-inversion is detected only
by an `R` probe — an `F` probe's sequence differs from the mutant's window at every `q`,
since the whole window's orientation is flipped, while some `R` probe's sequence (the
reverse complement of *some* sub-window of the original locus) exactly equals the
mutant's window at the mirrored `q`. A locus hit by SNP-cluster is detected by an `F`
probe only if that probe's own `[pos, pos+P)` sub-window avoids every published
SNP-cluster offset; whether shift `0`, shift `W-P`, or some other shift achieves that is
decided by the actual offset list in the input, not by anything visible in this
statement — you must read it and search. (Illustrative only: none of the above numbers
are the actual per-test offsets.)
