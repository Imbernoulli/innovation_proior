# GC-Balanced DNA Barcode Packing (Constant-Weight Codes)

## Problem
In DNA data-storage systems, short synthetic strands are prefixed with **barcodes** that
must be (a) *GC-balanced* — a fixed number of G/C bases so every barcode melts and amplifies
at nearly the same temperature — and (b) *mutually far apart* in Hamming distance, so that
substitution errors introduced during synthesis and sequencing cannot turn one barcode into
another.

Model each barcode as a binary string of length `n` where a `1` marks a **strong (G or C)**
position and a `0` marks a **weak (A or T)** position. "GC-balanced with content `w`" means the
string has **exactly `w` ones**. Reliable demultiplexing requires that any two distinct barcodes
differ in at least `d` positions (minimum Hamming distance `d`).

Your job: build as **large** a set of usable barcodes as possible.

## Input (stdin)
A single line with three integers:
```
n w d
```
`n` = barcode length, `w` = required GC content (weight), `d` = minimum Hamming distance
(`d` is even and `d <= 2w`).

## Output (stdout)
Print your barcode set, one binary string per line (whitespace-separated tokens are also
accepted). Each token must be a string of exactly `n` characters over `{0,1}`. Order does not
matter; you may optionally omit any count line — only the binary strings are read.

## Feasibility
A submission is valid iff **every** token:
- has length exactly `n` and uses only characters `0`/`1`;
- has Hamming weight exactly `w` (GC-balanced);
- is distinct from all others; and
- the whole set is pairwise at Hamming distance `>= d`.

Any violation (bad character, wrong length, wrong weight, duplicate, or a too-close pair)
scores **0**. Because all words share weight `w`, the Hamming distance between two words equals
`2w - 2*|overlap|`, so `distance >= d` is equivalent to `overlap <= w - d/2`.

## Objective (maximize)
The number of valid barcodes in your set.

## Scoring
Let `F` be the number of valid barcodes you submit and let
`B = floor(n / w)` be the size of the trivial "disjoint GC-block" construction (partition the
positions into blocks of `w` consecutive strong bases; these blocks are pairwise disjoint, hence
at distance `2w >= d`). The score is
```
Ratio = min(1000, 100 * F / B) / 1000
```
so reproducing the trivial construction scores `0.1`, and reaching 10x the baseline caps at `1.0`.
The maximum achievable packing is **not known** in closed form for these sizes — many strategies
(algebraic designs, greedy packing, local search) are viable and none is provably optimal.

## Constraints
- `15 <= n <= 45`, `4 <= w <= 7`, `d = 2w - 2` (even, `<= 2w`).
- At most `5000` barcodes are read from your output.

## Example
For `n=8, w=3, d=4` (so `overlap <= 1`), the set
```
11100000
00011100
00000111
10010010
```
is valid: each has weight 3 and every pair overlaps in at most one position (distance >= 4).
Here `B = floor(8/3) = 2`, so `F = 4` gives `Ratio = 100*4/2/1000 = 0.2`. (This is an
illustrative instance only; the graded instances are larger.)
