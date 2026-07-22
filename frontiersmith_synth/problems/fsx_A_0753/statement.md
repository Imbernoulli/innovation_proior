# Codebook Tuned to a Published Burst-Error Fingerprint

## Problem
Design a binary codebook of `K` distinct codewords, each `L` bits long, for a channel whose
noise is **not** arbitrary: the decoder will only ever be tested against a published, fixed
fingerprint of error patterns — a low-dimensional structured family, not every possible sparse
error vector.

The fingerprint consists of:
- **Single bursts**: every contiguous run of `b` flipped bits, for every length `b` in
  `[Bmin, Bmax]`, at every one of the `L` cyclic starting offsets (bursts wrap around the
  codeword).
- **Composite double-bursts**: for each published length pair `(b1, b2)`, a burst of length `b1`
  followed — after a fixed gap `G` of untouched bits — by a second burst of length `b2`, again
  swept over every cyclic offset.

Applying error pattern `e` to codeword `c` gives the received word `r = c XOR e` (bitwise). The
receiver decodes `r` to whichever codeword of your book is nearest under Hamming distance.

## Input (stdin)
```
L K
Bmin Bmax
G D
b1 b2        (repeated D times)
```
`D` is the number of published double-burst length pairs.

## Output (stdout)
`K` lines, each an `L`-character string over `{0,1}`: your codebook, one codeword per line.

## Feasibility
- exactly `K` lines, each exactly `L` characters, every character `0` or `1`;
- the `K` codewords are pairwise distinct.

Any violation scores `Ratio: 0.0`.

## Objective
For codeword `c_i` under attack pattern `e`, its **decode margin** is
`(distance from c_i XOR e to the nearest OTHER codeword) − (distance from c_i XOR e to c_i)`
(the second term always equals `popcount(e)`, the error weight). A positive margin means
`c_i XOR e` still decodes correctly with that much room to spare; zero or negative means the
error either ties or is decoded to the wrong codeword.

For each codeword, take its **worst-case margin over the entire published fingerprint** (single
bursts and composite double-bursts together). Maximize `F`, the **mean over all `K` codewords of
these worst-case margins**, normalized by `L` (so `F` is dimensionless, roughly in `[-1, 1]`).

Note that raw pairwise Hamming distance alone does not determine `F`: two codebooks with
identical minimum pairwise distance can have very different worst-case margins, because the
margin depends on how much a pairwise difference *overlaps* a specific swept pattern, not just
how large that difference is.

## Scoring
The checker builds its own reference codebook — a valid, spread-out ("interleaved") construction
that is deliberately loosely tuned rather than the tightest possible one — and evaluates the same
`F` on it, giving a baseline `B` (always positive on every test of this problem). With `F` your
submission's score:
```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = max(0.0, min(1.0, sc / 1000.0))
```
Reproducing the reference scores `Ratio = 0.1`. A codebook with a much better worst-case margin
scores higher, up to `1.0`. A codebook that lets some published pattern erase (or invert) a
pair's decode margin scores below `0.1`, down to `0.0`.

## Constraints
`6 <= K <= 16`, `40 <= L <= 112`, `2 <= Bmin <= Bmax <= 10`, `1 <= D <= 1`, `1 <= b1,b2 <= Bmax`,
`1 <= G <= Bmax`. Time limit 5s, memory 256MB.

## Example
`L=42, K=6, Bmin=2, Bmax=3, G=2`, one pair `(2,2)`. The reference codebook scores `B ≈ 0.0238`
(reproducing it exactly earns `Ratio 0.1`). A codebook dedicating one contiguous 14-bit block per
message-index-bit (maximizing raw pairwise Hamming distance, ignoring the fingerprint's shape)
reaches `F ≈ 0.1429`, `Ratio 0.6`. A codebook that instead spreads each message-index-bit across
a comb of positions with large, even spacing — so no swept burst, single or composite, ever lands
on more than one bit of any pairwise difference — reaches `F ≈ 0.1508`, `Ratio ≈ 0.633`: a further
gain purely from the *shape* of the differences, not from extra raw distance. On harder tests this
gap turns dramatic: once `Bmax` exceeds half a naive block's length, one swept burst can fall
entirely inside a block and erase that pair's margin, while a comb codebook of equal raw distance
keeps a comfortable positive margin.
