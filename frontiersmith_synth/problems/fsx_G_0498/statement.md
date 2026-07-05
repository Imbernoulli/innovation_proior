# Deep-Space LDPC Batch Decoder

## Problem

A deep-space probe streams telemetry to a ground station using a fixed binary
**LDPC (low-density parity-check) block code** — the same family of codes standardized
by CCSDS for deep-space downlinks. The code is defined by a sparse **parity-check
matrix** `H` with `r` rows and `n` columns over GF(2): a word `x in {0,1}^n` is a valid
**codeword** iff every parity check is satisfied, i.e. `H x = 0` (all arithmetic mod 2).

During one downlink pass the station captures a **batch of `m` frames**. Each frame was a
codeword when it left the spacecraft, but the channel flipped some bits, so the received
frame is `y = c XOR e` for an unknown codeword `c` and an unknown error pattern `e`. Some
frames were hit by only a light error; others by a heavy noise burst.

Your job is the ground-station decoder: for **every** received frame, output a codeword.
A frame is **corrected** when the codeword you output is genuinely a codeword of the code
**and** it is close enough to the received frame — within Hamming distance `T` (the
downlink's declared decoding radius). Maximize the number of corrected frames.

Nearest-codeword decoding of a general linear code is NP-hard, so there is no known
efficient way to correct every correctable frame; better decoders simply correct more.

## Input (stdin)

```
n r m T
<r lines>   parity-check matrix H, one row per line; each row is a length-n binary string
<m lines>   received frames, one per line; each is a length-n binary string
```

In every binary string, character position `j` (0-indexed, left to right) holds
coordinate `j` of the vector. `2 <= n`, `r = n/2`, and `T` is the decoding radius.

## Output (stdout)

Exactly `m` whitespace-separated tokens (one per line is cleanest). The `i`-th token is a
**length-`n` binary string** — the codeword you decoded frame `i` to. Output nothing else
(no counts, no labels). You must emit one decoding for every frame, in the input order.

## Feasibility

The artifact is rejected (score `0`) unless it is **exactly `m` tokens**, each a string of
**exactly `n` characters, all `'0'` or `'1'`**. Wrong token count, wrong length,
non-binary characters, or `nan`/`inf` all score `0`. (A per-frame token that happens not
to be a codeword is allowed — that frame simply does not count as corrected.)

## Objective (maximize)

Let `F` be the number of frames `i` such that your token is a codeword (`H w = 0`) **and**
`Hamming(w, y_i) <= T`. Maximize `F`.

## Scoring

The checker builds its own trivial baseline `B` = the number of received frames that are
**already codewords** (correctable by doing nothing). Then

```
sc    = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```

Reproducing the baseline scores `Ratio ~= 0.1`. Scoring higher requires actually decoding
the noisy frames. `Ratio = 1.0` would need ten times the baseline count — far beyond any
known efficient decoder — so substantial headroom always remains.

## Constraints

- `2 <= n <= 300`, `r = n/2`, batch size `m` up to a few hundred, `T = 8`.
- Deterministic scoring, exact GF(2) / integer arithmetic only.

## Example

Suppose `n = 4`, `T = 1`, and `H` has the single row `1111` (so a word is a codeword iff
it has even weight). A received frame `1000` (odd weight) is not a codeword. Flipping its
one bit gives `0000`, a codeword at Hamming distance `1 <= T` — that frame is corrected.
The received frame `1110` has odd weight too; the nearest even-weight word `1111` is at
distance `1` and is also corrected. A frame like `1011` is already even weight, so echoing
it corrects it for free (this is what the baseline counts). Emitting `1111` for a frame
`0000` would be a valid codeword but at distance `4 > T`, so it would **not** count.
