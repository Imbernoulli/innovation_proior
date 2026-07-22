# Sealed Corridors: The Assembly Insertion Order

## Problem

A rectangular assembly bay is an integer grid of `W x H` cells. `n` parts
must be installed into it one at a time, in an order you choose. Part `i`
has:

- a **footprint**: an axis-aligned rectangle of cells it will permanently
  occupy once installed;
- an **access corridor**: a separate axis-aligned rectangle of cells that
  must be *entirely empty* at the moment you attempt to install the part
  (its clear channel of approach);
- a **value** `value_i`.

You submit an **installation order** — a permutation of the `n` parts. The
bay starts empty. Parts are attempted strictly in your submitted order. When
part `i` is attempted:

- if any cell of its footprint **or** any cell of its corridor is already
  occupied by a previously installed part, the attempt **fails**: part `i`
  is permanently skipped, contributes no value, and its footprint cells are
  never occupied (it never entered the bay);
- otherwise the attempt **succeeds**: part `i`'s footprint cells become
  permanently occupied (irreversible commit) and `value_i` is added to your
  score.

There is no retrying — a failed part is gone for good, and an installed
part can never be moved or removed. Because later corridors can be
sealed by *any* footprint installed so far, the state that matters at each
step is the running union of all previously installed footprints: this is
why the effect of an early insertion must be propagated forward through the
whole remaining sequence, not judged one pair at a time.

## Input (stdin)

```
n W H
value_1 fx_1 fy_1 fw_1 fh_1 cx_1 cy_1 cw_1 ch_1
...
value_n fx_n fy_n fw_n fh_n cx_n cy_n cw_n ch_n
```

Part `i`'s footprint is the rectangle `[fx_i, fx_i+fw_i) x [fy_i, fy_i+fh_i)`;
its corridor is `[cx_i, cx_i+cw_i) x [cy_i, cy_i+ch_i)`. All rectangles lie
within `[0,W) x [0,H)`. A part's own corridor never overlaps its own
footprint. `1 <= n <= 20`, `1 <= value_i <= 200`.

## Output (stdout)

`n` space-separated integers: a permutation of `0..n-1`, the order in which
you attempt to install the parts (0-indexed by input order).

## Feasibility

Your output must be a permutation of exactly `0..n-1` (no repeats, no
omissions, no out-of-range or non-integer tokens). Any violation scores
`0`. There is no other feasibility requirement — every permutation is a
"legal attempt sequence"; the *quality* of the sequence is what varies.

## Objective

Maximize the total value of parts that actually succeed under the
simulation described above.

## Scoring

The checker simulates your order exactly as described and computes your
total `F`. It also simulates the plain index order `0,1,...,n-1` itself to
get a reference total `B` (always positive, since the very first part
attempted in any order can never be blocked). Your ratio is
`min(1000, 100*F/B) / 1000`, printed as `Ratio: <float>`. Beating the
reference order substantially pushes the ratio toward and above `0.1`;
finding an order that rescues parts the reference order loses (or vice
versa, avoiding losses the reference order stumbles into) pushes it higher,
capped at `1.0`.

## Example (illustrative shape only, not a real test)

3 parts, `W=6, H=1`: part 0 `value=10, footprint=[0,1), corridor=[1,4)`;
part 1 `value=5, footprint=[2,3), corridor=[4,6)`; part 2 `value=8,
footprint=[4,5), corridor=[0,1)`. Order `[1,2,0]`: part 1 installs
(footprint `{2}`, corridor `{4,5}` clear) -> occupied `{2}`, score 5. Part 2:
corridor `{0}` clear, footprint `{4}` clear -> installs, occupied `{2,4}`,
score 13. Part 0: corridor `{1,2,3}` contains occupied cell `2` -> fails.
Total `F=13`. Order `[0,1,2]` instead installs part 0 first (score 10),
then part 1's corridor `{4,5}` is clear (score 15), then part 2's footprint
`{4}` is now occupied -> fails: total `13` also, but for a different reason
— illustrating that *which* parts survive depends entirely on commit order,
not just on which parts you'd "want".

## Constraints

Time limit 5s, memory 512MB per test. `n <= 16`, `W,H <= 40`.
