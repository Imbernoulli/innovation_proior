# Archivist's Compaction Cadence

## Problem

An archive receives **N** boxes of letters, one at a time, interleaved with
**M** reading-room requests. Box `i` holds `s_i` letters and covers a
*catalogue range* `[lo_i, hi_i]` — the alphabetical span of names it might
contain. A reading-room request names a catalogue key `q`; to answer it the
archivist must pull down and check **every currently un-merged box whose
range could contain `q`** — one probe per box that overlaps the key, because
without consolidating them the archivist cannot tell which one actually has
the letter.

Between arrivals, at any moment you choose, you may **fully re-file** a
contiguous run of currently-separate boxes into one: pick boxes that were
delivered consecutively and are still all separate, and merge them. This
costs one unit of effort per letter in every box you touch — even letters
that were already re-filed together before (re-filing always rewrites
everything you pick up again). After the merge, the new box's range is the
union of its parts', and future requests only probe it once.

You know the **entire** delivery-and-request timeline in advance. Choose when
(and what) to merge to minimize the **total cost**: effort spent re-filing,
plus probes spent answering requests.

## Input (stdin)
```
N M
s_1 lo_1 hi_1
...
s_N lo_N hi_N
T
ev_1
...
ev_T
```
`T = N+M`. Each `ev` line is either `I` (the next box, in delivery order,
arrives now) or `L q` (a request for catalogue key `q` happens now), listed
in chronological order; exactly `N` `I` lines and `M` `L` lines appear.
`1 <= N <= 30`, `1 <= M <= 200`, sizes and keys are positive integers
`<= 10^6`.

## Output (stdout)
```
K
gap_1 first_1 last_1
...
gap_K first_K last_K
```
Instruction `j` means: immediately after the first `gap_j` timeline events
have happened (`gap_j = 0` = before anything), fully merge the boxes with
original delivery-ids `first_j..last_j`. Gaps must be listed in
non-decreasing order.

## Feasibility
Every instruction must satisfy `0 <= gap_j <= T` and
`1 <= first_j <= last_j <= N`. At the moment `gap_j`, every id in
`[first_j, last_j]` must already have arrived, and the currently-alive boxes
(original boxes, or earlier merge results) must **exactly** tile
`[first_j, last_j]` — no box straddling the boundary, none missing. Any
violation, non-integer/non-finite token, or malformed output scores `0`.

## Objective & Scoring
Simulate the timeline under your plan. Each merge instruction adds
`sum(s_i for i in [first_j,last_j])` to the cost (bytes re-filed, charged in
full every time, even on already-merged bytes). Each request `L q` adds the
number of currently-alive boxes whose `[lo,hi]` contains `q`. Let `F` be your
total cost. The checker also searches, over the SAME instance, the cheapest
**fixed-cadence** policy — merge everything alive every `p` arrivals, for
every constant `p` from never to every-arrival — and calls its cost `B`
(this already picks the best constant shape for this instance, blind to
*when* requests actually land).
```
Ratio = min(1, 0.1 * B / F)
```
Matching the best fixed cadence scores `0.1`; a plan ten times cheaper than
it caps the score at `1.0`.

## Constraints
`1 <= N <= 30`, `1 <= M <= 200`, time limit 5s, memory 512MB. Deterministic:
same plan, same score, always.

## Worked Example (illustrative shape only)
Two boxes both covering key `5` (`s=10,[1,9]` and `s=10,[3,9]`), then three
requests for key `5`. Never merging costs `2+2+2=6`. Merging both boxes first
(gap after both arrive) costs `20` rewritten plus `1` per request `=23` —
worse here, because three requests don't repay a size-`20` rewrite. With many
more requests, or cheaper boxes, the same merge would pay off; reading the
actual counts, not a fixed rule, is what a good plan must do.
