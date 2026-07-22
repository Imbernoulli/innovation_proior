# Cursor-Biased Scroll: Finger Anchors Across Jump Epochs

## Problem

A scribe works a scroll of `N` positions (`1..N`) pinned flat on the desk --
it never rolls up, so a position never moves once written. Over the day the
scribe's quill touches the scroll `M` times, at absolute positions
`p_1, ..., p_M` (read off the trace in the input). Touches cluster tightly
for a while (finishing a paragraph) before the eye jumps somewhere else on
the scroll -- a burst near one spot, a jump, a burst near another, with
burst *lengths* varying wildly and some jumps *returning* to a spot visited
earlier.

The desk has a fixed **central spindle** at `C0 = floor((N+1)/2)` that
services any touch for free (no setup cost) but slowly: reaching position
`p` from the spindle costs `hop(C0, p) = bitlength(|C0-p|) + 1` (the
finger-search-tree property: cost grows with *distance spanned*, not `N`).

You may also plant up to `F` **finger bookmarks** (ids `1..F`) into the
scroll. Each starts unset. A **relocation** moves bookmark `i` to position
`q`, taking effect immediately before some touch `t` is serviced; it costs
`reloc(old, q) = floor(|old - q| / 4) + 2`, where `old` is `C0` if the
bookmark has never been placed. This is the cost of walking a ribbon across
the scroll to the new spot -- **linear** in distance, unlike the spindle's
log-cost hops. A relocated bookmark stays exactly where you left it until
relocated again; it does **not** silently follow the cursor, so a sudden
jump strands it.

Each touch `t` is serviced by whichever anchor (the spindle, or any
currently-placed bookmark) is cheapest at that moment:
`cost(t) = min( hop(C0, p_t), min_{i placed} hop(pos_i, p_t) )`.

Your total cost is the sum of every relocation's cost plus every touch's
service cost. **Minimize it.**

The exact epoch lengths, spreads, and how often the cursor revisits an
earlier hub are all baked into the trace -- read the data; do not assume a
uniform pattern.

## Input (stdin)
```
N M F
p_1
p_2
...
p_M
```
`1 <= N <= 200000`, `1 <= M <= 15000`, `1 <= F <= 12`, `1 <= p_t <= N`.

## Output (stdout)
```
K
t_1 i_1 q_1
t_2 i_2 q_2
...
t_K i_K q_K
```
`K` is the number of relocation events (`0 <= K <= 5000`). Each event
`(t, i, q)` means: immediately before touch `t` is serviced, move bookmark
`i` to position `q`. Events may appear in any order in the output; each
distinct `(t, i)` pair may appear **at most once**.

## Feasibility
- The first token is `K`, followed by exactly `3*K` more tokens -- no
  missing or trailing tokens.
- Every `t` in `[1, M]`, every `i` in `[1, F]`, every `q` in `[1, N]`.
- No two events share the same `(t, i)` pair.
- All tokens must be finite integers (`nan`/`inf`/non-numeric -> infeasible).
Any violation scores `Ratio: 0.0`.

## Objective (minimize)
`Total = sum of all reloc(old, q) over applied events + sum over t of
cost(t)` as defined above, replayed in touch order `t = 1..M` (events
scheduled for `t` apply before touch `t` is serviced).

## Scoring
The checker builds its own reference: never place any bookmark, service
every touch from the spindle alone. That gives a baseline cost `B`. Your
score is `Ratio = min(1.0, 0.1 * B / Total)` -- matching the baseline scores
`0.1`; a construction ten times cheaper than the baseline saturates at
`1.0`.

## Constraints
Time limit 5s, memory 512MB. `K <= 5000`, output tokens `<= 20000`.

## Example (illustrative form only, not a real test case)
`N=100, F=1`, spindle `C0=50`. A single isolated touch at `p=90` costs
`hop(50,90)=7` from the spindle; relocating there instead costs
`reloc(50,90)=12` plus `hop(90,90)=1` = `13`, *worse* -- one touch never
repays a relocation. Now a burst of four touches `90,91,92,93`: from the
spindle that is `4*7=28`; relocating bookmark `1` to `91` first costs
`reloc(50,91)=12`, then four cheap hops (`~2` each) total `~20` -- already
a net win. Longer, tighter bursts win by more; real instances mix many
bursts of very different lengths and some repeated hubs, which is the
whole game.
