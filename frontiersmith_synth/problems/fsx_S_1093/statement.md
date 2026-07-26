# The Foresighted Referee

## Problem

A single-elimination fitness league runs on a fixed script that is handed to the
referee **before the league starts**: `n` players register (`ENROLL`) over time with
secret, pairwise-distinct strength ratings, and at various points the referee must
`CUT` — publicly remove the currently weakest *registered-but-not-yet-cut* player.
Because the referee has the whole script (every rating, every event, in order) in
advance, they don't run a live scoreboard (heap, tournament bracket, ...) during the
league. Instead, before the first event happens, they file a written **certificate**:
a flat list of pairwise rating comparisons and roster bookkeeping entries. During the
league, an auditor replays only the *comparisons already on file* — never re-examining
ratings — to confirm every `CUT` announcement was inevitable.

A comparison "player `i` vs player `j`" is a fact: whichever of the two has the lower
rating is *provably* below the other, and provably below anything that fact chains to
transitively (if the file shows `i` below `j`, and elsewhere `j` below `k`, then `i` is
below `k` for free — no separate comparison needed). A `CUT` announcing player `a` is
**certified** only if, using comparisons already filed by that point, `a` can be shown
below *every other* player who is registered and not yet cut. Filing extra comparisons
that were never needed to certify anything just burns budget — the referee is paid to
file the minimal certificate that survives the audit for *this exact script*, not to
replay a general-purpose data structure that would work for every possible script.

Every registered player must also get exactly one roster entry (any order, anywhere in
the file) — this is fixed paperwork, unrelated to comparisons.

## Input (stdin)

```
n q
r_1 r_2 ... r_n
e_1 e_2 ... e_T          (T = n + q, tokens 'E' or 'C')
```
`r_k` is the rating of the `k`-th player to register (players are named `1..n` by
registration order — distinct positive integers, no ties). Reading the event tokens
left to right: the `i`-th `'E'` token is player `i` registering; a `'C'` token is a cut
that must remove the current minimum-rating player among those registered so far and
not yet cut. `q` = number of `'C'` tokens.

## Output (stdout)

```
a_1 a_2 ... a_q
<any number of lines, each "C i j" or "M i">
```
`a_1..a_q` are your claimed cut identities, in order (must exactly match the true
cuts). Each remaining line is either `C i j` (file the comparison between players `i`
and `j`, `i != j`) or `M i` (file player `i`'s roster entry). Every player `1..n` must
get **exactly one** `M` line, no more, no less.

## Feasibility

The claimed cut sequence must exactly equal the true one. Every `C`/`M` index must be
a valid player in `1..n`. Every player must be `M`-ed exactly once. For every cut,
letting `a` be the claimed identity and `L` the set of players registered-and-not-yet-
cut at that moment, `a` must reach every other member of `L` via the transitive closure
of the comparisons filed *anywhere* in your output (order within the file doesn't
matter — you already know the whole script). Any violation scores `Ratio: 0.0`.

## Objective (minimize)

`cost = (#C lines) + 0.25 * (#M lines)`. Fewer, better-targeted comparisons win.

## Scoring

The checker computes a baseline `B`: the comparison cost of blindly inserting the `n`
players, in registration order, one at a time into a fully re-searched sorted array via
ordinary binary search (a correct but search-happy strategy that never reuses the fact
that the whole script was already known). Then `Ratio = min(1, B / (10 * cost))`.

## Example (worked, not the intended difficulty)

`n=3,q=2`; ratings `5 1 9`; events `E E C E C`. True cuts: after players 1,2 register,
cut removes player 2 (rating 1); after player 3 registers, cut removes player 1
(rating 5). A valid certificate: `2 1` / `M 1` `M 2` `M 3` `C 2 1` `C 1 3` — one
comparison certifies the first cut (2 below 1), and with player 3's rating above
player 1's, the second cut is certified transitively through the same fact plus the
new comparison. Cost = 2 + 0.75 = 2.75.

## Constraints

`n` up to ~1300, time limit 5s, memory 512MB. Comparisons/roster-entries in the output
are capped generously (well above any sane solution's needs) purely to bound checker
runtime.
