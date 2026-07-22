# Custom Pressing for a Look-Ahead Pickup Arm

## Problem

A record press stamps `N` audio cells (ids `0..N-1`) onto a disc with `T`
concentric **grooves** (ids `0..T-1`); each groove can physically hold at
most `cap` cells. **You choose which groove holds every cell** — that
choice is your only lever; you never touch the player.

A listener's **cue sheet** plays `M` cells in a fixed order
`Q[0], Q[1], ..., Q[M-1]` (cell ids, repeats allowed). The pickup arm is
*not* a strict jukebox: it has a **look-ahead buffer of depth `w`**. At every
step the buffer holds the oldest still-unplayed cues (up to `w` of them,
fewer only once the sheet is nearly exhausted). The arm always plays
**whichever buffered cue sits on the groove nearest its current position**
(ties broken by the earlier cue), moves to that groove, and the buffer is
refilled with the next cue from the sheet. This is a **fixed** rule — it is
not something you can tune — so the arm can only ever choose among cues
currently sitting in its buffer; a cue that keeps losing to nearer rivals
is never skipped outright, just delayed until it is nearest or is the
last cue left.

The **cost** of a pressing is the total groove-to-groove travel the arm
makes while working through the whole cue sheet this way (arm starts
parked at groove 0). **Minimize** this cost.

There is no known optimal layout: the arm's window lets it silently "fix"
locally scrambled cue order for free, so the best pressing is not the one
that minimizes travel for a *strict*, un-reordered playback — it is one
that sets up exactly the local groove neighbourhoods the fixed window rule
can sweep through cheaply, while still respecting the per-groove capacity.

## Input (stdin)
```
N T cap w M
Q[0] Q[1] ... Q[M-1]
```
All values are non-negative integers; `0 <= Q[i] < N`.

## Output (stdout)
Exactly `N` integers `trk[0..N-1]`, one groove id per cell (`trk[i]` is the
groove holding cell `i`). `trk[i]` must lie in `[0, T)`, and every groove
may be assigned to **at most `cap`** cells. Any other token count, a
non-finite/non-integer token, an out-of-range groove id, or a
capacity breach scores **0**.

## Scoring
The checker simulates the fixed window rule above on your `trk[]` against
the given `Q`, producing your total travel `F` (an exact non-negative
integer; lower is better). It also simulates the same rule on its own
**reference pressing** — cells packed by raw id order, `cap` cells per
groove, `trk_base[i] = min(T-1, i // cap)` — giving a baseline travel `B`.
Your score for this test is
```
score = min(1.0, 0.1 * B / F)
```
so a pressing that ties the reference scores `0.1`, and every 10x
reduction in travel below the reference raises the score by roughly
0.1x more, capped at `1.0` (the printed `Ratio:` is this value, already
in `[0,1]`). `B` is always positive (`cap < N` on every instance, so the
reference must move); `F` cannot legally reach 0 either, since every cell
is cued at least once and `cap < N` forbids parking them all on one
groove. The final score is the mean over 10 fixed, seeded instances of
increasing size.

## Feasibility
* Output has exactly `N` tokens, each a finite integer in `[0, T)`.
* No groove is assigned more than `cap` cells.
Violating either scores 0 for that test.

## Example (worked, illustrative shape only)
`N=6, T=3, cap=2, w=2, M=5`, `Q = [0, 3, 1, 4, 2]`, reference pressing
`trk_base = [0,0,1,1,2,2]` (cells `0,1`->groove 0; `2,3`->groove 1;
`4,5`->groove 2). Arm starts at groove 0.
* Buf `{c0=cell0@0, c1=cell3@1}` -> nearest c0 (dist 0); play, arm=0;
  refill c2=cell1@0.
* Buf `{c1=cell3@1, c2=cell1@0}` -> nearest c2 (dist 0); play, arm=0;
  refill c3=cell4@2.
* Buf `{c1=cell3@1, c3=cell4@2}` -> nearest c1 (dist 1); play, cost+=1,
  arm=1; refill c4=cell2@1.
* Buf `{c3=cell4@2, c4=cell2@1}` -> nearest c4 (dist 0); play, arm=1.
* Buf `{c3=cell4@2}` -> play, cost+=1, arm=2.

Total `B = 2`. Your `F` is this same simulation on your own `trk[]`;
`score = min(1, 0.1*B/F)` for this test.

## Constraints
`72 <= N <= 264`, `T` chosen with headroom above `ceil(N/cap)`,
`2 <= cap <= 4`, `6 <= w <= 11`, `M` up to a few thousand. Time limit 5s,
memory 512MB.
