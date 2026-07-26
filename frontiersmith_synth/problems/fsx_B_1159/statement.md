# Harbor Pilot and the Warm Berth

You are the harbor pilot for a quay of `HEAP` bytes, addressed `0 .. HEAP-1`,
managed as a classic **buddy system**: the whole quay starts as one free
block, a block may be split into two equal halves ("berths") to serve a
smaller ship, and two sibling halves may later be rejoined into their parent.
Ships request power-of-two length berths, are worked by tugboats while
moored, and eventually depart.

The quay is also divided into fixed `PAGE`-byte **sectors**. The harbor crew
keeps a roster of only the `TLB` most recently visited sectors (exact LRU).
Every tugboat visit to a moored ship needs its sector on the roster; if it
isn't there, the crew must re-brief on that sector before working (a "miss",
cost `1`), which also bumps the least-recently-used sector off the roster.
Splitting a berth or rejoining two sibling berths costs `0.01` in pilot
overhead each. Every ship's requested size is strictly smaller than `PAGE`,
so a berth never straddles two sectors.

You see the **entire trace** of events in advance and must decide, for every
arrival, which currently free berth to place the ship on, and for every
departure, whether to rejoin the freed berth with its sibling right away or
leave it alone for now.

## Input (stdin)

```
HEAP PAGE TLB N
```
followed by `N` lines, one event each, in order:
```
A id size      ship <id> arrives requesting a <size>-byte berth
F id           ship <id> departs (its berth is freed)
T id offset    a tugboat visits ship <id> at byte offset <offset> (0<=offset<size)
```
Ship ids are unique for the whole trace (never reused); `size` is always a
power of two.

## Output (stdout)

For every `A` and `F` event, in the SAME order they appear in the input
(skip `T` events -- they need no decision), print one line:
```
A id addr      place ship <id>'s berth starting at byte <addr>
F id flag      flag=1: rejoin with sibling now if it is free; flag=0: leave it
```
`addr` must be the start of a currently free berth (possibly reached by
implied splits) of size >= the ship's requested size, aligned to that size.

## Feasibility

`addr` must land inside a legal free block (after any necessary splits are
carried out top-down toward `addr`); the flag must be `0` or `1`; touches
target only currently-moored ships. Any violation scores `0` for that case.

## Objective & Scoring

Minimize `cost = misses + 0.01 * (splits + rejoins)` over the whole trace.
The checker also computes an internal baseline `B`: a naive construction
that, whenever several free berths of the right size exist, actively AVOIDS
whichever sector the crew's roster currently remembers (the opposite of the
locality this problem rewards), and ALWAYS rejoins on every departure. Your
score is
```
Ratio = min(1.0, 0.32 * B / max(1e-9, cost))
```
so matching that locality-blind baseline earns a small fraction of the
score; you must cut real crew re-briefings to climb from there.

## What makes it hard

Fragmentation is not the only cost: the same physical addresses get reused
by later ships, and whichever ship lands on a sector the crew's roster
already remembers gets worked for free. When several free berths of the
right size exist, the textbook rule (lowest address / best fit) ignores
which one sits on a warm sector. Rejoining a freed berth with its sibling
the instant it's free is "tidy", but if that exact size is requested again
soon, you've paid to merge and now must pay again to re-split -- and the
re-split, by convention, does not have to land back on the same warm
address, so the crew re-briefs for nothing. Deferring the rejoin keeps the
warm berth ready; rejoining too late instead fragments the quay.

## Example scoring

Say a solution achieves `misses=40, splits=30, rejoins=20` -> `cost = 40 +
0.01*50 = 40.5`. If the checker's naive baseline achieves `baseMisses=90,
baseSplits=10, baseRejoins=10` -> `B = 90.2`. Then
`Ratio = min(1, 0.32*90.2/40.5) ~= 0.713`.

## Constraints

`HEAP = 1048576`, `PAGE = 4096`, `TLB = 16`, `1 <= N <= 10000`. Time limit 5s,
memory 512MB. Scoring is deterministic.
