# Where to Put the Error Checks: Checksum Placement Against a Bursty Channel

## Problem

A message has **M** bits, numbered `0..M-1`. You have a fixed budget of
**K** checksum bits to protect it: each checksum is a parity bit over
whichever subset of the M message positions you assign to it. You must
assign every message position to exactly one of the K checksums (this is
your **routing**). A checksum's parity flips — and *catches* the
corruption — exactly when an **odd** number of its assigned positions were
corrupted; if that count is even (including zero), that checksum stays
silent. The whole scheme catches a corruption event iff **at least one** of
your K checksums flips.

The channel corrupts the message in one of two ways, mixed according to a
published profile:

- **Independent mode**: a handful (1–3) of scattered, unrelated bit flips.
- **Burst mode**: a single contiguous run of **L** consecutive bits flipped
  (wrapping around position `M-1 -> 0` if it runs off the end), where L is
  drawn from a published discrete length distribution and the run's start
  position is unconstrained.

You do not see the individual corruption events — only the channel's
published profile (how often burst mode fires, and the weights on each
burst length). Your routing is scored against a large held-out set of
events sampled from that same published profile.

## Input (stdin)

```
M K SEED PBURST
NBL
L_1 W_1
...
L_NBL W_NBL
```

`M` = message length, `K` = checksum budget, `SEED` = an instance tag
(uninterpreted by you), `PBURST` = probability a corruption event is burst
mode (else independent mode). Then `NBL` lines of `(length, weight)` giving
the burst-length distribution (weight is a relative frequency, not a
probability). `120 <= M <= 456`, `K = 6`, `1 <= L < M`.

## Output (stdout)

Exactly **M** integers `g_0 g_1 ... g_{M-1}` (whitespace-separated,
line breaks don't matter), each in `[0, K-1]`: `g_i` is the checksum group
that message position `i` is routed to.

## Feasibility

Output must contain exactly M tokens, each a plain integer (no `nan`,
`inf`, or non-integer text) in `[0, K-1]`. Any violation scores `0`.

## Scoring

Let `F` = the fraction of held-out corruption events your routing catches.
Let `B` = the fraction caught by a single global parity bit (every position
routed to one checksum, the rest unused) — the checker's own do-nothing
reference. Your score is

```
Ratio = min(1.0, F / (10 * B))
```

Matching the baseline scores `0.1`. Routings that catch corruption at a
rate the baseline can't touch score higher, capped below `1.0` so there is
always room above a strong routing.

## Illustrative example (form only, hand-worked; M, K far smaller than real
instances)

`M=8, K=2`, two held-out events: a scattered flip at `{3}`, and a length-4
burst `{0,1,2,3}`.

- Baseline `[0,0,0,0,0,0,0,0]`: group 0 sees 1 (odd, caught) on event 1;
  sees 4 (even, missed) on event 2. `B = 1/2`.
- Contiguous blocks `[0,0,0,0,1,1,1,1]`: event 1 caught. Event 2: group 0
  sees 4, group 1 sees 0 — both even, **missed** (the burst sits inside one
  block).
- Interleaved `[0,1,0,1,0,1,0,1]`: event 1 caught. Event 2: group 0 sees
  `{0,2}`=2, group 1 sees `{1,3}`=2 — also both even here, since `L=4` is a
  multiple of `K=2` with an even quotient. (Real instances weight the
  length table so this coincidence is the exception, not the rule.)

## Structure you can exploit

Nothing forces your routing to follow the message's natural order. Whether
a corruption event's flipped positions land clustered in one group or split
across all K depends only on which positions you routed together — not on
how many checksums you have. The published length table tells you exactly
which run lengths the channel favors; a routing robust across that whole
table beats one that only looks correct on paper.

## Constraints

`120 <= M <= 456`, `K = 6`. Time limit 5 s, memory 512 MB. Scoring reads a
fixed, seeded held-out set and is bit-for-bit deterministic.
