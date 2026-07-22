# Jam Storm: Anticipatory Packet Routing

A data-center dispatcher must clear a **fully known traffic plan** through `K`
parallel links to a shared destination. The plan is a sequence of **rounds**;
round `r` delivers a batch of `n_r` packets, each with an integer weight, that
must simultaneously pick one of the `K` links. You see the *entire* plan (every
round's weights, every link's parameters) up front and must commit, in one
shot, to a link choice for every packet in every round.

**Congestion.** Each link `e` has base latency `t0_e`, capacity `cap_e`, and a
superlinear exponent `p_e >= 2`. If load `x` lands on link `e` in a round, the
per-unit latency is `L_e(x) = t0_e * (1 + (x/cap_e)^p_e)`, and that round's
delay cost from link `e` is `x * L_e(x)`.

**Carry-over queue.** Congestion does not vanish at the end of a round. Each
link keeps a backlog: whatever exceeds its capacity only partially drains —
`backlog_next = max(0, load - cap_e) * decay` (a fixed `decay` in `[0,1]` given
in the instance) — and starts the *next* round as extra load already sitting
on that link, on top of whatever you route there next. Overloading a link
doesn't just cost you this round; it haunts several rounds afterward.

**The trap.** The obvious policy is: each round, look at the link states as of
the *start* of that round (before this round's packets have landed anywhere),
pick whichever single link currently looks cheapest, and route the whole
round through it. This ignores that the round's packets are landing there
*simultaneously* — flooding a link that only looked cheap because it was
empty. Because latency is superlinear, that flood is disproportionately
costly, and the resulting backlog now makes that link look expensive next
round, so the same rule swings everyone to a *different* link — a herding /
ping-pong dynamic that compounds through the carried-over queue into real
congestion collapse. A policy that accounts for the load *it is itself*
concurrently committing avoids this.

## Candidate program contract

Standalone program: read ONE JSON object (public instance) from **stdin**,
write ONE JSON object (your answer) to **stdout**. Runs isolated; sees only
the public instance.

### Public instance (stdin)
```json
{
  "name": "trap_symmetric_burst",
  "k": 4,
  "edges": [{"t0": 2.0, "cap": 20.0, "p": 3}, ...],   // K entries
  "decay": 0.55,
  "rounds": [{"n": 24, "weights": [3,1,2,...]}, ...]  // n weights per round
}
```

### Answer (stdout)
```json
{"routes": [[0,2,1,0,...], ...]}   // one row per round, row length == n_r,
                                    // entries are link indices in [0, K)
```
Any shape/type violation (wrong row count, wrong row length, a non-integer or
out-of-range link index), a crash, a timeout, or non-JSON output scores that
instance `0.0`. There is no hard capacity limit — overloading is legal, it
just costs a lot via the dynamics above.

## Scoring (deterministic)

The evaluator replays your full routing plan through the dynamics above
(backlog starts at `0`, updates round by round exactly as described) to get
your total cost `obj` (sum of `load * L_e(load)` over all links and rounds).
It also computes, itself, an **unreachable continuous floor**: for each round
independently (queue reset to `0`, packets treated as infinitely divisible),
the exact convex-optimal split across links — found by equalizing each used
link's *marginal* delay `d/dx[x*L_e(x)]` — summed over rounds. No discrete,
queue-respecting policy can beat this floor.

```
r = clamp( 0.1 + 0.765 * (floor / max(obj, 1e-9)) ** 0.75, 0, 1 )
```

Matching the floor exactly would score `~0.865` (never `1.0` — the floor is
unreachable, so headroom stays open above any real policy); any *valid*
routing scores strictly above `0.1` and decays smoothly toward that floor as
`obj` grows arbitrarily large. Only an invalid answer (bad shape/type, a
crash, a timeout, or non-JSON output) scores exactly `0.0`. The `0.75`
exponent compresses the wide (orders-of-magnitude) cost range a herding
collapse can produce, so bad decisions are punished smoothly rather than
every collapse looking identically catastrophic.

**Ratio** = mean of `r` over 10 seeded instances (varying `K`, link
parameters, decay, and traffic burstiness — several designed so a single
current-cheapest link gets flooded by a round far larger than it can absorb;
some are larger held-out cases). **Vector** lists the per-instance scores.

## Suggested strategies

1. **Static cheapest link**: ignore load; always use the lowest-`t0` link.
2. **Myopic least-latency**: each round, route everyone to whichever link's
   *current* (pre-round) state looks cheapest.
3. **Marginal-cost water-filling**: within a round, insert packets one at a
   time onto whichever link has the lowest *marginal* delay increase right
   now (recomputed after each insertion), internalizing your own contribution.
4. **Lookahead**: reason across rounds about the queue you are about to leave
   behind, not just the round in front of you.
