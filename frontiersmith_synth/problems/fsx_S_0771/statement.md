# Cantilever Deck: Order-Constrained Truss Assembly

## Problem

You are erecting a truss bridge across a gap. Joints sit on a grid: horizontal
position `x` in `[0,W]`, height level `y` in `[0,H]`. A joint's id is
`y*(W+1)+x`. Joints with `y=0` (ids `0..W`) are **anchors**: pre-existing
foundations that are always present, weightless, and can absorb unlimited
weight/load.

The input lists `M` **candidate members** (struts/braces), each a pair of
joints `(u,v)` with an integer **capacity** `cap`. You choose a subset of
these members to actually build, and an **order** to build them in, one at a
time. You may not use any member, or connect any joints, that are not in the
candidate list.

Every joint with `y>=1` that touches at least one already-built member has
**self-weight** 1. At every moment during construction (after each member is
added), every such "erected" joint's self-weight must be routable to *some*
anchor using only the members built *so far*, without exceeding any built
member's capacity anywhere along the way — capacity can be shared: several
units of weight may cross the same member as long as their sum stays within
its capacity, and weight may split across multiple members. Formally: it must
be possible to simultaneously send 1 unit from every erected joint to some
anchor through the currently-built members (each member, used in either
direction, capped at `cap`) — this is an exact **maximum-flow** feasibility
check. If at any step this is impossible, the whole structure has collapsed
and your submission scores 0, no matter how good the final design would have
been.

The last input line gives `D` **deck joints** at height `H` that must be
reached to matter for scoring. Your **objective**, evaluated on the *final*
built structure only, is the maximum flow deliverable from a virtual source
touching every deck joint (unlimited supply) to a virtual sink touching every
anchor (unlimited capacity), through your final member set at full capacity.
This is the bridge's rateable load capacity. Building strong members is not
enough — a fine final design that has no valid gravity-safe construction
order is worthless, and a construction order that overloads a member before
its supporting partner exists is also worthless, even if that partner is
built one step later.

## Input (stdin)
```
W H M D
u_1 v_1 cap_1
...
u_M v_M cap_M
d_1 d_2 ... d_D
```
`M` candidate members are indexed `0..M-1` in the order listed. `1<=D<=4`.

## Output (stdout)
```
K
i_1
i_2
...
i_K
```
`K` = number of members you build (`0<=K<=M`). The following `K` lines list
the chosen member indices (each in `[0,M-1]`, all distinct, referencing the
input order), in the order you build them.

## Feasibility
- All indices in range, distinct, exactly `K` of them.
- At every prefix of the build order, every joint touching a built member
  (other than an anchor) must have its 1 unit of self-weight simultaneously
  routable to an anchor through the built members so far, without any member
  exceeding its capacity (checked via max-flow). Any violation -> score 0.

## Scoring
Let `F` be the max-flow objective of your final structure (deck source to
anchor sink). The checker independently builds a weak reference structure
(the union of each deck joint's own best single bottleneck-widest path to an
anchor) and computes its flow value `B > 0`. Score = `min(1000, 100*F/B)/1000`
(a structure matching the reference scores ~0.1; roughly 10x better caps at
1.0).

## Constraints
`2<=W<=6`, `1<=H<=3`, `M<=80`, time limit 5s, memory 512MB.

## Example (worked score, illustrative shape only)
Suppose `W=2,H=1`: anchors are ids 0,1,2; joint 3=(x=0,y=1), joint 4=(x=1,y=1),
joint 5=(x=2,y=1); deck = {3,4,5}. If member `(0,3,cap=3)` is the only one
built, only joint 3 is erected, self-weight 1<=3 OK, and final flow from
deck={3,4,5} to anchors is just 3 (only joint 3 is reachable). Building more
members that connect 4 and 5 to anchors, without overloading any of them
mid-construction, raises `F` further.
