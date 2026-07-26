# Quarry Foreman: Carving a Cliff Face into Liftable Slabs

A quarry foreman must cut a cliff face, modeled as the line **[0, L)**, into
contiguous **slabs** so a crane can lift each one out separately. You choose
the cut positions; the two ends 0 and L are always slab boundaries.

Each meter **x** of rock has a posted **hardness** reading `h[x]` — a natural
per-meter difficulty signal (given in the input, 1–20). It is tempting to cut
slabs of roughly *equal hardness mass* so every lift is comparably hard. That
signal is a distraction from what actually drives cost.

Before cutting begins, **Q inspection passes** were already scheduled: pass
*i* re-surveys the stretch **[l_i, r_i)** and must re-anchor its scaffolding
on **every slab that stretch overlaps**, at a posted price `w_i` per slab
touched. A wide pass that straddles many slab boundaries is expensive to
re-anchor; a pass that lands cleanly inside one slab is cheap.

Independently, mobilizing the crane for a slab of length `size` costs
`BASE + size^1.5` (a fixed real-number exponent 1.5 for every instance;
`BASE` is posted in the input). A single giant slab is disproportionately
costly to lift (the 1.5 power), but every additional cut also pays a fresh
`BASE` — so neither "leave it whole" nor "cut everywhere" is free.

**Total cost** `F = BUILD + TOUCH`, where

  `BUILD = sum over slabs of ( BASE + size^1.5 )`
  `TOUCH = sum over passes i of  w_i * (number of slabs [l_i, r_i) overlaps)`

**Minimize F.**

## Input (stdin)

```
L Q BASE
h_0 h_1 ... h_{L-1}
l_1 r_1 w_1
...
l_Q r_Q w_Q
```

All values are integers. `2 <= L`, `1 <= Q`, `1 <= BASE`, `1 <= h[x] <= 20`,
`0 <= l_i < r_i <= L`, `1 <= w_i`.

## Output (stdout)

```
m
b_1 b_2 ... b_m
```

`m` interior cut positions, strictly increasing integers with `0 < b_j < L`.
`m = 0` means no cuts (one slab); then the second line is empty / omitted —
the checker reads exactly `m` further tokens regardless of line breaks. Any
such cut set is feasible; only its cost is graded.

## Scoring

The checker computes `F` exactly from your cuts, plus the baseline `B` = `F`
for the single-slab (`m=0`, no cuts) construction, and reports

  `Ratio = min(1.0, 0.1 * B / F)`

Lower cost ⇒ higher ratio. The no-cut construction itself scores ≈ 0.1.

## Structure you can exploit

A slab boundary `b` only ever hurts you through the passes that straddle it:
placing a cut at `b` costs exactly `pen(b) = sum of w_i over passes with
l_i < b < r_i`, **independent of any other cut you place** (it doesn't
matter how many other cuts exist elsewhere). So the touch side of the
objective is a simple per-position penalty, while the build side rewards
matching slab boundaries to where these penalties are *small* or *absent* —
i.e. to the passes' own endpoints, not to the hardness profile. The posted
`h[]` array's local extrema are unrelated to where passes' endpoints
cluster; balancing hardness mass gives you a segment count that may be
sound, but places every cut at the wrong location. The passes' endpoints
themselves cluster at multiple scales — some widely separated, a few tightly
bunched inside one narrow stretch — so a single uniform cut spacing is never
right everywhere at once.

## Constraints

`L <= 17000`, `Q <= 200`. Time limit 5 s, memory 512 MB. Scoring is exact
floating point (fixed exponent 1.5) and deterministic.

## Example

`L=12`, passes `(2,10,w=30)` and `(6,8,w=30)`, `BASE=3`.
No cuts: `BUILD=3+12^1.5=44.6`, `TOUCH=60`, `F=104.6` → Ratio 0.1000.
Cut at `b=6` straddles pass 1 (`pen(6)=30`; pass 2's own edge sits exactly
on 6, so it does *not* count): `BUILD=2*(3+6^1.5)=35.4`, `TOUCH=90`,
`F=125.4` — *worse*, despite the smaller build cost, because the touch
penalty is heavy. Cut at `b=2` instead — exactly on pass 1's own left edge,
so `pen(2)=0`: `BUILD=(3+2^1.5)+(3+10^1.5)=40.5`, `TOUCH=60` (unchanged),
`F=100.5` → Ratio 0.1041, a pure win. A cut only pays for itself when it
lands where the *passes' own endpoints* leave a gap — not wherever the
hardness happens to balance.
