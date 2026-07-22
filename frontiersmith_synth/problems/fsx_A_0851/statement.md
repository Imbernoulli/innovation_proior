# Quantize a Stream Hiding on a Curve

## Setting
A telemetry stream delivers vectors in `R^D` in `R` rounds. Every point actually sits
(plus small noise) near a hidden 1-D curve embedded through an unknown rotation of
`R^D` — the curve's own two "shape" coordinates are mixed into all `D` ambient axes, so
no single ambient axis reveals it. Which stretch of the curve is currently active
*drifts* from round to round: early rounds sample one region, later rounds another.

**Your program is run once PER ROUND, each time as a fresh process that knows nothing
of earlier rounds' raw data.** On round `r` you receive only that round's own new
points, plus the codebook you yourself returned last round (`null` on round 0) — that
returned codebook is your *only* memory; you must decide everything else about that
round from what you're given right now.

You reconstruct every point of the current round with one of `K` codewords. There is a
persistent notion of codeword **slot**: slot `k` this round is compared to slot `k`
last round when metering movement. The total Euclidean movement summed over all `K`
slots from last round's codebook to this round's codebook may not exceed
`move_budget`. Re-deriving a codebook that only fits the round you can currently see,
and nudging every slot a little toward it, quickly runs the budget dry while leaving
every slot still slightly behind the drift. Recognizing which few slots the drift has
left stranded far off-curve, and fully relocating just those (leaving well-placed slots
untouched), makes much better use of the same movement budget.

## Input (stdin, one JSON object per round)
```
{"instance": str, "r": int, "R": int, "D": int, "K": int,
 "points": [[x_0..x_{D-1}], ...],     # ONLY this round's new points
 "prev_codebook": [[..K rows..]] or null,   # null iff r == 0
 "move_budget": float or null}        # null iff r == 0 (free initial placement);
                                        # else caps total per-slot movement from
                                        # prev_codebook to this round's codebook
```

## Output (stdout, one JSON object per round)
```
{"codebook": [[c_0..c_{D-1}], ... K rows],
 "assign":   [k_0 ... k_{n-1}]}   # codeword index (0<=k_i<K) for each input point, in order
```

## Feasibility
Score `0.0` on the WHOLE instance unless, on EVERY round: `codebook` has exactly `K`
length-`D` lists of finite numbers with absolute value `<= 1000`; `assign` has one
integer in `[0, K)` per input point; and (when `r >= 1`) the sum over `k` of the
Euclidean distance between this round's `codebook[k]` and last round's `codebook[k]` is
`<= move_budget` (a `1e-6` tolerance is allowed).

## Objective (minimize)
Reconstruction error for a point is the squared Euclidean distance between it and the
codeword it was assigned to (using that round's own committed codebook). Let `F` be the
mean of this over all points pooled across all rounds (LOWER is better).

## Scoring
```
F_base = mean squared error of the round-0 mean, repeated K times and never revisited
         again (an honestly-online "commit once, never adapt" reference)
LB     = 0.8 * (a per-round-INDEPENDENT best-effort K-means cost computed with FULL
         knowledge of that round's own points -- something no online policy has in
         advance -- ignoring the persistent-slot identity and movement budget entirely)
Ratio  = clamp( 0.1 + 0.9 * (F_base - F) / (F_base - LB), 0, 1 )
```
The round-0-mean-forever codebook scores exactly `0.1`. `LB` assumes foreknowledge no
online candidate ever receives (and is deflated by 20% besides), so it is not reachable
by any admissible submission — there is always headroom above any single strategy. The
final score is the mean of `Ratio` over 10 fixed, seeded instances.

## Constraints
`1 <= n_r <= 300` new points per round; `2 <= K <= 20`; `2 <= R <= 8`; `4 <= D <= 10`;
time limit 20s per round-call, memory 512m.

## Example (worked, small)
`D=2, K=1`. Round 0: `points=[[0,0],[0,0.2]]`, `prev_codebook=null` → reply
`{"codebook":[[0,0.1]], "assign":[0,0]}` (perfect fit). Round 1:
`points=[[3,0],[3,0.2]]`, `prev_codebook=[[0,0.1]]`, `move_budget=5` → moving the one
slot the `3` units needed, `{"codebook":[[3,0.1]], "assign":[0,0]}`, reconstructs both
points almost exactly. Leaving the codebook at `[[0,0.1]]` (zero movement) instead
costs squared error `~9` per point — movement was available and cheap, but unused.
