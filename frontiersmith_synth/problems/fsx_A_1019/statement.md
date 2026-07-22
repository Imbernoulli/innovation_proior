# Relief Bazaar: Endowments Before the Trading Rounds

## Problem

Before market day, a shipment of `G` relief goods must be handed out to `N`
households, which are grouped into `K` neighbor-communities. You choose the
**initial allocation** of the shipment. After you hand it out, a fixed,
already-published trading protocol runs on its own: households repeatedly try
bilateral barters (one unit of good `g` for one unit of good `h`), and a barter
only executes if it clears a friction fee — **both** households must gain at
least `eps` utils from it. Two households can ever trade directly only if they
belong to the **same community**; there is no other trade link, and none across
communities, ever. You do not design this protocol — it always runs the same
way. Your only lever is the initial allocation you hand out before it starts.

Household `i` holding `x` units of good `g` gets saturating happiness from it:
`w_{i,g} * min(x, cap_g)` — beyond `cap_g` units, extra copies of `g` are
worthless to `i`. Household `i`'s total utility is `u_i = 1 + sum_g w_{i,g} *
min(x_{i,g}, cap_g)` (the `+1` is a subsistence floor). The final score, judged
**after all trading finishes**, is the Nash Social Welfare: the geometric mean
of every household's `u_i`. Your goal is to choose the initial split that
leaves the *replayed* market in the best final state — not necessarily the
split that looks individually best before any trading happens.

## Input (stdin)

```
N G K R eps
cap_1 cap_2 ... cap_G
S_1 S_2 ... S_G
comm_0 w_{0,1} ... w_{0,G}
...
comm_{N-1} w_{N-1,1} ... w_{N-1,G}
```
`S_g` is the total shipment of good `g` (you must place all of it).
`comm_i` in `[0,K-1]` is household `i`'s community. `w_{i,g} >= 0` is how much
`i` values one saturating unit of good `g`. Households `i` and `j` can trade
directly iff `comm_i == comm_j` — every same-community pair is linked, no
cross-community pair ever is.

## Output (stdout)

`N` lines, each with `G` non-negative integers `x_{i,1} ... x_{i,G}`: your
chosen initial allocation. For every good `g`, `sum_i x_{i,g}` must equal
`S_g` exactly.

## Feasibility

All `N*G` tokens must be present, parse as finite integers, be non-negative,
and every good's column must sum exactly to its supply. Any violation scores 0.

## Trading replay (fixed — not yours to design)

For `R` rounds, visit every same-community pair `(i,j)` with `i<j` (households
in index order) and every ordered pair of distinct goods `(g,h)`, in index
order. If `i` currently holds `>=1` unit of `g` and `j` holds `>=1` unit of
`h`: let `i`'s utility change from giving up that unit of `g` and receiving a
unit of `h` be `dU_i`, and `j`'s utility change from the mirror trade be
`dU_j`. The swap executes iff **both** `dU_i >= eps` and `dU_j >= eps`. A
round that produces no swap ends the replay early.

## Scoring

Let `F` be the Nash Social Welfare after replaying your allocation. Let `B` be
the Nash Social Welfare after replaying a reference allocation: each good
split as evenly as possible among a fixed first batch of `round(0.85*N)`
households (by index; remainder to the lowest indices in the batch), leaving
the highest-indexed households out of the reference batch entirely. Score is
`min(1, F / (10*B))`. Matching the reference scores about `0.1`; a genuinely
better initial allocation scores higher, with headroom left above the best
known strategy.

## Example (illustrative shape only)

`N=2, G=2, K=1, cap=[3,3], S=[3,3]`, both households in community 0.
Household 0: `w=[9,1]` (loves good 0). Household 1: `w=[1,9]` (loves good 1).
A naive uniform split (`[1,1]`/`[2,2]`... any split ignoring preference) still
lets barter push units toward whoever values them, but starting the replay
from an allocation that already leans toward each household's favorite good
(e.g. `[2,1]`/`[1,2]`) reaches a higher final geometric mean, since fewer
friction-blocked trades are needed to get there.

## Constraints

`1<=N<=40`, `1<=G<=6`, `1<=K<=6`, `1<=R<=30`, `1<=cap_g<=8`, `0<=w_{i,g}<=60`,
`1<=eps<=5`. Time limit: 5s. Memory: 512MB.
