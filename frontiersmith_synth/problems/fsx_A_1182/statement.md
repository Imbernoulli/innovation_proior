# Patient Zero — Backtracking an Outbreak to Its Index Case

An outbreak spread over a known **contact network** of `N` people (nodes 0..N-1)
and `M` contact edges. Each edge `(u, v)` carries an integer weight `w` (1..9):
the per-round transmission probability is `p = w / 10`.

The epidemic is a discrete-time **SI cascade** (susceptible → infected, no
recovery), started by one hidden **index case** `s0`. In round `t = 1..T`,
every currently-infected node independently attempts to infect each of its
still-susceptible neighbours; the attempt across edge `(u, v)` succeeds with
probability `p_uv`, independently every round. After exactly `T` rounds the
epidemiologists take **one snapshot**: they learn *which* nodes are infected
right now — nothing about *when* each of them was infected, and nothing about
who infected whom. `s0` itself is never revealed.

Your job: given the network, `T`, and the infected set, hand back a
**plausibility weight** for every infected node — how likely each one is to be
the index case.

## Input (stdin)
```
testId
N M
u_1 v_1 w_1
...
u_M v_M w_M
T
K
c_1 c_2 ... c_K
```
`c_1 < ... < c_K` are the infected node ids (this is the *only* place the
infected set appears — `s0` is one of them, but which one is for you to find).

## Output (stdout)
```
K'
w_1 w_2 ... w_K'
```
`K'` must equal `K`. `w_1..w_K'` are non-negative reals, in the SAME order as
`c_1..c_K` in the input — `w_i` is your plausibility weight for candidate
`c_i`. Weights need not sum to 1; the checker normalizes them.

## Feasibility
`K'` must equal `K`; every weight must be finite and `>= 0`; the weights must
not all be zero. Any violation scores 0.

## Objective (maximize)
The checker normalizes your weights into a distribution `p_i = w_i / sum(w)`
and reads off `F = p_{s0}`, the probability mass you placed on the TRUE
(hidden) index case. Score is `F` compared against the mass a uniform "no
idea" guess would place on `s0` (`1/K`): a uniform submission scores ~0.1,
and concentrating far more than uniform mass onto the true source climbs
toward — but never all the way to — 1.

## Why this is hard
The obvious move is to rank infected nodes by **contact-network degree** —
"the best-connected sick person is probably patient zero." This is a trap.
A hub that bridges many otherwise-separate communities gets infected early
from *whichever* direction the outbreak actually started in, simply because
it's reachable from everywhere — that says nothing about where the outbreak
*began*. A true index case, by contrast, has had the whole `T` rounds to
convert its own neighbourhood, so its immediate contacts should be almost
entirely infected, while a mere conduit like the hub typically still has many
untouched susceptible contacts in the directions the outbreak never reached.

Two structural facts you can actually compute from the input are worth more
than raw degree: (1) — a **necessary feasibility check** — a candidate `c`
cannot be the source if some infected node is more than `T` hops away from
`c` in the contact graph (infection can move at most one hop per round); (2)
among candidates that pass that check, prefer ones whose `T`-hop reachable
footprint is a *tight* fit around the infected set rather than a footprint
that "predicts" many more people should be sick than actually are — a hub's
wide reach in all directions makes its footprint far larger than the outbreak
itself, while a genuine source's footprint tends to match it closely.

## Example (illustrative shape only)
A 5-node path 0-1-2-3-4 with all `p=0.7`, `T=1`, infected `{1,2,3}`. Degree
favours node 2 (degree 2) over 1 or 3. But every candidate here is 1 hop from
the whole infected set, so degree alone under-uses the structure — real
instances hide a much sharper asymmetry between a well-connected relay and
the true origin.

## Constraints
`N` up to a few hundred, `T` up to ~10. Time 5s, memory 512MB. Fully
deterministic scoring.
