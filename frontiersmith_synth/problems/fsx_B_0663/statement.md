# Rumor Flood: Fastest-Spreading Couriers in the Matrix Kingdom

## Problem
The Matrix Kingdom's states are the elements of the group `SL(2, F_p)`: 2x2 integer
matrices with entries taken mod a prime `p` and determinant `1 (mod p)`. The calm
state is the identity `I`.

You must appoint `k` couriers, each a matrix `A_1, ..., A_k` in `SL(2, F_p)`. A rumor
sitting at state `X` can hop in one step to `X*A_i` or `X*A_i^{-1}` (any courier, run
forward or reversed) for any `i`. Starting the rumor at `I`, after at most `r` hops it
has flooded the set `Ball_r` of every state reachable within `r` hops. Your job: choose
the `k` couriers so `Ball_r` is as large as possible.

**The catch (subgroup-avoidance).** If your couriers all stay inside one proper
subgroup `H` of `SL(2, F_p)` — e.g. they pairwise commute, or they all preserve a
common line (an upper-triangular / Borel pattern), or together they only ever generate
a small cyclic/dihedral pattern — then `Ball_r` is trapped inside `H` forever: no matter
how large `r` is, the flood stalls almost immediately, because `H` itself is small and
polynomially-bounded in growth. `SL(2, F_p)`'s proper subgroups are fully classified
(cyclic, dihedral, Borel/upper-triangular, and a handful of small exceptional groups).
Fast flooding requires the couriers to jointly escape *every* one of them; once they
do, growth is close to the free-group ceiling `2k*(2k-1)^(r-1)` (no coincidental
short relations), which is far larger than anything a subgroup-trapped set can reach.

The kingdom's registry ships a **starter set** of `k` "suggested" couriers with the
input. It is always a *valid* (feasible) choice, but its flood quality varies wildly
from prime to prime — sometimes it is excellent, sometimes it is exactly the kind of
subgroup-trapped set described above. You may use it, ignore it, or replace only part
of it.

## Input (stdin)
```
p k r
a_1 b_1 c_1 d_1
...
a_k b_k c_k d_k
```
`p` is prime, `2 <= k <= 3`, `3 <= r <= 4`. The `k` following lines are the registry's
suggested couriers: each is a matrix `[[a_i,b_i],[c_i,d_i]]`, entries in `[0,p-1]`,
`a_i*d_i - b_i*c_i == 1 (mod p)`.

## Output (stdout)
```
a_1 b_1 c_1 d_1
...
a_k b_k c_k d_k
```
Exactly `k` lines, your own chosen couriers (integers; will be reduced mod `p`).

## Feasibility
Each of the `k` lines must contain exactly 4 integers, each with absolute value
`<= 10^9`; after reduction mod `p`, every matrix must satisfy
`a_i*d_i - b_i*c_i == 1 (mod p)`. Any violation (wrong line/token count, non-integer
token, non-finite value, failed determinant condition) scores `Ratio: 0.0`.

## Objective
Build the symmetric generating set `S = {A_1,...,A_k} U {A_1^-1,...,A_k^-1}`. Run BFS
from `I` in the Cayley graph of `<A_1,...,A_k>` under `S`, out to distance `r`. Let
`F` = number of distinct matrices reached (including `I`). Maximize `F`.

## Scoring
The checker also builds its own baseline flood `B`: `k` couriers that are all
upper-triangular unipotent shears (a fixed, always-feasible, always subgroup-trapped
construction — pairwise commuting, so its `Ball_r` grows only linearly in `r`), scored
with the same BFS. With maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the baseline scores `Ratio = 0.1`; a flood `10x` larger caps at `1.0`.

## Constraints
- `23 <= p <= 337`, `2 <= k <= 3`, `3 <= r <= 4`.
- Time limit 5s, memory 256m. 10 test cases.

## Example
Suppose `p=23, k=2, r=3` and the registry suggests `A_1=[[1,1],[1,2]]`,
`A_2=[[2,1],[1,1]]` (both determinant 1). Running BFS to distance 3 from `I` under
`{A_1,A_2,A_1^-1,A_2^-1}` happens to reach every one of the `53` distinct matrices the
free-group ceiling allows for `k=2,r=3` (no short relations occur) — an excellent,
subgroup-escaping choice. If instead you output two matrices that share the invariant
line `(1,0)` (e.g. two upper-triangular shears), the flood collapses to a handful of
states no matter how they're tuned — a trapped, low-scoring choice.
