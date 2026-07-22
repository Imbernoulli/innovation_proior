# Welding a Frame That Flinches at Every Joint

## Problem
A fabricator must weld together a planar frame: a graph on `N` joints
(nodes) and `M` struts (edges). Every strut `i` connects two joints
`u_i, w_i` and carries a fixed **pull rating** `eff_i` (a nonzero integer,
positive or negative — its sign already encodes which way its weld normal
points). Struts are welded one at a time, in an order the fabricator
chooses, and each weld is fired from one of two sides, a choice recorded as
`s_i in {+1,-1}`.

Every joint has a running **displacement** (a single scalar, initially 0)
that only grows through welds touching it. When strut `i = (u,w,eff)` is
welded with side `s`, BOTH endpoints shift immediately:
```
disp[u] += s * eff / (1 + stiffness(u))
disp[w] -= s * eff / (1 + stiffness(w))
```
`stiffness(v)` is the number of struts **already** welded (before this one)
anywhere inside `v`'s current connected component of *already-welded*
struts — i.e. the weld count accumulated by whatever piece of the frame `v`
currently belongs to. A fresh, never-yet-welded joint has stiffness 0.
Welding a strut immediately merges the components of `u` and `w` (if they
were separate); the merged component's weld count becomes
`count(u)+count(w)+1`. If `u` and `w` were already in the same component
(a cycle-closing strut), that shared component's weld count simply
increases by 1 — and, crucially, `u` and `w` then share the *same*
stiffness value at the moment that strut fires.

The frame "flinches" less as it hardens: the very first weld into a
component always pulls at full rating (`stiffness=0`), while later welds
into an already-busy component pull weaker. Two struts with equal and
opposite ratings do **not** cancel a joint's displacement if they fire at
different points in the component's weld history, because the divisor
`1+stiffness` has changed between them — cancellation must be planned in
this drifting stiffness metric, not read off the struts' ratings alone.

## Input (stdin)
```
N M
u_1 w_1 eff_1
...
u_M w_M eff_M
```
`0 <= u_i, w_i < N`, `u_i != w_i`, `1 <= |eff_i| <= 100` (a nonzero integer
of bounded magnitude). The frame is simple (no repeated struts) and every
joint has degree >= 2. The frame may consist of several disconnected
pieces.

## Output (stdout)
```
M
e_1 s_1
e_2 s_2
...
e_M s_M
```
`e_1..e_M` must be a permutation of `0..M-1` (the weld order, each strut
welded exactly once); each `s_k` is `+1` or `-1`.

## Feasibility
`M` must match; `e_1..e_M` must be a permutation of all strut indices;
every `s_k` must be exactly `1` or `-1` (as an integer token). Any
violation, extra/missing tokens, or non-finite value scores `0`.

## Objective
Simulate the weld sequence exactly as described above (accumulating `disp`
with 64-bit floats, in the given order). Let `F = max over all joints v of
|disp[v]|` after all `M` welds. **Minimize `F`.**

## Scoring
The checker builds its own reference weld sequence `R` (weld the struts in
input order `0,1,...,M-1`, every side `+1`) and lets `B` be its resulting
`F`. With your submission's `F`:
```
score = min(1000, 100 * B / max(1e-9, F)) / 1000
```
so matching the reference exactly scores `0.1`; a sequence that halves the
reference's residual scores `0.2`, and so on (capped at `1.0`).

## Constraints
`4 <= N <= 260`, `3 <= M <= 400`. Time limit 5s, memory 512MB, `.in` well
under 1MB.

## Example (illustrative FORM only — not a real test case)
A 4-cycle `(0,1,20),(1,2,20),(2,3,20),(3,0,20)` welded in order `0,1,2,3`,
all sides `+1`: weld 0 gives `disp[0]+=20, disp[1]-=20` (both fresh,
stiffness 0). Weld 1: joint 1's component (`{0,1}`, weld count 1) has
stiffness 1, joint 2 is fresh: `disp[1]+=20/2=10` (total `-10`),
`disp[2]-=20` (total `-20`). The "obvious" fix — weld the opposite strut
`(3,0)` right after `(0,1)` hoping the two pulls on joint 0 cancel — still
fails: by the time strut `3` fires, joint 0's component has already
absorbed other welds, so its stiffness has moved and the divisor no longer
matches. Genuine cancellation requires choosing *which* welds share a
matched stiffness state, not just matching struts by rating.
