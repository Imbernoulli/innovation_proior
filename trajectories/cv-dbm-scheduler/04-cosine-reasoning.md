The power-law warp split the difference and the split was uneven, in a way the per-workload numbers make
unmissable. Karras improved exactly where I predicted it would: DIODE fell from uniform's 15.002 to 8.772
(the NFE = 3 workload had the most to gain from dragging the interior stop into the near-data decade), and
edges2handbags dropped from 5.180 to 4.416. But ImageNet inpainting *regressed*, hard — from uniform's
6.070 back up to 12.025, nearly to loglinear's 13.748. That is precisely the failure mode I flagged when I
fixed `ρ = 7`: "if `ρ = 7` over-bets low `t` where ImageNet's mask needs high-`t` resolution, it could
cost a little there." It did not cost a little; it cost almost everything uniform had won on that workload.
The geometric-mean came out at 7.752, a hair under uniform's 7.784 — so on the aggregate the power-law is
the strongest schedule so far, but only because DIODE's large gain barely outweighs ImageNet's large loss.
That is not a robust schedule; that is a schedule whose single fixed bet is right for two bridges and badly
wrong for the third.

Read what the ImageNet regression is telling me, because it is the key to the next move. ImageNet's bridge
is an I2SB inpainting bridge: the source endpoint at `t_max` carries a hard mask — a sharp,
spatially-structured boundary between the known region and the hole. That structure has to be resolved at
*high* `t`, early in the trajectory, where the sampler decides how to propagate context into the hole. The
`ρ = 7` power-law warp does the opposite: it monotonically starves the high-`t` region to feed the
near-data region. For DIODE's smooth depth→image VP bridge, that starvation is fine — the high-`t` half is
nearly straight, nothing structural lives there. For ImageNet, starving high-`t` throws away the steps that
resolve the mask, and the near-data steps cannot reconstruct boundary structure the early steps never
established. So the lesson across all three failures is now complete: loglinear bet everything on low `t`
(one ruinous high-`t` step), uniform bet nothing (near-data region coarse), and karras bet monotonically
toward low `t` (high-`t` structure starved on the workload that has it). Every schedule so far is
*monotone* in where it spends resolution — each makes a single directional bet about which end of the
trajectory is hard. But the evidence says **both ends are hard, in different ways**: the near-data end
because that is where fine image content is committed (DIODE's gain), and the high-`t` end because that is
where conditioning structure is established (ImageNet's loss). A monotone warp cannot serve both ends; by
construction it robs one to pay the other.

So the move is to stop betting on a single end and instead resolve *both endpoints densely while spending
the coarse steps in the middle*, where the VP/I2SB bridge trajectory is closest to a straight interpolation
between its two structured ends. I want a schedule that is symmetric in its treatment of the two endpoints
— small steps approaching `t_max` (to resolve the conditioning) and small steps approaching `t_min` (to
resolve the content), with the large steps in the bend-free middle. That is a fundamentally different shape
from every monotone warp I have tried, and it is exactly the shape a cosine ramp produces. Let me derive
it from the requirements rather than reach for it by name.

I want a smooth, parameter-free warp `g(u)` that maps a uniform ramp `u ∈ [0, 1]` to the placement, with
the property that the resulting `t` changes *slowly* near both `u = 0` (the `t_max` end) and `u = 1` (the
`t_min` end), and *fast* in the middle. "Changes slowly" means the warp has **zero slope at both
endpoints**, so consecutive nodes are bunched there; "changes fast in the middle" means the slope is
maximal at `u = 1/2`, so the nodes are spread there. Write the placement as
`t(u) = t_max + (t_min − t_max)·g(u)` with `g` monotone increasing from `g(0) = 0` (so `t = t_max`) to
`g(1) = 1` (so `t = t_min`). The four conditions are `g(0) = 0`, `g(1) = 1`, `g'(0) = 0`, `g'(1) = 0`.
These pin a specific S-curve. A straight line `g(u) = u` fails the slope conditions (constant slope —
that is just the uniform grid, my `ρ = 1` point). A power `u^p` is flat only at one end. The
half-period-shifted cosine `g(u) = (1 − cos(π u))/2` satisfies all four *exactly*: `g(0) = (1−1)/2 = 0`,
`g(1) = (1−cos π)/2 = (1−(−1))/2 = 1`, and `g'(u) = (π/2)sin(π u)` vanishes at both `u = 0` and `u = 1`
while peaking at `u = 1/2`. It is the unique parameter-free member of the family of monotone S-curves with
zero slope at both ends — no exponent, no period to tune, which keeps it inside the task's
no-dataset-constants rule. So the schedule is `t_i = t_max + (t_min − t_max)·(1 − cos(π·u_i))/2` with
`u_i = i/n`, the full-period cosine warp.

Now I have to be careful about which way this cosine bunches the nodes, because it is the opposite of the
loose description "denser in the middle," and the actual geometry is what matters. The warp `g` has small
slope near the endpoints, so a uniform step in `u` produces a *small* step in `t` there — nodes are dense
near `t_max` and near `t_min`. In the middle the slope is large, so a uniform `u`-step produces a *large*
`t`-step — nodes are sparse, the steps are big, in the middle. Let me trace it for the NFE = 5 workloads
(`n = 4`, ramp `{0, .25, .5, .75, 1}`): the schedule is `{1.0, 0.854, 0.500, 0.147, 1e-4}`. The step
sizes are `{0.146, 0.354, 0.354, 0.147}` — small at both ends, large in the two middle intervals. That is
exactly the both-ends-dense shape I argued for: two of the four steps resolve the near-`t_max`
conditioning region and the near-`t_min` content region, and the two coarse steps span the bend-free
middle. Compare karras's `{1.0, 0.243, 0.041, 0.004, 1e-4}`, which put *three* of its four nodes below
`t = 0.25` and only one step above — all resolution at low `t`, nothing for the high-`t` mask. The cosine
returns a node to the high-`t` region (0.854, a small step off `t_max`) precisely to resolve the structure
karras starved.

There is one consequence of the symmetric warp I have to be honest about, because it tells me where the
aggregate win can and cannot come from. At `n = 2` (DIODE, NFE = 3) the ramp is `{0, 0.5, 1}` and the
cosine schedule is `{1.0, 0.50005, 1e-4}` — *identical to the uniform grid*, because the only interior
node sits at `u = 0.5` where `g(0.5) = 0.5` exactly. So on DIODE the cosine warp is schedule-neutral
against uniform: any difference in DIODE FID between the two is run-level sampling variance, not a schedule
effect, and DIODE is therefore *not* a workload where cosine can claim a curve-driven gain over uniform —
nor a guaranteed loss. The symmetry that lets cosine serve both ends only acts once there are at least two
interior nodes, i.e. at NFE = 5. That sharpens the bet: cosine's case has to be made entirely on the two
NFE = 5 workloads, edges2handbags and ImageNet, where the both-ends-dense shape actually places distinct
nodes near `t_max` and `t_min`. If returning a dense high-`t` node recovers the mask resolution karras
destroyed, ImageNet should fall from 12.025 back toward — or below — uniform's 6.070, and edges2handbags,
whose VP bridge has mild structure at *both* ends, should hold near karras's strong 4.416. The aggregate
geometric mean rewards a schedule that is merely *good at both ends* over one that is *excellent at low `t`
on two workloads and broken at high `t` on the third*: it punishes the worst workload hardest — that is why
karras's 12.025 on ImageNet dragged its aggregate to a near-tie with uniform despite winning two workloads
— so fixing the ImageNet disaster while leaving DIODE schedule-unchanged should win the aggregate outright.
The full scaffold module is in the answer.

The implementation is the cleanest fill on the ladder after uniform. The cosine ramp lands `g(0) = 0` and
`g(1) = 1` exactly, so `t(0) = t_max` and `t(1) = t_min` are exact and — like uniform, unlike loglinear and
karras — *no terminal pin is needed*; the contract's terminal-equals-`t_min` clause holds for free.
Length is `n + 1` by construction, monotonicity follows because `g` is strictly increasing on `[0, 1]` so
`t` is strictly decreasing, and the tensor goes to `device`. The only import is `math` for `π`.

So the delta from the karras rung is a shape change, not a knob change: where every prior schedule was
monotone in where it spent resolution — and therefore forced to rob one end of the trajectory to pay the
other — the cosine warp is symmetric, bunching nodes at *both* the conditioning end (`t_max`) and the
content end (`t_min`) and spending its coarse steps in the bend-free middle. Here is what I expect against
karras's numbers. ImageNet inpainting is the workload that should move most: karras's 12.025 was a high-`t`
starvation failure, and returning a dense high-`t` node should recover the mask resolution, so I expect
ImageNet to drop sharply — back below uniform's 6.070 is the target. edges2handbags should hold near
karras's strong 4.416, since its VP bridge has structure at both ends that the symmetric placement serves
(though giving a node back to high `t` costs some of karras's low-`t` packing, so a small slip there would
not surprise me). DIODE I have no curve-driven claim on — cosine ≡ uniform at `n = 2`, so whatever DIODE
prints is sampling variance around uniform's neighborhood, neither karras's near-data gain nor uniform's
plateau guaranteed. The falsifiable aggregate claim is therefore narrow and clean: cosine's geometric mean
clears karras's 7.752 — and clears it comfortably, not by a hair — driven almost entirely by fixing the
ImageNet disaster (a factor-of-two FID move on the workload the geometric mean weights most), with DIODE
schedule-unchanged and edges roughly flat. If instead cosine lands near karras on the aggregate, that would
mean the both-ends-dense bet over-corrected — that the low-`t` packing it gave back mattered more than
ImageNet's high-`t` structure — and the next question would be how to keep cosine's high-`t` node *and*
karras's near-data packing at once, which no single parameter-free warp on this contract can do. But the
evidence so far — both ends carrying structure, the geometric mean punishing the worst workload — says the
symmetric
schedule is the one that stops the ladder from robbing Peter to pay Paul.
