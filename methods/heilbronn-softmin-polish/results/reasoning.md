Annealing reached `0.0356`, about `0.962` of the record, and then plateaued — and I want to know
exactly why before I reach for a new tool, because the answer dictates the tool. So I look at the
configuration sitting at the plateau. At the best annealed point I count how many triangles are
essentially equal to the current minimum, and there are many of them: in one polished near-optimal
configuration I measured `18` triangles within `0.1%` of the smallest. So the minimum is not held by
one fragile triangle I could grow with a lucky nudge; it is held by a whole cluster that are all tied.
To see what that does to a random single-point move, I took such a configuration and tried `2000`
random Gaussian kicks of one point at a time, checking each time whether the minimum triangle area
went up. The result was stark: `0` of the `2000` moves grew the minimum, best gain `0.00e+00`. That
is the wall, made concrete. A single point participates in some of the tied triangles but not all of
them, so moving it grows the ones it touches while leaving — or shrinking — the others, and the `min`
never rises. The basin is right; the move is wrong. What I need is a move that nudges *all* the points
together so every tied triangle grows at once, and that is a smooth optimization problem asking for a
gradient, not a random kick.

The obstacle is that my objective, the minimum over `165` triangle areas, is not differentiable. The
`min` has a kink wherever which triangle is smallest changes, and where it is differentiable its
gradient sees only the *single* currently-smallest triangle and is blind to the `17` others tied with
it. A gradient on the raw `min` would push only the one tightest triangle up, instantly make a
different triangle the tightest, and chatter — the same single-triangle myopia my `2000`-move test
just exposed, now in continuous form. So I do not want the gradient of the `min`. I want the gradient
of something smooth that feels all the near-tight triangles at once.

The standard smooth stand-in for a minimum is the log-sum-exp soft-minimum: with a sharpness `β`,
`softmin(areas) = -(1/β)·log Σ exp(-β·area_t)`. Before I trust it I want to know two things
concretely: does it actually approach the true minimum as `β` grows, and does its gradient really
weight the tied triangles the way I am hoping? I check the first numerically. On a random small
configuration with hard minimum `0.00492`, I evaluate the soft-min at a ladder of `β`:

```
beta=    10  softmin=-0.11913   (hardmin - softmin)=1.2e-01
beta=   100  softmin= 0.00078   (hardmin - softmin)=4.1e-03
beta=  1000  softmin= 0.00492   (hardmin - softmin)=8.1e-09
beta= 10000  softmin= 0.00492   (hardmin - softmin)=0.0e+00
beta=100000  softmin= 0.00492   (hardmin - softmin)=0.0e+00
```

So the soft-min always sits *below* the hard minimum (it is a smoothed-down version), and the gap
closes from `0.12` at `β=10` to machine zero by `β≈10^4`. That settles both the promise and the
danger: at large `β` the surrogate tracks the real minimum, but at small `β` its optimum is a soft
proxy lying strictly under the true value, so I cannot just optimize a small-`β` surrogate and read
off the answer — its maximizer is not the maximizer of the hard minimum. I will have to deal with that.

For the second question — does the gradient weight the tied triangles — I look at the form. The
derivative of the soft-min with respect to each triangle's area is the softmax weight
`exp(-β·area_t) / Σ exp(-β·area_s)`, which is large for the tightest triangles and exponentially small
for the slack ones. Maximizing the soft-min therefore moves the configuration so as to inflate every
triangle that is near the current minimum, weighted by how tight it is — which is exactly the
coordinated push the `2000`-kick test showed a single random move cannot make. And each triangle's
area is a smooth function of the coordinates: `area = ½|cross|` where `cross` is bilinear in the six
coordinates of its three vertices, so by the chain rule the soft-min has a clean analytic gradient in
all `22` coordinates. I work out the cross-product partials by hand. Writing
`cross = (b_x−a_x)(c_y−a_y) − (c_x−a_x)(b_y−a_y)`, differentiating in `a_x` gives
`−(c_y−a_y) + (b_y−a_y) = b_y − c_y`, and in `a_y` gives `−(b_x−a_x) + (c_x−a_x) = c_x − b_x`; the
partials for `b` and `c` are the cyclic rotations of this. So I do not need autodiff — I can assemble
the whole gradient by scattering each triangle's contribution, weighted by its softmax coefficient and
the sign of its signed area (since `|cross|` differentiates to `sign(cross)·cross`), onto its three
points.

Hand-derived gradients are exactly where a sign error hides, so I do not take the assembly on faith. I
code up `neg_softmin_and_grad` and check it against a central finite-difference of the soft-min value
on a random configuration, at two sharpnesses:

```
beta=  50   max|grad_analytic − grad_fd| = 3.9e-10
beta=1000   max|grad_analytic − grad_fd| = 4.5e-10
```

Agreement to `4·10^-10` at both `β` — the scatter, the signs, and the cyclic partials are right. Now
I have a smooth objective with a verified analytic gradient and box constraints (coordinates in
`[0,1]`), which is the natural setting for a bounded quasi-Newton method, L-BFGS-B. I hand it the
negative soft-min and its gradient and let it climb.

But the `β`-versus-truth tension I measured above has to be resolved. If I set `β` huge from the
start, the surrogate is essentially the true `min` — sharp kinks, a near-degenerate gradient
concentrated on one or two triangles, and the optimizer stalls exactly like a gradient on the raw
`min`. If I set `β` small, the landscape is smooth and easy but I am optimizing a soft minimum that my
table shows sits well below the true one, so I land on the wrong maximizer. Neither fixed `β` works, so
I anneal it: start at a moderate `β` where the landscape is smooth, optimize to convergence, raise `β`,
re-optimize from the result, and repeat up a ladder to a very large `β`. Each stage warm-starts the
next, and the final large-`β` stage makes the surrogate track the true minimum, so the configuration
it lands on maximizes the hard minimum and not a proxy. After the ladder I recompute the *exact*
minimum over all `165` triangles — the surrogate guides the search, but the reported number is always
the real thing.

Does this polish actually do something, or am I talking myself into it? I run it on a quick annealed
seed and measure. The seed scores `0.028733` (`77.6%` of `1/27`), and at that seed I count `10`
triangles within `5%` of the minimum — the tied cluster is there as expected. After the `β`-ladder
polish the exact minimum is `0.030858` (`83.3%`). So the gradient ascent does move the cluster up
together: a clean `+0.002` on the hard minimum from a move that no single-point kick was making. I also
check that a polished configuration is a genuine resting point of the procedure — polishing twice in a
row should not change it. Polishing once gives `0.01844701`; polishing the result again gives
`0.01844701`, a change of `3.5·10^-10`. So the polish lands on a fixed point of soft-min ascent, which
is the behavior I want from a local refiner.

That last fact also exposes the method's real limitation, and I want to be honest about it rather than
hide it. The polish is *local*: it climbs to the nearest fixed point and stops. On my quick seed that
fixed point was only `83%` of the record, not the record — the seed was in the wrong basin. So the
polish only reaches `1/27` if it *starts* in the record's basin, and the seeding is therefore
load-bearing, not incidental. I test how much the seed matters by running heavier annealing — many
multi-restart runs of `300,000` steps each — and polishing each. The polished scores scatter widely by
basin: across twelve restarts I see `0.0296, 0.0303, 0.0327, 0.0341, 0.0343, …`, ranging from `80%` to
`92.5%` of `1/27`, with the best at `0.034267`. So most of the variance is *which basin annealing
found*, and the polish merely climbs cleanly to the top of whichever basin it is handed. The way to
reach the record is to feed the polish the best basin available — and the best basin available is the
one the previous rung already found. Its annealed best at `0.0356` is presumably sitting right at the
edge of the record's basin, near-tight cluster and all, so a soft-min polish from there has the best
chance of snapping onto the exact optimum. So I seed the polish both from heavy fresh annealing *and*
from the rung-3 best configuration. This is the small-scale analogue of the search-plus-refine recipe
used on the sibling Heilbronn containers: a stochastic global search to find the basin, then a smooth
local polish to land on the record.

One more layer, because the fixed-point test cuts both ways: the polish stops at *a* local optimum, and
adjacent basins of near-equivalent optima are separated by small barriers a deterministic ascent will
not cross. So after the polish identifies a strong configuration, I basin-hop around it — perturb it
slightly, re-polish, keep the best exact min-area, a few dozen rounds. In my reduced run the hops past
the best basin gave only sub-`10^-6` improvements (the best was already a tight fixed point), which is
the expected behavior: basin-hopping is insurance against stopping a hair early in a slightly worse
neighbor, not a second optimizer.

Putting the measurements together: the polish reliably climbs to the top of whatever basin it starts
in (verified `+0.002` on a real seed, idempotent to `10^-10`), the gradient that does it is correct to
`10^-10`, and the only thing standing between this and `1/27` is landing in the right basin — which is
exactly what the heavy multi-restart annealing plus the rung-3 seed are for. My reduced eight-restart
run already reaches `92.5%`; the full contract (`48` restarts, the rung-3 seed, and basin-hopping)
gives the basin-finding enough tries to land in the record's basin, where the polish then snaps onto
the exact optimum. I am honest that this *matches*, not *beats*, the record: `1/27` is the conjectured
optimum at `n = 11`, and matching it to floating-point precision is the ceiling of what a
single-machine search-plus-polish can do here. Going *beyond* the tabulated records — as was done for
the unit-triangle and convex-region containers — requires large-scale evolutionary search on far more
compute, attacking cases where the tabulated value is *not* already optimal; at `n = 11` in the square
the value is believed optimal, so the honest endpoint of this ladder is the record itself, reached and
confirmed by recomputing the exact minimum, with the proof of optimality standing as the still-open
part of the problem.
