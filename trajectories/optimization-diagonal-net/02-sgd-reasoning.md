AdaGrad came back as the weakest possible thing to run, and it failed in exactly the shape I bet it
would. The numbers: `n* = 175` on `d200_k5_s01`, `487` on both `d500_k10` settings, and a full `2000`
on `d10000_k50` — scores `−7.45, −8.93, −8.93, −10.97`. Read the mechanism off those: on the smallest
problem (`d=200, k=5`) the per-coordinate denominator only mildly dilutes the bias, so recovery still
happens, just expensively at `175` samples; but as the ambient dimension grows the gap widens monotonically,
and at `d=10000` AdaGrad needs the *largest grid point in the search* — it never recovered cheaply at
all, it bottomed out at the grid ceiling. That is the prediction confirmed: the denominator damps the
support coordinates whose multiplicative escape is the recovery engine, inflates the off-support steps,
and the monotone accumulation decays the effective rate so the support coordinates stall before the
plateau rule fires. The widening-with-`d` curve is the signature — the dilution of the sparse bias is a
high-dimensional effect, worst exactly where the problem is hardest. So my read on the geometry was
right, and it tells me what to do next with no ambiguity: the adaptive rescaling is the problem, so
*strip it out* and let plain gradient descent ride the diagonal-net's escape dynamics unimpeded. The
scaffold default already is gradient descent at `lr = 0.01`; the question is whether bare GD, at the
right step, beats the adaptive method, and why.

Let me re-derive what plain stochastic gradient descent is doing here, because I want to be sure the
mechanism I am betting on is real and not just "adaptivity bad." The clean abstraction is: I want to
minimize an expected loss I can only see through noise. Exact batch descent on a strongly convex
objective converges linearly, but it pays a full data pass per step and — the part that matters here —
on a non-convex parameterization the loss does not pin the limit. The cheapest noisy estimate of the
gradient is the perturbed one, and the recursion `w ← w − γ g` with a noisy unbiased `g` is a
stochastic-approximation step. The classical analysis (Robbins–Monro) is about *convergence* —
`Σ γ = ∞`, `Σ γ² < ∞` to forget the start and kill the variance — and a fixed step gives only
convergence to a *noise ball* of radius proportional to `γ` rather than the exact optimum. For a
generic convex problem that noise ball is a defect. Here it is not a defect at all: the noise is the
mechanism. The harness adds fresh Rademacher label noise `±delta` to `y` every step, and that
perturbation, ridden by the diagonal-net dynamics, is what *strengthens* the sparse bias. So the
stochastic-approximation framing flips: I do not want a vanishing-step schedule that anneals the noise
away; I want a *constant* step that keeps the noise alive at a useful temperature, because the noise is
doing the regularization the loss cannot.

Now make that concrete on the parameterization, which is where the diagonal-net story diverges sharply
from generic SGD. Write the predictor `w_i = u_i² − v_i²`. The gradient of the squared loss w.r.t. `u_i`
is `(∂L/∂w_i)·2u_i`, and likewise `−(∂L/∂w_i)·2v_i` for `v_i`, where `∂L/∂w_i = (1/n) Σ_j (x_j·w − y_j)
x_{j,i}` is the residual correlation with feature `i`. So the GD step on `u_i` is
`u_i ← u_i − γ·2u_i·r_i` with `r_i` the residual correlation — multiplicative in `u_i`. From the
near-zero, equal start (`u = v = alpha/sqrt(2d)`, `w_hat = 0`), each coordinate's magnitude evolves
roughly like `exp` of the accumulated correlation it has seen: support coordinates, where `r_i` is
persistently large (the residual keeps demanding them), grow first and fast; off-support coordinates,
where `r_i` is only the noise-driven dribble, stay near the floor far longer. This is the
saddle-to-saddle / incremental-learning picture: the trajectory leaves the origin saddle, races up the
support coordinates one tranche at a time, and reaches a sparse interpolator — *before* the off-support
coordinates have grown enough to overfit the noise. The label-noise perturbation sharpens this: it
keeps the off-support coordinates from quietly drifting up in lockstep and biases the implicit
regularizer further toward the ℓ1-like sparse solution (the Pesme–Pillaud-Vivien–Flammarion picture,
that stochasticity is a *provable benefit* on exactly this parameterization). Crucially, plain GD does
*nothing* to interfere with this. There is no per-coordinate denominator to damp the support escape,
no accumulated-history term to decay the rate — every coordinate moves at the same global `γ` scaled by
its own current magnitude and its own residual, which is precisely the multiplicative law that produces
the ordering. That contrast is the diagnosis of AdaGrad made constructive: AdaGrad's `sqrt(Σ g²)`
denominator is *largest* exactly on the support coordinates (they carry the most gradient mass), so it
slows the escape that should be fast and speeds the drift that should be slow — it fights the geometry.
Plain GD does not fight it. That is the entire reason to expect bare GD to beat the adaptive method on
sample complexity, and it is testable.

It is worth being precise about why this is *recovery* and not merely interpolation, because that is
what the harness scores and it is where the diagonal-net's bias pays off. With `n < d` the training
system `Xw = y` is underdetermined: a whole affine subspace of `w` fits the (noisy) labels, and a dense
optimizer will happily settle on a high-norm interpolator that overfits the label noise and tests
badly. The diagonal-net trajectory under plain GD does not explore that subspace uniformly — the
multiplicative escape selects, among all interpolators, the one whose support grew first and whose
off-support coordinates never left the floor, i.e. the *minimum-ℓ1-like* solution, which for a truly
`k`-sparse ground truth is the ground truth itself once `n` is a small multiple of `k log d`. That is
why bare GD can recover at `n` as small as `50` when `d` is `200` and `k` is `5`: the sparse bias
converts the underdetermined problem into a well-posed one. AdaGrad broke this by reshaping which
interpolator the dynamics select; plain GD leaves the selection rule intact. The whole sample-complexity
question is therefore "how few samples until the sparse interpolator the dynamics pick *is* the ground
truth," and the optimizer's only job is not to corrupt that selection.

The one real design choice left is the step size, and it matters more than it would for a convex
problem because of two competing pressures, both visible in the constant-step stochastic-approximation
analysis. The fixed step `γ` sets the radius of the noise ball — the temperature of the perturbation
that drives the sparse bias — and it sets the speed of the multiplicative escape. Too small a `γ` and
two things go wrong: the escape is slow, so on the high-dimensional settings the support coordinates may
not have climbed out of the saddle before the million-step cap or the plateau rule stops training (the
default `lr = 0.01` risks exactly this on `d10000`); and the noise temperature is low, weakening the
sparse-bias benefit. Too large a `γ` and the multiplicative `1 − 2γ r_i` factor can overshoot — a
coordinate can be driven negative or oscillate, and on the squared parameterization a large step can
destabilize the `u² − v²` cancellation that keeps off-support `w_i` near zero, letting noise leak into
the predictor. So I want the largest `γ` that is still stable, to get fast escape and a healthy noise
temperature without blowing up the cancellation. Among the textbook fixed values the harness's baseline
family sweeps (`0.005, 0.01, 0.05, 0.1`), `lr = 0.1` is the top of that stable range, and it is the
right bet here precisely because the binding failure I just watched was *too-slow / damped escape* at
`d=10000`, not instability — I should push the step up, not down. I keep the method otherwise bare:
no momentum, no per-coordinate state, just `w ← w − γ g` applied identically to `u` and `v`, with a
step counter as the only state. The constant step is deliberate — I do *not* anneal it toward zero,
because annealing would cool the noise that is doing the regularizing and would slow the escape on the
large setting; I run at fixed temperature to a plateau and read off the sparse interpolator. The full
scaffold module is in the answer.

So the falsifiable expectation, against AdaGrad's measured `175 / 487 / 487 / 2000`. If the mechanism
story holds — plain GD rides the multiplicative escape that AdaGrad damps — bare SGD at `lr = 0.1`
should recover from *fewer* samples on every setting, and the improvement should be *largest exactly
where AdaGrad was worst*: at `d=10000`, where AdaGrad bottomed out at the grid ceiling of `2000`, plain
GD should recover well below it, because removing the denominator un-stalls the support escape that the
high dimension made fragile. On the small `d200_k5` problem, where AdaGrad already recovered (at `175`),
I expect SGD to need *substantially* fewer — down toward the floor of the grid — since here the issue
was never feasibility, only the bias dilution that bare GD does not suffer. The two `d500_k10` settings
should track each other (they did under AdaGrad) and land between the two extremes. If instead bare GD
does *not* beat AdaGrad — if removing the adaptivity leaves recovery just as expensive — then the
sparse bias is not coming from the multiplicative escape after all and my whole reading is wrong, and
the next rung would have to look for the bias elsewhere. But I expect SGD to be a clear step up the
ladder, and the residual question it will leave open — does *any* adaptivity help, or only
AdaGrad's particular damping — is what sets up the rung after it: a smoothed, forgetting second-moment
preconditioner that might reshape the geometry more gently than AdaGrad's monotone accumulation.
