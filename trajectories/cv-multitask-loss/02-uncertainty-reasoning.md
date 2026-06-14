The gradient-surgery run told me what I suspected it would, and it told me in the spread across the
three backbones. PCGrad landed at 64.31 on ResNet-20, 70.20 on ResNet-56, 74.17 on VGG-16-BN. Look at
the shape: on the larger-headed VGG it is fine, but on the small ResNet-20 it is *the worst point* on
that backbone — 64.31 is a long way below what a plain weighting would reach there. That is exactly the
failure I flagged. Coarse is a deterministic coarsening of fine, so on most steps the two gradients are
*correlated*, `cos φ ≥ 0`, and PCGrad does nothing but pay two extra backward passes; and on the steps
where it does fire, it fixes only the *direction* of the conflict and never asks the question that
actually decides fine accuracy on a small backbone — *how much should the coarse task count at all?* On
ResNet-20 the trunk is tiny, capacity is scarce, and an unweighted coarse loss of the wrong size steals
that capacity; PCGrad cannot see that, because magnitude imbalance is invisible to a direction-only
fix. The deeper ResNet-56 and the larger VGG have capacity to spare, so the mis-weighting hurts less,
which is why the gap to the others shrinks as the backbone grows. So the lesson is sharp: the lever I
left on the table — the *relative weight* between fine and coarse — is the one that matters here, and I
should pick it up next. The trouble is that picking it up naively is its own trap, and I want to walk
into the right scheme rather than the wrong one.

The bare version of the lever is `L = w_fine L_fine + w_coarse L_coarse` with the `w_i` as trainable
parameters, and let SGD set them. Stare at the gradient for one second. `∂L/∂w_i = L_i ≥ 0`, which is
nonnegative, so gradient descent pushes every `w_i` *down*, and there is nothing in this objective that
pushes back. The global optimum over the weights is `w_fine = w_coarse = 0`, where `L = 0` and the
network has learned nothing. The weights collapse to zero. So I cannot just learn the weights of a bare
weighted sum — the objective is happy to turn off both tasks. Whatever scheme learns the weights has to
carry a term that *resists shrinking them*, a cost for declaring a task unimportant. That is the real
constraint: I need a principled coupling between "how much I down-weight a task" and "a penalty for
doing so," not a free knob. And note this is a genuinely different kind of fix from PCGrad — PCGrad
operated on the gradient *directions* and refused to ask about magnitudes; here I am going straight at
the magnitudes, learning them, which is exactly the axis PCGrad's ResNet-20 collapse said was
underserved.

Where would such a coupling come from naturally? The fine and coarse losses are both cross-entropies,
and a cross-entropy is a negative log-likelihood in disguise — it is `−log Softmax(logits)_c` for the
true class `c`. In ordinary single-task training I treat the observation noise of each head as a fixed
constant, fold it into the learning rate, and never write it down. But here is the thing: that constant
is *exactly* a per-task scale. A task that is noisier, or whose loss sits at a different numerical scale
(and a 100-way and a 20-way cross-entropy genuinely do sit at different scales and difficulties), has a
larger effective noise. And that noise does not have to be fixed — I can make it a free parameter and
learn it. This is the heteroscedastic-regression move (Nix & Weigend, 1994: train a mean head and a
variance head, learn the noise by maximizing the Gaussian likelihood). They never needed "uncertainty
labels," because the likelihood couples the noise to the residuals and recovers it implicitly. I do not
need their input-dependence; I want one *constant* noise scale `σ_i` per task — the homoscedastic,
task-dependent version — because what varies across my problem is not the noise from pixel to pixel, it
is the noise and the scale from *task to task*. That is precisely the missing magnitude lever, derived
rather than guessed.

Let me get the form first from the regression case where it is cleanest, then port it to the two
classifiers I actually have. If a head's output is a Gaussian observation `p(y | f(x)) = N(f(x), σ²)`,
its negative log-likelihood is `(1/2σ²)‖y − f(x)‖² + (1/2)log σ² + const`. Writing the bare per-task
loss as `L_i`, the contribution is `(1/2σ_i²) L_i + log σ_i`. Now look at what fell out. The coefficient
on each task's loss is `1/(2σ_i²)` — an *inverse-variance weighting*. As `σ_i` grows, the weight on `L_i`
shrinks: the noisier or larger-scale a task is, the less it is allowed to dominate the shared gradient;
as `σ_i` shrinks, its weight grows. That is exactly the adaptive relative weighting I wanted, and it is
not a heuristic — it is the maximum-likelihood way to combine measurements of differing precision. And
crucially there is a `+log σ_i` sitting there, and it is the anti-collapse term I argued I needed. Watch:
if the optimizer tries the degenerate route of sending `σ_i → ∞` to zero out the `1/(2σ_i²)` coefficient
and make a task free, the `log σ_i` term shoots off to `+∞` and punishes it. The likelihood will not let
me declare a task infinitely noisy and walk away. So the *same* probabilistic model that gives me
inverse-variance weighting automatically supplies the regularizer that the bare learnable weights were
missing — the collapse I hit a moment ago is structurally impossible here. I can confirm the balance is
sane by setting the gradient with respect to `σ_i` to zero: `∂/∂σ_i = −L_i/σ_i³ + 1/σ_i = 0` gives
`σ_i² = L_i`, so at the optimum the learned variance tracks that task's *current* loss. A big-loss task
gets a big `σ` (low weight), a well-fit task a small `σ` (high weight), and the fixed point cannot run to
zero or infinity because the two terms pull against each other.

Both my heads are classifiers, not regressors, so I have to check the cross-entropy case gives the same
shape. The analogue of "scaling the observation noise" for a softmax is temperature: scale the logits by
`1/σ²` before the softmax, `p(y | f(x), σ) = Softmax((1/σ²) f(x))`, so `σ²` is the temperature — large
`σ²` flattens the distribution toward uniform (high uncertainty), small `σ²` sharpens it toward one-hot
(high confidence). Working through the negative log-likelihood, the inverse-variance weight `1/σ_i²`
appears on the unscaled cross-entropy exactly as in the regression case, but the regularizer comes out as
a messy log-ratio of two log-sum-exps that depends on the logits. Under the approximation
`(1/σ) Σ exp((1/σ²) f_{c'}) ≈ (Σ exp f_{c'})^{1/σ²}`, which is *exact at `σ = 1`* and degrades smoothly
away from it, that bracket collapses back to a clean `log σ`. So the classification term is
`(1/σ_i²) L_i + log σ_i`, parallel to regression up to the factor-of-two on the coefficient (the Gaussian
NLL carries a `1/2` the temperature scaling does not). The point for me is that both heads land on the
same shape — inverse-variance weight plus a logarithmic scale penalty — so a single uniform rule works
for the two cross-entropies I actually have.

I had better make sure I can optimize the scale stably, because there are two landmines. `σ` appears as
`1/σ²`, so if it ever wanders to zero I divide by zero and blow up; and `σ` is a variance scale,
constrained positive, which is awkward for plain SGD that will happily step a parameter negative. Both
vanish with one reparameterization: do not learn `σ`, learn the *log-variance* `s := log σ²`. Then
`1/σ² = exp(−s)` (always strictly positive, no divide-by-zero), `log σ = s/2`, and `s` ranges over all
of ℝ so SGD can step it freely. In terms of `s` the per-task term is `exp(−s_i) L_i + s_i` (taking the
canonical convention that absorbs the factor-of-two, which moves the term's value but not its optimum in
`s`). The objective is strictly convex in `s` — `∂/∂s = −exp(−s)L + 1`, zero at `s = log L`, second
derivative `exp(−s)L > 0` — so it has a single minimum and is robust to where I start. That means I can
initialize `s = 0`, i.e. `σ² = 1`, every task weighted equally, the most neutral possible start with no
preference baked in, and let each scalar move toward its fixed point with no extra hand-tuned
hyperparameter. This is the cleanest possible answer to PCGrad's ResNet-20 problem: instead of leaving
the fine/coarse weight to chance, I start them equal and let the likelihood walk them to wherever the
data says.

Now land it in this task's edit surface, and notice how much *simpler* it is than the PCGrad rung — and
why. PCGrad had to fight the interface: walk the autograd graph to recover the shared parameters, call
`torch.autograd.grad` twice, build a surrogate loss to smuggle a projected gradient through the
"return a scalar" contract. Uncertainty weighting needs *none* of that, because it is a pure
loss-weighting rule and the interface hands me exactly the two scalar losses it reads. I register one
log-variance parameter per task, `self.log_vars = nn.Parameter(torch.zeros(2))` — and here the load-bearing
detail of the substrate pays off: the loop builds the optimizer over `model.parameters() +
mtl_loss.parameters()`, so these two scalars are trained jointly with the network by the same SGD, no
extra machinery. The forward is two lines: for each of the two losses compute the precision
`exp(−s_i)` and accumulate `exp(−s_i) · L_i + s_i`, return the sum. That is the whole module. (The full
scaffold module is in the answer.)

So the delta from the PCGrad rung is a change of *axis*: where PCGrad operated on gradient directions
and was blind to magnitude, this rung learns the per-task magnitude directly, with a likelihood-derived
log-variance penalty that keeps it from collapsing to zero. There is one more reason to expect this to
beat both the equal-weighting default and a static grid weight, and it bears on PCGrad's backbone
pattern. A grid weight is *static* for the whole run; the learned `σ` is *dynamic* — early on every loss
is large so every `σ_i` is large and the weighting is roughly even, and as the model masters the easier
20-way coarse task its loss drops, its `σ` drops, and its weight *rises*, with the schedule of relative
weights evolving over training in a way no fixed point can match. That adaptivity is exactly what the
small ResNet-20 trunk needed and could not get from PCGrad.

Now the falsifiable expectations against the numbers I have. The PCGrad run was 64.31 / 70.20 / 74.17.
My sharp claim is on **ResNet-20**: if the ResNet-20 collapse really was a *weighting* problem, then
learning the weight should pull that point *up*, clearly above 64.31 — that is the headline test of this
whole diagnosis, and if it fails there my read of PCGrad was wrong. On **ResNet-56** I expect a smaller
gain over 70.20, because the deeper trunk was already less starved by mis-weighting, so there is less for
the lever to recover. On **VGG-16-BN** I am genuinely unsure of the sign: PCGrad already hit 74.17 there,
VGG has large-capacity heads where the auxiliary signal matters least, and the `σ → 1` approximation in
the classification derivation is only first-order, so the learned weighting might land *near or even
slightly under* PCGrad on that backbone. If the pattern comes out as "big ResNet-20 recovery, modest
ResNet-56 gain, roughly flat VGG," that confirms the magnitude lever is what the task rewards — but it
also tells me what uncertainty weighting still cannot do: it learns *one constant* weight per task and
nothing time-aware beyond the slow fixed-point drift, so if the real story is that the *rate* at which
each task is learning matters within a run, the next rung should weight by the relative *rate of change*
of each loss rather than by a learned static scale.
