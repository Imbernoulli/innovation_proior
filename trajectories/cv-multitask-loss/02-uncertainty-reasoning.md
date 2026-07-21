The gradient-surgery run told me what I suspected it would, and it told me in the spread across the
three backbones. PCGrad landed at 64.31 on ResNet-20, 70.20 on ResNet-56, 74.17 on VGG-16-BN. Before I
read the shape I want the single summary number so I have something to beat: the task score is the
geometric mean, `(64.31 · 70.20 · 74.17)^{1/3}`, and the product is about `3.348×10⁵`, whose cube root
is `69.44`. So 69.44 is the bar. But the mean hides the thing I actually care about, which is the
*pattern* of the deficit, so let me difference the three points. VGG minus ResNet-20 is
`74.17 − 64.31 = 9.86`; ResNet-56 minus ResNet-20 is `70.20 − 64.31 = 5.89`; VGG minus ResNet-56 is
`74.17 − 70.20 = 3.97`. Two things jump out. The gap is enormous — nearly ten points across backbones
that all run the identical loss rule — and it is *monotone in capacity*: smallest on the biggest
backbone, and it grows by `5.89` going from ResNet-56 down to the tiny ResNet-20 and by a further
`3.97` going from VGG down to ResNet-56. The deficit tracks how little trunk there is to go around,
and ResNet-20 (which is `[3,3,3]` blocks against ResNet-56's `[9,9,9]`, a third of the depth) is where
it bottoms out. That is not the signature of a direction problem, which would hit all three backbones
roughly alike since conflict geometry does not much care about trunk size. It is the signature of a
*capacity* problem: on the small trunk an unweighted coarse loss of the wrong size steals capacity the
fine head needs, and PCGrad cannot see that, because magnitude imbalance is invisible to a
direction-only fix — it never asks *how much* the coarse task should count, only whether it fights.
The deeper ResNet-56 and the larger VGG have capacity to spare, so the mis-weighting hurts less, which
is exactly why the gap shrinks as the backbone grows. The 64.31 on ResNet-20 is the loudest data point
I have, and it is telling me the lever I left on the table — the *relative weight* between fine and
coarse — is the one that matters here. So I should pick it up next. The trouble is that picking it up
naively is its own trap, and I want to walk into the right scheme rather than the wrong one.

What could "learning the weight" even mean here? The most literal is a *static* weight
`L = w L_fine + (1−w) L_coarse` found by grid search — but I have no honest way to search it: the
substrate fixes a single 200-epoch run per configuration, seed 42, with no held-out split, so a
"searched" weight would really be one I peeked at the test metric to pick, exactly the leakage the
benchmark forbids. And a static `w` is blind to the training-time evolution anyway — it commits to one
balance for all 200 epochs, while the capacity-dependent deficit suggests the right balance is something
the run should discover. So the weight has to be *learned* jointly: make `w_fine`, `w_coarse` trainable
and let SGD set them. But `∂L/∂w_i = L_i ≥ 0` is nonnegative, so descent pushes every `w_i` *down* with
nothing pushing back; the global optimum is `w_fine = w_coarse = 0`, where `L = 0` and the network has
learned nothing. The weights collapse. So a bare learned weighted sum cannot work — whatever learns the
weights must carry a term that *resists shrinking them*, a cost for declaring a task unimportant.

The third candidate is genuinely tempting, so I walk it a few steps before putting it down: balance the
tasks by their *gradient magnitudes*, GradNorm-style — drive the weights so the two per-task gradient
norms on the shared trunk track a common target rate. It goes straight at the magnitude imbalance the
ResNet-20 number just fingered. But two things kill it here. First, per-task gradient norms on the
shared trunk put me back inside the machinery I paid so dearly for with PCGrad: with only the two scalar
losses, getting `∂L_i/∂θ_shared` means walking the autograd graph and calling `torch.autograd.grad`
twice per step, reintroducing the two-to-three-times step cost I would love to shed now that the
geometry was not the disease. Second, GradNorm carries a restoring-force hyperparameter `α` and a
running gradient-norm target — a knob I would have to set with no validation budget, the same leakage as
the grid weight in a different hat. So GradNorm is a heavy, knob-laden route to a magnitude lever, and I
want a light one. That is the real constraint the collapse taught me: I need a principled coupling
between how much I down-weight a task and a penalty for doing so, built in rather than bolted on, ideally
with no free knob at all.

Where would such a coupling come from naturally? The fine and coarse losses are both cross-entropies,
and a cross-entropy is a negative log-likelihood in disguise — it is `−log Softmax(logits)_c` for the
true class `c`. In ordinary single-task training I treat the observation noise of each head as a fixed
constant, fold it into the learning rate, and never write it down. But here is the thing: that constant
is *exactly* a per-task scale. A task that is noisier, or whose loss sits at a different numerical scale
(and a 100-way and a 20-way cross-entropy genuinely do sit at different scales and difficulties — I saw
their initial values sit near `log 100 ≈ 4.6` and `log 20 ≈ 3.0`), has a larger effective noise. And
that noise does not have to be fixed — I can make it a free parameter and learn it. This is the
heteroscedastic-regression move (Nix & Weigend, 1994: train a mean head and a variance head, learn the
noise by maximizing the Gaussian likelihood). They never needed "uncertainty labels," because the
likelihood couples the noise to the residuals and recovers it implicitly. I do not need their
input-dependence; I want one *constant* noise scale `σ_i` per task — the homoscedastic,
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
missing — the collapse I hit a moment ago is structurally impossible here, and I did not have to pick a
penalty strength for it, which is the whole reason it beats bolting an arbitrary `L2` penalty onto the
bare weights: the coupling's strength is fixed by the likelihood, not by a knob I would have no budget
to tune. I can confirm the balance is sane by setting the gradient with respect to `σ_i` to zero:
`∂/∂σ_i = −L_i/σ_i³ + 1/σ_i = 0` gives `σ_i² = L_i`, so at the optimum the learned variance tracks that
task's *current* loss. The fixed point cannot run to zero (the `1/σ³` term diverges and shoves it back
up) or to infinity (the `1/σ` term keeps pulling it down), because the two terms pull against each
other; and the meaning of `σ_i² = L_i` is that the smaller-loss task ends with the larger inverse
weight `1/L_i`. Which way that tips the fine/coarse balance in the fully coupled system — where
weighting a task harder also drives its loss down and moves its own fixed point — I genuinely cannot
read off statically, and I will not pretend to; the ResNet-20 number is what tells me whether letting
the likelihood set that balance beats freezing both weights at 1.

Both my heads are classifiers, not regressors, so I have to check the cross-entropy case gives the same
shape. The analogue of "scaling the observation noise" for a softmax is temperature: scale the logits by
`1/σ²` before the softmax, `p(y | f(x), σ) = Softmax((1/σ²) f(x))`, so `σ²` is the temperature — large
`σ²` flattens the distribution toward uniform (high uncertainty), small `σ²` sharpens it toward one-hot
(high confidence). Working through the negative log-likelihood, the inverse-variance weight `1/σ_i²`
appears on the unscaled cross-entropy exactly as in the regression case, but the regularizer comes out as
a messy log-ratio of two log-sum-exps that depends on the logits. Under the approximation
`(1/σ) Σ_c' exp((1/σ²) f_{c'}) ≈ (Σ_c' exp f_{c'})^{1/σ²}` the bracket collapses back to a clean
`log σ`, and that approximation is *exact* at `σ = 1`: the left side is `Σ_c' exp f_{c'}` and the right
side is `(Σ_c' exp f_{c'})^1`, literally equal, degrading smoothly as `σ` moves away.
Since I am going to *initialize* every `σ_i` at 1, the approximation is exact at the start and only
first-order off as the scales drift, which is the regime it is least wrong in. So the classification term
is `(1/σ_i²) L_i + log σ_i`, parallel to regression up to the factor-of-two on the coefficient (the
Gaussian NLL carries a `1/2` the temperature scaling does not). The point for me is that both heads land
on the same shape — inverse-variance weight plus a logarithmic scale penalty — so a single uniform rule
works for the two cross-entropies I actually have.

I had better make sure I can optimize the scale stably, because there are two landmines. `σ` appears as
`1/σ²`, so if it wanders to zero I divide by zero and blow up; and `σ` is a positive variance scale,
awkward for plain SGD that will happily step it negative. Both vanish with one reparameterization: learn
the *log-variance* `s := log σ²` instead of `σ`. Then `1/σ² = exp(−s)` is always strictly positive,
`log σ = s/2`, and `s` ranges over all of ℝ so SGD steps it freely. The per-task term becomes
`exp(−s_i) L_i + s_i` (the canonical convention absorbs the Gaussian factor-of-two, which scales the
term's height but not its minimum in `s`). Its derivative `−exp(−s)L + 1` is zero at `s = log L`, with
second derivative `exp(−s)L > 0`, so the term is strictly convex in `s` with a single minimum — robust
to where I start. And the neutral start is the payoff: `s = 0` means `σ² = 1`, every task weighted
equally, no preference baked in, and the convexity means I do not have to tune where it begins. This is
the cleanest answer to PCGrad's ResNet-20 problem: start the fine/coarse weights equal and let the
likelihood walk them wherever the data says, with no hand-tuned hyperparameter anywhere in the rule.

At the first step `s = 0` everywhere, so the weights are exactly `(1,1)` and the rule *is* the default
sum; it departs only as the losses reveal their relative scales — neutral at `t = 0`, adaptive thereafter.

Now land it in this task's edit surface, and notice how much *simpler* it is than the PCGrad rung — and
why. PCGrad had to fight the interface: walk the autograd graph to recover the shared parameters, call
`torch.autograd.grad` twice, build a surrogate loss to smuggle a projected gradient through the
"return a scalar" contract, and pay two extra backward passes a step for it. Uncertainty weighting needs
*none* of that, because it is a pure loss-weighting rule and the interface hands me exactly the two
scalar losses it reads. I register one log-variance parameter per task,
`self.log_vars = nn.Parameter(torch.zeros(2))` — and here the load-bearing detail of the substrate pays
off: the loop builds the optimizer over `model.parameters() + mtl_loss.parameters()`, so these two
scalars are trained jointly with the network by the same SGD, no extra machinery, no separate optimizer,
no second learning rate to guess. The forward is two lines: for each of the two losses compute the
precision `exp(−s_i)` and accumulate `exp(−s_i) · L_i + s_i`, return the sum. That is the whole module,
and it is back to one forward and one backward a step — I have shed PCGrad's tax entirely. (The full
scaffold module is in the answer.)

So the delta from the PCGrad attempt is a change of *axis*: PCGrad operated on gradient directions and
was blind to magnitude; this rule learns the per-task magnitude directly, with a likelihood-derived
log-variance penalty that keeps it from collapsing to zero. And unlike a static grid weight, the learned
`σ` is *dynamic*: at the fixed point `σ_i² = L_i`, so early on both losses are large, both `σ` are
large, and the weighting is roughly even; as the easier 20-way coarse task is mastered and its loss
drops, its `σ` drops and its weight *rises*, a schedule of relative weights no fixed number can match.
That adaptivity is exactly what the small ResNet-20 trunk needed and could not get from PCGrad. The one
caveat I keep honest is that the drift is *slow*: `s_fine` and `s_coarse` are two scalars stepped by the
same SGD learning rate that is simultaneously moving ten million network weights, so they crawl toward
their fixed points rather than snap to them — the equilibrium is the target, not the instantaneous state
at any given epoch.

Now the falsifiable expectations against PCGrad's 64.31 / 70.20 / 74.17 (gmean 69.44). My sharp claim is
on **ResNet-20**: if its collapse really was a *weighting* problem, learning the weight should pull that
point clearly above 64.31 — the headline test of the whole diagnosis, and if it fails there my read of
PCGrad was wrong. On **ResNet-56** I expect a smaller gain, since the deeper trunk was already less
starved — most of the deficit spread sat between it and ResNet-20. On **VGG-16-BN** I am genuinely
unsure of the sign: PCGrad already reached 74.17, VGG has the large heads where the auxiliary signal
matters least, and the `σ → 1` classification approximation is only first-order once the scales drift,
so the learned weighting could land near or slightly below PCGrad there. Even if the magnitude lever
wins overall, it exposes its own ceiling: this rule learns *one constant* scale per task, time-aware
only through that slow fixed-point drift. If the balance the trunk needs at epoch 20 differs from the
one it needs at epoch 150 — a within-run schedule a slowly-drifting learned *level* cannot track — then
a level-based rule has climbed as far as it can, and the next lever has to read something the loss level,
however cleverly learned, cannot see.
