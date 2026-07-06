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

Let me lay out what "learning the weight" could even mean here, because there are three or four
candidates and I do not want to grab the first one. The most literal is a *static* weight,
`L = w L_fine + (1−w) L_coarse`, with `w` found by grid search. But grid search is not available to me
in any honest form: the substrate fixes everything to a single 200-epoch run per configuration, seed
42, with no held-out split to tune `w` on, so a "searched" weight would really be a weight I peeked at
the test metric to pick, which is exactly the leakage the benchmark exists to forbid. And even if I
could search it, a static `w` is by construction blind to the training-time evolution — it commits to
one balance for all 200 epochs, and the fact that the deficit is capacity-dependent suggests the right
balance is not a universal constant but something the run should discover. So the weight has to be
*learned* jointly, and that is the second candidate: make `w_fine`, `w_coarse` trainable parameters and
let SGD set them. Stare at the gradient for one second. `∂L/∂w_i = L_i ≥ 0`, which is nonnegative, so
gradient descent pushes every `w_i` *down*, and there is nothing in this objective that pushes back.
The global optimum over the weights is `w_fine = w_coarse = 0`, where `L = 0` and the network has
learned nothing. The weights collapse to zero. So I cannot just learn the weights of a bare weighted
sum — the objective is happy to turn off both tasks. Whatever scheme learns the weights has to carry a
term that *resists shrinking them*, a cost for declaring a task unimportant.

The third candidate is the one that is genuinely tempting, so I have to walk it a few steps before I
can put it down: balance the tasks by their *gradient magnitudes*, GradNorm-style — measure
`‖∂(w_i L_i)/∂θ_shared‖` for each task and drive the weights so the two gradient norms track a common
target rate. It is attractive because it goes straight at the magnitude imbalance the ResNet-20 number
just fingered. But two things kill it for me here. First, computing per-task gradient norms on the
shared trunk means I am back inside the exact machinery I paid so dearly for in the PCGrad rung: I have
only the two scalar losses, so to get `∂L_i/∂θ_shared` I would have to walk the autograd graph and call
`torch.autograd.grad` twice per step, reintroducing the two-to-three-times step cost I would love to
shed now that I know the geometry was not the disease. Second, GradNorm carries a restoring-force
hyperparameter `α` and a running target for the gradient-norm ratio — a knob I would have to set with
no validation budget to set it against, which is the same leakage problem as the grid weight wearing a
different hat. So GradNorm is a heavy, hyperparameter-laden way to reach a magnitude lever, and I want
a *light* one with no tuning. That is the real constraint the collapse taught me: I need a principled
coupling between "how much I down-weight a task" and "a penalty for doing so," built in rather than
bolted on, and ideally with no free knob at all. And note this is a genuinely different kind of fix
from PCGrad — PCGrad operated on the gradient *directions* and refused to ask about magnitudes; here I
am going straight at the magnitudes, learning them, which is exactly the axis PCGrad's ResNet-20
collapse said was underserved.

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
`log σ`, and I should check *where* that approximation is exact rather than wave at it: at `σ = 1` the
left side is `Σ_c' exp f_{c'}` and the right side is `(Σ_c' exp f_{c'})^1 = Σ_c' exp f_{c'}`, the two
are literally equal, so the approximation is exact at `σ = 1` and degrades smoothly as `σ` moves away.
Since I am going to *initialize* every `σ_i` at 1, the approximation is exact at the start and only
first-order off as the scales drift, which is the regime it is least wrong in. So the classification term
is `(1/σ_i²) L_i + log σ_i`, parallel to regression up to the factor-of-two on the coefficient (the
Gaussian NLL carries a `1/2` the temperature scaling does not). The point for me is that both heads land
on the same shape — inverse-variance weight plus a logarithmic scale penalty — so a single uniform rule
works for the two cross-entropies I actually have.

I had better make sure I can optimize the scale stably, because there are two landmines. `σ` appears as
`1/σ²`, so if it ever wanders to zero I divide by zero and blow up; and `σ` is a variance scale,
constrained positive, which is awkward for plain SGD that will happily step a parameter negative. Both
vanish with one reparameterization: do not learn `σ`, learn the *log-variance* `s := log σ²`. Then
`1/σ² = exp(−s)` (always strictly positive, no divide-by-zero), `log σ = s/2`, and `s` ranges over all
of ℝ so SGD can step it freely. In terms of `s` the per-task term is `exp(−s_i) L_i + s_i` (taking the
canonical convention that absorbs the factor-of-two, which moves the term's value but not its optimum in
`s`). Let me verify this is well-behaved rather than assert it: `∂/∂s = −exp(−s)L + 1`, which is zero at
`s = log L`, and the second derivative is `exp(−s)L > 0`, so the term is strictly convex in `s` with a
single minimum — robust to where I start. I can trace the pull concretely. Take a task sitting at loss
`L = 2` and start it at `s = 0`: the derivative is `−exp(0)·2 + 1 = −1`, negative, so descent *raises*
`s`, down-weighting the task; step to `s = 1` and the derivative is `−exp(−1)·2 + 1 = −0.736 + 1 =
+0.264`, now positive, so descent *lowers* `s` — the minimum is caught between `s = 0` and `s = 1`,
right where `log 2 ≈ 0.693` says it should be. So the scalar really does walk to its fixed point from a
neutral start. I should also check that absorbing the Gaussian `1/2` into the canonical
`exp(−s) L + s` form is harmless, since I dropped it casually. The strict form is
`½ exp(−s) L + ½ s`, whose derivative is `−½ exp(−s) L + ½`, zero when `exp(−s) L = 1`, i.e. at
`s = log L` — the *same* fixed point as the canonical form's `−exp(−s) L + 1 = 0`. The `1/2` scales the
term's height and it scales the gradient the scalar feels, but it does not move where the minimum sits,
so the two conventions learn the same equilibrium scale and I lose nothing by taking the tidier one.
And the neutral start is the payoff: `s = 0` means `σ² = 1`, every task weighted equally,
the most neutral possible start with no preference baked in, and the strict convexity means I do not
have to tune where it begins. This is the cleanest possible answer to PCGrad's ResNet-20 problem:
instead of leaving the fine/coarse weight to chance, I start them equal and let the likelihood walk them
to wherever the data says, with no extra hand-tuned hyperparameter anywhere in the rule.

One more limit check to make sure the rule degenerates sensibly. If the two tasks ever sit at the same
loss, `L_fine = L_coarse`, their fixed points coincide, `s_fine = s_coarse`, the two `exp(−s)` weights
are equal, and the rule reduces to the equal-weighted sum — so at the worst case it does no harm
relative to the default. And at the very first step, `s = 0` everywhere, the weights are exactly `(1,1)`
and I *am* the default sum, so the method starts from the scaffold and only departs from it as the
losses reveal their relative scales. That is the behaviour I want: neutral at `t = 0`, adaptive
thereafter.

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

So the delta from the PCGrad rung is a change of *axis*: where PCGrad operated on gradient directions
and was blind to magnitude, this rung learns the per-task magnitude directly, with a likelihood-derived
log-variance penalty that keeps it from collapsing to zero. There is one more reason to expect this to
beat both the equal-weighting default and a static grid weight, and it bears on PCGrad's backbone
pattern. A grid weight is *static* for the whole run; the learned `σ` is *dynamic* — early on every loss
is large so every `σ_i` is large and the weighting is roughly even, and as the model masters the easier
20-way coarse task its loss drops, its `σ` drops, and its weight *rises*, with the schedule of relative
weights evolving over training in a way no fixed point can match. Let me trace that with concrete
numbers so I know what "dynamic" actually buys. At init both losses are near their uniform-guess values,
`L_fine ≈ 4.6` and `L_coarse ≈ 3.0`, both starting from `s = 0`, so the weights are `(1, 1)` and the two
fixed points the scalars are being pulled toward are `s_fine → log 4.6 ≈ 1.53` and
`s_coarse → log 3.0 ≈ 1.10` — both positive, so both scalars climb and both weights fall from 1, the
fine one falling faster since its target is higher. By mid-training the coarse task, being a 20-way
problem, has been largely mastered, say `L_coarse ≈ 1.0` while `L_fine ≈ 2.0` still lags; the fixed
points have moved to `s_coarse → log 1.0 = 0` and `s_fine → log 2.0 ≈ 0.69`, so the coarse scalar,
which had climbed, is now pulled back *down* toward 0, its weight `exp(−s_coarse)` climbing back toward
1 and above, while the fine scalar settles near `0.69` with weight `≈ 0.5`. So the relative weight the
trunk feels genuinely *turns over* across the run — the coarse task's pull on the shared features waxes
as it is mastered — and a static grid weight, committed to one number for all 200 epochs, cannot
express that turnover. That adaptivity is exactly what the small ResNet-20 trunk needed and could not
get from PCGrad. The one caveat I have to keep honest is that this drift is *slow*: `s_fine` and
`s_coarse` are two scalars stepped by the same SGD learning rate that is simultaneously moving ten
million network weights, so the log-vars crawl toward these fixed points rather than snap to them, and
the equilibrium I just traced is the target, not the instantaneous state at any given epoch.

Now the falsifiable expectations against the numbers I have. The PCGrad run was 64.31 / 70.20 / 74.17,
gmean 69.44. My sharp claim is on **ResNet-20**: if the ResNet-20 collapse really was a *weighting*
problem, then learning the weight should pull that point *up*, clearly above 64.31 — that is the
headline test of this whole diagnosis, and if it fails there my read of PCGrad was wrong. On
**ResNet-56** I expect a smaller gain over 70.20, because the deeper trunk was already less starved by
mis-weighting — the deficit differences I computed said as much, `5.89` of the gap sat between ResNet-20
and ResNet-56, so there is simply less for the lever to recover on the deeper backbone. On
**VGG-16-BN** I am genuinely unsure of the sign: PCGrad already hit 74.17 there, VGG has large-capacity
heads where the auxiliary signal matters least, and the `σ → 1` approximation in the classification
derivation is only first-order once the scales drift, so the learned weighting might land *near or even
slightly under* PCGrad on that backbone rather than above it. If the pattern comes out as "big
ResNet-20 recovery, modest ResNet-56 gain, roughly flat or slightly down VGG," that confirms the
magnitude lever is what the task rewards — but it also tells me what uncertainty weighting still cannot
do: it learns *one constant* weight per task and nothing time-aware beyond the slow fixed-point drift,
and that drift is slow precisely because `s_i` is one scalar crawling under the same SGD learning rate
as ten million network parameters. So if the real story is that the balance the trunk needs at epoch 20
is not the balance it needs at epoch 150 — a within-run schedule that a slowly-drifting learned *level*
cannot track — then this rung's static learned scale is as far as a level-based rule can climb, and the
next rung will have to read something the loss *level*, however cleverly learned, simply cannot see.
