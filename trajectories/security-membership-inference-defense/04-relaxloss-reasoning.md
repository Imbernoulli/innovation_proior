The confidence penalty confirmed both halves of my worry, and the CIFAR-100 number is the headline.
On **resnet20-cifar10** the adaptivity paid off as predicted: the attack AUC dropped to 0.7130 (below
smoothing's 0.7678 and even ERM's 0.7275) at preserved 0.7974 accuracy, so privacy_score rose to
0.5844, the best of the simple rungs there. On **mobilenetv2-fmnist** it was a clean small win
(0.8671, essentially back to ERM's 0.8685, better than smoothing's 0.8460). But on
**vgg16bn-cifar100** the run *collapsed*: test_acc 0.01 — chance on a 100-class problem — with AUC
0.0026 and a meaningless privacy_score of 0.01. That is exactly the failure I flagged: a constant,
unscheduled entropy push on the highest-capacity model, fighting the cross-entropy while the learning
rate decays, prevented the model from ever committing to the correct class. So I now have the full
shape of the wall. The three mean-control rungs together show: (1) shifting confidence uniformly
(smoothing) does nothing to separability; (2) shifting it adaptively (confidence penalty) dents the
attack on easy/moderate cases but never breaks it; and (3) pushing a mean-control penalty hard enough
to matter on the hardest case destabilizes training. None of them touched the attack's actual lever.
The attack thresholds member-versus-non-member confidence, and not one of these regularizers made the
two distributions *overlap* — they only relocated, compressed, or destroyed the member one. I need a
method that targets overlap directly, and does so without the collapse risk.

Let me restart from what the attacker actually reads, because that is the only thing worth fighting.
There is a clean result that says the attack is essentially one-dimensional. Model the trained
parameters as a posterior `P(θ | z_1,…,z_n) ∝ exp(−(1/T) Σ_i ℓ(θ, z_i))`, single out one sample `z_1`,
and ask for the Bayes-optimal membership guess. Push the algebra through and the dependence on `θ`
collapses: the optimal score is `s(z_1) = (1/T)(τ(z_1) − ℓ(θ, z_1))`, where `τ(z_1)` is the typical
loss `z_1` incurs under models that *didn't* train on it. The optimal attacker — even white-box — reads
exactly one number: the sample's loss `ℓ(θ, z_1)`. The confidence the harness thresholds is a monotone
proxy for that loss. And there is a second result pointing the same way: Yeom and colleagues showed the
simplest bounded-loss attacker has membership advantage equal to the *generalization gap*,
`R_gen = E_test[ℓ] − E_train[ℓ]`. So the entire job reduces to a statement about *distributions*: let
`P` be the per-sample loss distribution over members, `Q` over non-members. If a threshold separates
`P` from `Q`, the attacker wins; if they overlap, it cannot. Defense = make `P` and `Q`
indistinguishable. That reframes every failure above. Plain cross-entropy crushes `P` to a tight spike
near zero, well-separated from `Q` (the 0.84–0.87 AUCs). Smoothing and the confidence penalty *shifted
or compressed* `P` but left it a recognizable, separable hump. I have been moving `P` around; I need to
make it *bleed into* `Q`.

Two things make two humps hard to tell apart: the distance between their means, and their widths. And
here is the structural fact I had not used: the non-member distribution `Q` already has the *larger*
variance — never optimized, its losses are scattered — while `P` is a tight spike near zero. The member
hump is narrow *and* low; doubly easy to threshold. So I want two things at once: bring `P`'s mean *up*
toward `Q`'s, *and* spread `P` out so it stops being a spike. The three prior rungs only ever attempted
the first, and even that rigidly. The second lever — variance — is the one nobody pulled, and it is the
one that produces overlap.

The mean part I can simplify drastically. Manipulating full distributions is hopeless — I would need a
hold-out non-member set just to estimate `Q`, which a real defender does not have. So target the
*mean*. Pick a target loss level `α > 0` that is *achievable for non-members too*, not the unreachable
zero, and aim the average member loss at `α` instead of at 0. Concretely: while the batch mean loss is
above `α`, the model is genuinely undertrained — do ordinary descent. Once the mean drops *below* `α`,
do not keep descending (that re-opens the gap); push it back *up* to `α`. Pushing loss up is gradient
*ascent*. I can fold both directions into one objective the fixed optimizer just minimizes:
`|L − α|`. When `L > α`, `d|L−α|/dθ = +∇L`, descent; when `L < α`, `d|L−α|/dθ = −∇L`, which under the
minimizer's subtract-the-gradient rule is ascent. One absolute-value loss is the whole thermostat:
descend above `α`, ascend below, settle at `α`. Crucially, this is *not* the confidence penalty's
unbounded flattening — it does not keep pushing the output toward uniform once the loss is high enough;
it *holds* the loss at a finite, non-degenerate `α`. That is the relief valve CIFAR-100 lacked: the
objective stops fighting once the member loss reaches `α`, so it cannot drive the model to chance.

Now the part the prior rungs never got: does ascent also *spread* `P`? Take one sample's loss change
under the batch ascent step `θ → θ + τ∇L` and Taylor-expand:
`Δℓ_i = (τ/B)‖∇ℓ_i‖² + (τ/B)Σ_{j≠i}⟨∇ℓ_i, ∇ℓ_j⟩ + o(τ)`. Under comparable gradient norms and
non-negative alignment this is `Δℓ_i ≈ c_1‖∇ℓ_i‖² + c_2` with `c_1 > 0`. Now connect the norm to the
loss: for cross-entropy `∇_θ ℓ = J^T(p − y)`, so as a sample's loss shrinks, `p → y`, `‖p − y‖ → 0`,
and `‖∇ℓ‖² → 0`. The gradient norm is small for low-loss samples and large for high-loss ones, so
`Cov(‖∇ℓ‖², ℓ) > 0`. Chain them: `Cov(ℓ, Δℓ) = c_1 Cov(ℓ, ‖∇ℓ‖²) > 0`. High-loss samples get pushed
up *more* than low-loss ones — and that is a variance-increasing operation, since
`Var(ℓ + Δℓ) = Var(ℓ) + Var(Δℓ) + 2Cov(ℓ, Δℓ) > Var(ℓ)`. The ascent step *fattens* `P` — it does not
just shift the mean, it stretches the tail. This is the lever the mean-shifting regularizers structurally
cannot pull: smoothing and the confidence penalty push every member up by roughly the same amount,
leaving `Var(P)` essentially untouched (which is *why* their distributions stayed separable in the
numbers); ascent pushes the already-high losses up harder, fattening `P` until it overlaps the wider
`Q`. Both levers — mean toward `α`, variance up — fall out of the one absolute-value objective.

That this lowers the attack is not hand-waving. Treat the attack as a hypothesis test `P` vs `Q`; the
optimal attacker's AUC is bounded by the total-variation distance,
`AUC ≤ −½ D_TV(P,Q)² + D_TV(P,Q) + ½`, increasing in `D_TV` on `[0,1]`. Bound `D_TV ≤ √2 D_H` by the
Hellinger distance, which for Gaussians `P ~ N(μ_1, σ_1²)`, `Q ~ N(μ_2, σ_2²)` has a closed form.
Shrinking the mean gap `(μ_1 − μ_2)²` and raising the member variance `σ_1²` (with `σ_2` fixed) both
raise the Hellinger exponential factor toward 1, and since `Q` started wider (`σ_2 ≥ σ_1`), fattening
`P` moves the variance ratio toward equality where the prefactor is maximized. So both levers shrink
`D_H`, hence `D_TV`, hence the AUC ceiling. Exactly the two moves I derived, exactly the metric the
harness reports.

Now the cost, because relaxing member loss threatens utility in a specific way. If I let `p_gt` (the
predicted probability on the true class) sit lower and the leftover mass `1 − p_gt` happens to
concentrate on one competing class — which is what happens on hard samples near a two-class decision
boundary — some wrong class can exceed `p_gt` and the argmax flips. I have relaxed confidence into
misclassification. The fix must *preserve* the privacy gain (do not re-sharpen `p_gt` toward 1, or I
undo the defense) while protecting the argmax. The argmax is safe as long as `p_gt` beats every
competitor, so at *fixed* `p_gt` I want to maximize the margin, i.e. make the largest competitor as
small as possible — which means spreading `1 − p_gt` *evenly* over the `K − 1` wrong classes, each
getting `(1 − p_gt)/(K − 1)`. Construct that target and train toward it:

  `t^c = p_gt` if `c` is the true class, else `t^c = (1 − p_gt)/(K − 1)`.

This is *posterior flattening*: do not change how confident I am in the truth, only equalize the doubt
so no single rival can overtake. Two details matter. The target `t` is built from the model's *own*
current `p_gt`, read off the forward pass, so it must be treated as a *constant* — apply stop-gradient
to `t`, or the gradient would flow back through the `p_gt` inside `t` and start re-sharpening the
confidence I am trying to leave alone. And *which* samples to flatten: a correctly-classified sample's
argmax is already fine, so flattening it spends effort on rivals that are not winning and can perturb a
settled prediction. The samples that need margin protection are the *misclassified* ones. So gate the
soft cross-entropy by `(1 − correct_i)`. Meanwhile keep pushing the loss up by subtracting the
per-sample cross-entropy, so the per-sample objective is
`loss_i = (1 − correct_i)·ℓ_soft_i − ℓ_CE_i`, batch-mean. The `−ℓ_CE_i` term is ascent (keeps the loss
elevated and spreading); the gated `ℓ_soft_i` fixes the argmax of exactly the samples that need it, at
fixed `p_gt`.

Now assemble it into the two-phase rule the scaffold lands, and note exactly what this harness exposes.
I alternate by epoch parity using the `epoch` argument the loop hands me — the prior rungs ignored it;
this is the first method that needs it. On **even epochs**, use the thermostat directly:
`loss = |L − α|` — descent above `α`, ascent (with its free variance-spreading) below. On **odd
epochs**, if `L > α` the model is still undertrained, so ordinary descent, `loss = L`; only when
`L ≤ α` does the posterior-flattening branch run,
`loss = mean_i[(1 − correct_i)·ℓ_soft_i − ℓ_CE_i]`. Alternating keeps the flattening branch from
swamping the ascent step. Two hyperparameters. `upper` clamps `p_gt` into `[0, upper]` before building
`t`; with `upper = 1.0` it is a no-op (it cannot clamp a probability below 1), kept for faithfulness to
the reference, and that is the value this benchmark uses. `α` is the real knob — a loss level reachable
by non-members — and this harness selects it by class count read from `logits.size(1)`: `α = 0.5` when
`K = 100` (CIFAR-100) and `α = 1.0` otherwise (the 10-class CIFAR-10 and FashionMNIST cases). That
per-dataset `α` is the one place the method adapts, and it is read at call time from the logits, not
configured externally — exactly what the fixed `compute_loss` signature allows.

The falsifiable expectations against the prior numbers. The decisive prediction is on
**vgg16bn-cifar100**, where the confidence penalty *collapsed* to 0.01. The thermostat holds the member
loss at a finite `α = 0.5` rather than driving it to a degenerate point, and the flattening branch
protects the argmax of misclassified samples, so accuracy should *not* collapse — I expect it to recover
to genuine, non-trivial accuracy (well above ERM's 0.5045, plausibly into the low-0.6s as the relaxation
acts as a regularizer), while the variance-spreading drags the attack AUC well down from the
0.86–0.87 the mean-control rungs were stuck at — toward the mid-0.6s. That combination would lift the
CIFAR-100 privacy_score from the wreckage (0.01) and the ERM floor (0.1368) to something several-fold
larger. On **resnet20-cifar10** I expect the attack to drop hardest of all — overlap, not just a shift —
plausibly to near the 0.5 coin-flip floor, at roughly ERM-level accuracy, so privacy_score should clear
every prior rung's ~0.57–0.58 by a wide margin and approach test_acc itself (the regime where the
attack is neutralized). On **mobilenetv2-fmnist**, where leakage was already mild, I expect the AUC at
or near 0.5 (the privacy term essentially zero), though here the relaxation may cost some accuracy on
an already-easy task, so privacy_score could sit a little below the ~0.87 of the easy rungs even as the
attack is fully defused. The clean, falsifiable signature that distinguishes this method from the three
that failed: the attack AUC moving *toward 0.5* — overlap — rather than the member distribution merely
relocating, and CIFAR-100 *not* collapsing. If those hold, the variance lever is what broke the wall.
The distilled scaffold fill is in the answer.
