The confidence penalty confirmed both halves of my worry, and the CIFAR-100 number is the headline.
On **resnet20-cifar10** the adaptivity paid off as predicted: the attack AUC dropped to 0.7130 (below
smoothing's 0.7678 and even ERM's 0.7275) at preserved 0.7974 accuracy, so privacy_score rose to
0.5844, the best of the simple rungs there. On **mobilenetv2-fmnist** a clean small win (0.8671,
essentially back to ERM's 0.8685, better than smoothing's 0.8460). But on **vgg16bn-cifar100** the run
*collapsed*: test_acc 0.01 — chance on a 100-class problem — with AUC 0.0026 and a meaningless
privacy_score of 0.01. Exactly the failure I flagged: a constant, unscheduled entropy push on the
highest-capacity model, fighting the cross-entropy while the learning rate decays, prevented the model
from ever committing to the correct class. So I now have the full shape of the wall. The three
mean-control rungs together show: (1) shifting confidence uniformly does nothing to separability; (2)
shifting it adaptively dents the attack on easy/moderate cases but never breaks it — the best AUC anyone
reached on CIFAR-10 is 0.7130, still a full 0.21 above the coin flip; (3) pushing a mean-control penalty
hard enough to matter on the hardest case destabilizes training, and the collapse landed to the digit I
computed, `1/K = 0.01`. None of them touched the attack's actual lever. The attack thresholds
member-versus-non-member confidence, and not one regularizer made the two distributions *overlap* — they
only relocated, compressed, or destroyed the member one. I need a method that targets overlap directly,
without the collapse risk.

Restart from what the attacker actually reads, the only thing worth fighting. There is a clean result
that the attack is essentially one-dimensional. Model the trained parameters as a posterior
`P(θ | z_1,…,z_n) ∝ exp(−(1/T) Σ_i ℓ(θ, z_i))`, single out one sample `z_1`, and ask for the
Bayes-optimal membership guess; the dependence on `θ` collapses and the optimal score is
`s(z_1) = (1/T)(τ(z_1) − ℓ(θ, z_1))`, where `τ(z_1)` is the typical loss `z_1` incurs under models that
*didn't* train on it. The optimal attacker — even white-box — reads exactly one number: the sample's
loss. The confidence the harness thresholds is a monotone proxy for it. A second result points the same
way: the simplest bounded-loss attacker's membership advantage equals the *generalization gap*
`R_gen = E_test[ℓ] − E_train[ℓ]`. So the job reduces to a statement about *distributions*: let `P` be the
per-sample loss distribution over members, `Q` over non-members. If a threshold separates `P` from `Q`
the attacker wins; if they overlap it cannot. Defense = make `P` and `Q` indistinguishable. That reframes
every failure above: plain cross-entropy crushes `P` to a tight spike near zero, well-separated from `Q`
(the 0.86–0.87 AUCs); smoothing and the confidence penalty *shifted or compressed* `P` but left it a
recognizable, separable hump. I have been moving `P` around; I need to make it *bleed into* `Q`.

And the generalization-gap result matches the columns I already have. If leakage rises with `R_gen`, the
benchmark with the larger train/test loss gap should show the larger AUC — and privacy_gap is a
confidence-domain proxy for exactly that gap. At the floor they line up strictly monotone: FashionMNIST
gap 0.0211 with AUC 0.5575, resnet 0.0924 with 0.7275, vgg 0.1479 with 0.8677. The target is not
"confidence" in the abstract but the *loss gap itself*, and CIFAR-100, with the widest gap and the run I
just destroyed, is where the work is — the method has to close that gap without the instability.

Before building the loss, confirm nothing outside `compute_loss` can do this more directly, because the
strongest privacy methods live there. Making `P` overlap `Q` by calibrated gradient noise — DP-SGD —
needs to clip and perturb per-example *gradients* in the optimizer step, not a scalar loss; out of reach.
Adversarial regularization needs an auxiliary model and a second optimizer; also outside a single
`compute_loss`. Distilling from a reference model — DMP/SELENA — needs a held-out reference set and an
extra training run; the loop gives me neither. Every method that beats a loss-level defense on paper
needs machinery the frozen loop forbids. What *is* expressible in one scalar per minibatch is a rule that
reshapes the member-loss distribution directly: hold its mean at a chosen level and inflate its variance.
The interface narrows me to a loss-shaping thermostat, and the only question is whether one scalar can
pull both levers at once.

Two things make two humps hard to tell apart: the distance between their means and their widths. Here is
the structural fact I had not used: the non-member distribution `Q` already has the *larger* variance —
never optimized, its losses scattered — while `P` is a tight spike near zero. The member hump is narrow
*and* low, doubly easy to threshold. So I want two things at once: bring `P`'s mean *up* toward `Q`'s,
*and* spread `P` out so it stops being a spike. The three prior rungs only ever attempted the first, and
rigidly. The second lever — variance — is the one nobody pulled, and it is the one that produces overlap.

The mean part I can simplify drastically. Manipulating full distributions is hopeless — I'd need a
hold-out non-member set to estimate `Q`, which a real defender does not have. So target the *mean*: pick
a target loss level `α > 0` *achievable for non-members too*, not the unreachable zero, and aim the
average member loss at `α`. While the batch mean is above `α` the model is undertrained — ordinary
descent; once it drops *below* `α`, do not keep descending (that re-opens the gap) but push it back *up*.
Pushing loss up is gradient *ascent*, and both directions fold into one objective the fixed optimizer
just minimizes: `|L − α|`. When `L > α`, `d|L−α|/dθ = +∇L`, descent; when `L < α`, `−∇L`, which under the
subtract-the-gradient rule is ascent. Descend above `α`, ascend below, settle at `α`. Crucially this is
*not* the confidence penalty's unbounded flattening — it does not keep pushing the output toward uniform
once the loss is high enough; it *holds* the loss at a finite, non-degenerate `α`. That is the relief
valve CIFAR-100 lacked: the objective stops fighting once the member loss reaches `α`, so it cannot drive
the model to chance.

At `L = α` the absolute value is non-differentiable and its subgradient spans `[−∇L, +∇L]`, which
contains zero — a genuine fixed point, an attractor at `L = α`. Line up the three attractors the ladder
has actually aimed at, because that is the whole story in one row. Cross-entropy's is `L = 0`: member
confidence `e^{−L} = 1`, maximal leak. The confidence penalty's is the uniform output: confidence `1/K`,
which on CIFAR-100 is 0.01 — the collapse I measured. The thermostat's is `L = α`: member confidence
about `e^{−α}`, a finite `0.607` at `α = 0.5` or `0.368` at `α = 1.0`, neither pinned at the ceiling nor
crushed to chance. Same optimizer, same loop; the difference between a leak, a collapse, and a defense is
entirely *where the loss objective places its attractor*, and this is the first rung to put that
attractor at a chosen interior point instead of at an extreme.

Now the part the prior rungs never got: does ascent also *spread* `P`? Take one sample's loss change
under the batch ascent step `θ → θ + τ∇L` and Taylor-expand:
`Δℓ_i = (τ/B)‖∇ℓ_i‖² + (τ/B)Σ_{j≠i}⟨∇ℓ_i, ∇ℓ_j⟩ + o(τ)`. Under comparable gradient norms and
non-negative alignment this is `Δℓ_i ≈ c_1‖∇ℓ_i‖² + c_2` with `c_1 > 0`. Connect the norm to the loss:
for cross-entropy `∇_θ ℓ = J^T(p − y)`, so as a sample's loss shrinks `‖p − y‖ → 0` and `‖∇ℓ‖² → 0` —
small for low-loss samples, large for high-loss ones, so `Cov(‖∇ℓ‖², ℓ) > 0`. Chaining,
`Cov(ℓ, Δℓ) = c_1 Cov(ℓ, ‖∇ℓ‖²) > 0`: high-loss samples get pushed up *more* than low-loss ones, which
is a variance-increasing operation since
`Var(ℓ + Δℓ) = Var(ℓ) + Var(Δℓ) + 2Cov(ℓ, Δℓ) > Var(ℓ)`. The ascent step *fattens* `P` — it does not
just shift the mean, it stretches the tail. This is the lever the mean-shifting regularizers structurally
could not pull: smoothing and the confidence penalty push every member up by roughly the same amount,
leaving `Var(P)` essentially untouched (which is *why* their distributions stayed separable in the
numbers); ascent pushes the already-high losses up harder, fattening `P` until it overlaps the wider `Q`.
Both levers — mean toward `α`, variance up — fall out of the one absolute-value objective.

That this lowers the attack is not hand-waving. Treat the attack as a hypothesis test `P` vs `Q`; the
optimal AUC is bounded by the total-variation distance, `AUC ≤ −½ D_TV² + D_TV + ½`, increasing in
`D_TV` on `[0,1]`, and `D_TV ≤ √2 D_H` by the Hellinger distance, which for Gaussians has a closed form.
Shrinking the mean gap `(μ_1 − μ_2)²` and raising the member variance `σ_1²` (with `σ_2` fixed) both push
the Hellinger factor toward 1, and since `Q` started wider (`σ_2 ≥ σ_1`), fattening `P` moves the
variance ratio toward equality where the prefactor is maximized. Both levers shrink `D_H`, hence `D_TV`,
hence the AUC ceiling — exactly the two moves I derived, exactly the metric the harness reports.

Now the cost, because relaxing member loss threatens utility in a specific way. If I let `p_gt` (the
predicted probability on the true class) sit lower and the leftover mass `1 − p_gt` concentrates on one
competing class — what happens on hard samples near a two-class boundary — some wrong class can exceed
`p_gt` and the argmax flips: I have relaxed confidence into misclassification. The fix must *preserve*
the privacy gain (don't re-sharpen `p_gt` toward 1, or I undo the defense) while protecting the argmax,
which is safe as long as `p_gt` beats every competitor. So at *fixed* `p_gt` maximize the margin — make
the largest competitor as small as possible — by spreading `1 − p_gt` *evenly* over the `K − 1` wrong
classes, each getting `(1 − p_gt)/(K − 1)`. Train toward that target:

  `t^c = p_gt` if `c` is the true class, else `t^c = (1 − p_gt)/(K − 1)`.

Posterior flattening: don't change how confident I am in the truth, only equalize the doubt so no single
rival can overtake. Two details matter. `t` is built from the model's *own* current `p_gt`, read off the
forward pass, so it must be a *constant* — stop-gradient on `t`, or the gradient would flow back through
the `p_gt` inside `t` and start re-sharpening the confidence I am trying to leave alone. And *which*
samples to flatten: a correctly-classified sample's argmax is already fine, so flattening it perturbs a
settled prediction, while the samples that need margin protection are the *misclassified* ones — so gate
the soft cross-entropy by `(1 − correct_i)`. Meanwhile keep pushing the loss up by subtracting the
per-sample cross-entropy, so `loss_i = (1 − correct_i)·ℓ_soft_i − ℓ_CE_i`, batch-mean: the `−ℓ_CE_i` is
ascent (keeps the loss elevated and spreading), the gated `ℓ_soft_i` fixes the argmax of exactly the
samples that need it at fixed `p_gt`.

Assemble into the two-phase rule, using the `epoch` argument the loop hands me — the prior rungs ignored
it, this is the first method that needs it. On **even epochs**, the thermostat directly: `loss = |L − α|`
— descent above `α`, ascent (with its free variance-spreading) below. On **odd epochs**, if `L > α` the
model is still undertrained so ordinary descent, `loss = L`; only when `L ≤ α` does the
posterior-flattening branch run, `loss = mean_i[(1 − correct_i)·ℓ_soft_i − ℓ_CE_i]`. Two hyperparameters.
`upper` clamps `p_gt` into `[0, upper]` before building `t`; at `upper = 1.0` it is a no-op (it cannot
clamp a probability below 1), kept for faithfulness. `α` is the real knob — a loss level reachable by
non-members — selected by class count read from `logits.size(1)`: `α = 0.5` when `K = 100` (CIFAR-100)
and `α = 1.0` otherwise. That per-dataset `α` is the one place the method adapts, read at call time from
the logits — exactly what the fixed signature allows.

The direction of that split is worth reasoning out, because at first glance it looks backwards. Higher
`α` holds member loss higher — more relaxation, more privacy, but more accuracy risk, since the attractor
confidence `e^{−α}` drops (0.368 at `α = 1.0` versus 0.607 at `α = 0.5`). CIFAR-100 gets the *lower*
`α = 0.5`, the *gentler* relaxation — exactly right for the fragile high-capacity 100-class case my
previous rung collapsed: spend less relaxation budget there, hold the loss closer to where the model can
actually sit, and accept a little more residual leakage rather than risk tipping accuracy again. The
robust 10-class benchmarks fit easily and can absorb the aggressive `α = 1.0` for more privacy. There is
also a hard floor the choice must respect: `α` must sit *below* the typical non-member loss, or the batch
mean would essentially always exceed `α`, the ascent branch would never fire, and the method would decay
to plain cross-entropy — no defense at all. A modest `α` on both K-regimes keeps me inside the band where
ascent fires.

One assembly choice deserves its reason: why alternate the thermostat and the flattening by epoch parity
instead of summing them each batch. Both branches contain an ascent term — the thermostat's `|L − α|`
ascends below `α`, the flattening branch carries its own `−ℓ_CE` ascent — so adding them would double the
upward push and let the soft-CE term swamp the clean loss-level control, re-introducing exactly the "one
force overpowers the other" instability that cratered the confidence penalty. Alternating gives each
mechanism uncontaminated batches: even epochs do pure loss-level thermostatting (with its free
variance-spreading), odd epochs do argmax repair only once the loss is already near `α`. The parity
switch is deterministic in `epoch`, so it costs nothing and needs no state.

The expectations against the prior numbers. The decisive prediction is on **vgg16bn-cifar100**, where the
confidence penalty *collapsed* to 0.01. The thermostat holds member loss at a finite `α = 0.5` rather
than a degenerate point, and the flattening branch protects the argmax of misclassified samples, so
accuracy should *not* collapse — I expect a recovery to genuine, non-trivial accuracy above ERM's 0.5045
(the relaxation acting as a regularizer), while the variance-spreading drags the attack AUC well down
from the 0.86–0.87 the mean-control rungs were stuck at. That would lift the CIFAR-100 privacy_score
several-fold above the wreckage (0.01) and the ERM floor (0.1368). On **resnet20-cifar10** I expect the
attack to drop hardest of all — overlap, not just a shift — plausibly toward the 0.5 coin-flip floor at
roughly ERM-level accuracy, so privacy_score should clear every prior rung's ~0.57–0.58 by a wide margin
and approach test_acc itself. On **mobilenetv2-fmnist**, where leakage was already mild, I expect the AUC
at or near 0.5 with the privacy term essentially zero, though the relaxation may cost some accuracy on an
already-easy task, so privacy_score could sit a little below the ~0.87 of the easy rungs even as the
attack is fully defused. The clean, falsifiable signature distinguishing this method from the three that
failed: the attack AUC moving *toward 0.5* — overlap — rather than the member distribution merely
relocating, and CIFAR-100 *not* collapsing. If those hold, the variance lever is what broke the wall. The
distilled scaffold fill is in the answer.
