The PGD result settles the question I posed and reframes it into a harder one. Dropping the momentum
and taking faithful larger steps did sharpen the maximizer exactly where I predicted: VGG11BN-C10
ticked 0.714 → 0.720 and VGG11BN-C100 0.720 → 0.736, with the easy scenarios holding at the ceiling
(0.992 / 0.988 on ResNet, 0.975 on MobileNetV2-C100). So part of the VGG residual *was* a blunted
maximizer — a better single-loss gradient climber recovered a couple of points. But look at the size of
that recovery: `0.720 − 0.714 = 0.006` on C10 and `0.736 − 0.720 = 0.016` on C100, an average of about
one VGG sample in a hundred. Set that beside the jump iterating bought over FGSM — `~0.10` on VGG — and
the trend is unmistakable: each successive improvement to the *gradient climber* returns an order of
magnitude less on VGG than the last. FGSM→MI-FGSM bought `~0.10`, MI-FGSM→PGD bought `~0.01`. A curve
that decays that fast is telling me the single-loss gradient climber has essentially converged, and the
`~28%` VGG survivors (`1 − 0.720`, `1 − 0.736`) are not going to fall to a fourth retuning of the same
kind of run. PGD is, by the Danskin argument, the principled strongest *first-order, single-loss,
single-run* attack I can build, and it has now plateaued on VGG. That tells
me the remaining headroom is no longer about climbing the cross-entropy surface better. Two failure
modes survive a strong fixed-step CE-PGD, and both are invisible from inside that single run, so I have
to attack them from outside it.

First failure mode: the *step schedule* itself. CE-PGD fixes `alpha = eps/4` for all 40 steps. A fixed
step cannot be both large enough for early exploration across the `2eps`-wide box and small enough for
late local refinement on the high-loss boundary. On a sample where the surface bends sharply near the
boundary, a fixed `eps/4` step keeps overshooting the local maximum and bouncing — the best-so-far loss
jumps early and then stalls, and the iteration count stops being a reliable measure of attack strength.
I patched this informally with "40 steps, total reach `10·eps`, so late steps refine," but a fixed step
does not *know* whether it is still improving; it cannot adapt. The honest fix is to make the step size
a function of the trajectory's own progress: start large to explore, and halve it when the iterates
stop making progress, restarting from the best point found so far. That removes the one tuned knob
(`alpha`) that I have been guessing per-method and replaces it with a budget-aware schedule.

The fix is an adaptive schedule: start large at `eta = 2·eps` (enough to cross the box before
projection and land at an informative boundary point), keep the best point `x_max` and best objective
seen, and at progress checkpoints halve `eta` whenever too few steps in the window actually increased
the objective — the step is overshooting — or the search has stalled at the current scale. The
checkpoints are spaced with a long opening window that shrinks toward a small floor, so a stall is
caught within a few steps late in the budget instead of wasting a quarter of it; each halving restarts
the iterate from `x_max`, so a smaller step always refines the best neighborhood the larger step found.
A heavy-ball blend `x_{next} = P_S(x + 0.75·(z − x) + 0.25·(x − x_prev))` on the projected step damps
the ping-pong of the deliberately-large early moves — without the staleness that made me drop momentum
before, because here it is coupled to a step that halves when progress stalls, so it cannot keep
pushing a wrong direction across a scale change. This is step-size-free PGD whose only knob is the
iteration budget: a strictly better version of the maximizer I hand-tuned earlier.

The second failure mode is subtler and is the one I think actually pins VGG: the *loss*. Cross-entropy
has a scale degree of freedom the decision rule does not. Its input gradient is
`(p_y − 1)∇z_y + Σ_{i≠y} p_i ∇z_i`; when the true-class probability `p_y` is near one — the
overconfident, well-classified VGG survivor — all those coefficients are near zero and in finite
precision can *be* zero. A decision-preserving rescaling `z → α·z` sharpens the softmax and drives the
gradient toward zero while the classifier behaves identically. Make it a number: on `z = (4, 1, 0)`,
`p_y = 0.936` and the coefficient `p_y − 1 = −0.064`, already small; rescale by `α = 4` to `(16, 4, 0)`
— same argmax — and `p_y = 1 − 5.6·10^{-6}`, so the gradient has shrunk four orders of magnitude under a
transformation the classifier does not notice. A fixed-step optimizer reading that flat gradient reports
the sample robust when the only thing robust is the loss surface I chose to climb. The fix is a loss with
the decision's invariances — shift- and positive-scale-invariant. A margin `−z_y + max_{i≠y} z_i` is
shift-invariant but not scale-invariant, so I need a *ratio* of logit differences. Sorting
`z_{π1} ≥ z_{π2} ≥ …`, use `DLR(x,y) = −(z_y − max_{i≠y} z_i) / (z_{π1} − z_{π3})`: the numerator is the
decision margin, and the `1st−3rd` denominator measures the logit scale while keeping the top-two
competition the attack is closing out of the denominator (avoiding `0/0` at the boundary). On the same
example `DLR = −3/4 = −0.75` and after rescaling `−12/16 = −0.75` — identical where CE collapsed by
`10^4`, since `α` cancels top and bottom. The `π_1 − π_3` choice is deliberate: with `π_1 − π_2` the
denominator would collapse and the ratio blow up exactly as the runner-up overtakes the true class,
whereas the third-place reference stays well-conditioned through the flip.

But even an adaptive-step DLR attack is still *one* gradient attack, and any single gradient attack has
a failure mode some configuration will sit on — discontinuous, flat, noisy, or otherwise uninformative
gradients. So the strongest honest move is not a better single attack but a small *ensemble* whose
members fail for different reasons, with the result taken as the worst case: a sample is robust only if
*none* flips it. Four members, four disjoint failure modes. Untargeted CE-APGD reduces true-class
confidence, useful under any output stochasticity. Targeted DLR-APGD keeps a gradient exactly on the
saturated overconfident points where CE goes flat. Targeted FAB linearizes the decision boundary and
minimizes perturbation *norm* to the nearest boundary rather than maximizing a fixed-ball loss, so it
does not care whether the loss surface is saturated — a point where both APGD objectives are flat can
still have a nearby boundary FAB walks to. And Square Attack uses *no gradient at all* — score-based
random search over sign-pattern squares — the backstop against gradient masking, where every first-order
member is blind at once. The worst-case aggregation is not a heuristic vote but the honest estimator of
robust accuracy, "the fraction no admissible attack can flip": with a finite toolbox the best
downward-unbiased estimate of that min is the min over the tools I have, and adding a member can only
lower the reported robust accuracy or leave it unchanged, never inflate it. Each member earns its place
by covering a failure the others share.

Run them sequentially on the shrinking set of still-robust points — mark the initially-correct
points robust, run APGD-CE and remove what it flips, then targeted DLR-APGD on the survivors, then FAB,
then Square — so the worst-case-over-attacks robust accuracy is computed without wasting queries on
already-broken samples. Standard fixed settings: APGD-CE one 100-iteration run, targeted DLR-APGD one
100-iteration run over the 9 top wrong classes, targeted FAB likewise, Square with 5000 queries. This
is a parameter-free evaluation built from two new pieces — the adaptive APGD schedule and the
scale-invariant DLR loss — plus complementary existing attacks chosen for diversity, not for another
tunable knob.

The whole ensemble already exists in the `torchattacks` package where the reference baselines live, so I
do not reimplement it — I fill `run_attack` with the single standard-version call
`torchattacks.AutoAttack(model, norm="Linf", eps=eps, version="standard", n_classes=n_classes,
seed=SEED, verbose=False)` on `(images, labels)`. Reimplementing the four members is precisely where I
would risk a subtle bug that silently *lowers* ASR and over-reports robustness — a DLR denominator
dividing by `π_1 − π_2` instead of `π_1 − π_3`, a mis-set APGD checkpoint — which is exactly the failure
this step exists to prevent, so the disciplined move is the reference call whose behavior is fixed, with
only `n_classes` and `seed` wired through. `n_classes` matters because it sets how many targeted classes
each targeted member sweeps (the standard ensemble aims each at the `9` highest-scoring wrong classes,
so `9` of `9` on CIFAR-10 and `9` of `99` on CIFAR-100); `seed` reads from the environment for
reproducibility under seed 42; `device` is unused since the package places tensors itself. The cost is
steep but affordable: a targeted member runs `9 × 100 = 900` gradient steps per sample, so with two
targeted members plus APGD-CE and Square's 5000 queries it is `6000+` evaluations per sample at the top
of the funnel versus PGD's 40 — but the sequential funnel runs each stage only on survivors, so APGD-CE
flips the easy `~0.99` on the first pass and only the `~28%` VGG survivors ever reach the expensive
targeted and Square stages. The spend concentrates exactly where the measurement is contested.

The bar this endpoint must clear, against PGD's numbers: match PGD at the ceiling on the already-solved
scenarios (ResNet near 0.99, MobileNetV2 near 0.975–1.0, since the worst case over the ensemble can only
hold or raise them). The decisive test is VGG. If the CE *loss* was what pinned it, the scale-invariant
DLR member flips samples CE-PGD could not get a gradient on, and VGG ASR sits at or above PGD's 0.720 /
0.736. But I expect a near-tie rather than a jump: PGD at `eps/4` over 40 steps is already a strong CE
maximizer, and AutoAttack's edge here is *reliability*, not raw ASR — most VGG survivors on these
undefended models are architectural `L_inf` robustness no in-budget attack touches. So the honest claim
is "no worse than PGD on every scenario, with the VGG number now trustworthy" — a measurement no
decision-preserving rescaling or single-attack blind spot can inflate. If AutoAttack instead came in
*below* PGD on VGG, that would mean fixed-step CE-PGD had been over-reporting via a surrogate-specific
quirk, and the ensemble's lower worst-case number is the more honest one — exactly the property that
makes this the right place to stop.

And it is the place to stop by construction, not for running out of ideas: each earlier attack
corrected a failure I could point to in the previous number, and after this one I cannot. The step
schedule is adaptive, the loss is invariant to the degrees of freedom the decision ignores, Square
closes the gradient-masking hatch, and worst-case-over-ensemble aggregation means more members can only
ratify the number, not overturn it. By elimination the remaining VGG survivors are where the model
genuinely has no within-budget adversarial neighbor — architectural robustness, not a surrogate
artifact.
