The PGD result settles the question I posed and reframes it into a harder one. Dropping the momentum
and taking faithful larger steps did sharpen the maximizer exactly where I predicted: VGG11BN-C10
ticked 0.714 → 0.720 and VGG11BN-C100 0.720 → 0.736, with the easy scenarios holding at the ceiling
(0.992 / 0.988 on ResNet, 0.975 on MobileNetV2-C100). So part of the VGG residual *was* a blunted
maximizer — a better single-loss gradient climber recovered a couple of points. But the gain is small
and the level is still ~0.72–0.74. PGD is, by the Danskin argument, the principled strongest
*first-order, single-loss, single-run* attack I can build, and it has now plateaued on VGG. That tells
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

Concretely, start at `eta = 2·eps` — in `L_inf` a single signed step of `2·eps` can cross the whole
box before projection, so the projected point lands at an informative boundary location — and keep the
best point `x_max` and best objective seen so far, because the raw iterate is not monotone. Put
checkpoints into the budget at fractions `p_0 = 0`, `p_1 = 0.22`, `p_{j+1} = p_j + max(p_j − p_{j−1} −
0.03, 0.06)`: a long first window to explore, later windows shrinking until they hit a `0.06N` floor,
so progress is checked more often as the search becomes local. At each checkpoint, halve `eta` if too
few steps in the window actually increased the objective (a count below `0.75` of the window — the step
is overshooting too often) or if the step was not reduced last checkpoint *and* the best objective has
not improved since (the search has stalled at a scale it cannot refine). Whenever I halve, restart the
iterate from `x_max`, because a smaller step should refine the best neighborhood the larger step found.
A momentum blend on the projected step stabilizes the deliberately-large early moves:
`z = P_S(x + eta·dir(grad))`, then `x_{next} = P_S(x + 0.75·(z − x) + 0.25·(x − x_prev))`. This is an
adaptive, step-size-free PGD whose only remaining knob is the iteration budget — strictly a better
version of the inner maximizer I built at rung three, fixing the schedule I had been hand-tuning.

Second failure mode is subtler and is the one I think actually pins VGG: the *loss*. I have been
climbing cross-entropy throughout, and cross-entropy has a scale degree of freedom the decision rule
does not. `CE(x,y) = −z_y + log Σ_j exp(z_j)`, and its input gradient is `(p_y − 1)∇z_y + Σ_{i≠y}
p_i ∇z_i`. When the true-class probability `p_y` is near one — an overconfident, well-classified point,
which is exactly the VGG survivor — all those coefficients are near zero, and in finite precision can
*be* zero. The gradient vanishes not because the point is hard to move but because the softmax has
saturated, and a decision-preserving rescaling of the logits (`h = α·g`, same argmax for every input)
sharpens the softmax and can drive the CE gradient to zero while the classifier behaves identically.
So a CE-PGD run can report a VGG sample as robust when the only thing robust is the *loss surface I
chose to climb*, not the model. The fix is to climb a loss with the same invariances as the decision:
shift-invariant and positive-scale-invariant. A margin `−z_y + max_{i≠y} z_i` is shift-invariant
(difference of logits) but not scale-invariant; I need a *ratio* of logit differences so the scale
cancels. Sort the logits `z_{π1} ≥ z_{π2} ≥ z_{π3} ≥ …` and use
`DLR(x,y) = −(z_y − max_{i≠y} z_i) / (z_{π1} − z_{π3})`: the numerator is the decision margin (so the
loss turns positive exactly when a wrong class overtakes `y`), and the `1st−3rd` denominator keeps the
top-two competition the attack is trying to close *out* of the denominator (avoiding a `0/0` at the
boundary) while measuring the logit scale. Being a ratio of differences, it is invariant to the exact
shift-and-scale that blinds cross-entropy — so it keeps a usable gradient precisely on the saturated,
overconfident VGG points where CE goes flat.

But even an adaptive-step DLR attack is still *one* gradient attack, and any single gradient attack has
a failure mode some configuration will sit on — discontinuous, flat, noisy, or otherwise uninformative
gradients. So the strongest honest move is not a better single attack but a small *ensemble* whose
members fail for different reasons, with the final result taken as the worst case over all of them: a
sample counts as flipped if *any* member flips it. Keep an untargeted CE-APGD run (reducing true-class
confidence is useful, especially under any output stochasticity). Add a targeted DLR-APGD run that
aims at the highest-scoring wrong classes — the scale-invariant gradient attack with a sharp objective.
Add targeted FAB, which searches by linearizing decision boundaries and minimizing perturbation norm
rather than maximizing a fixed-ball loss, so it fails differently from APGD. And add Square Attack, a
score-based random search using *no gradients at all*, the backstop against any form of gradient
masking. Run them sequentially on the shrinking set of still-robust points — mark the initially-correct
points robust, run APGD-CE and remove what it flips, then targeted DLR-APGD on the survivors, then FAB,
then Square — so the worst-case-over-attacks robust accuracy is computed without wasting queries on
already-broken samples. Standard fixed settings: APGD-CE one 100-iteration run, targeted DLR-APGD one
100-iteration run over the 9 top wrong classes, targeted FAB likewise, Square with 5000 queries. This
is a parameter-free evaluation built from two new pieces — the adaptive APGD schedule and the
scale-invariant DLR loss — plus complementary existing attacks chosen for diversity, not for another
tunable knob.

Now the crucial scaffold-grounding point, because the trajectory's code must match what this task's
harness actually runs, not the from-scratch construction I just derived. The reference implementation
of this entire ensemble lives in the `torchattacks` package, and the task's edit fills `run_attack`
with the single library call `torchattacks.AutoAttack(model, norm="Linf", eps=eps,
version="standard", n_classes=n_classes, seed=SEED, verbose=False)` applied to `(images, labels)`. So
the literal edit on this surface is *one line of construction plus the call* — the APGD schedule, the
DLR loss, FAB, and Square are all inside the package; the harness does not ask me to reimplement them,
it asks me to invoke the standard ensemble correctly. Two interface details matter and the fill gets
them right: `n_classes` must be passed (10 vs 100 changes how many targeted classes the targeted
members sweep — for CIFAR-100 the 9 targets are drawn from 99 wrong classes), and the `seed` is read
from the environment so the run is reproducible under the harness's single seed 42. Unlike the prior
rungs, `device` is unused here because the package places tensors itself; I bind it to a throwaway. The
derivation above is *why* this call is the right endpoint — it is the worst case over a diverse,
adaptive, parameter-free ensemble — but the code is the invocation, and that is what the answer shows.

The bar this endpoint must clear, against PGD's real numbers, with the failure modes named. On the easy
scenarios it should match PGD at the ceiling — ResNet20-C10/C100 near 0.99, MobileNetV2-C100 near
0.975–0.98 — because those are already solved and the worst-case-over-ensemble can only hold or
slightly raise them (MobileNetV2-C10 should reach ~1.0, the column PGD did not report). The decisive
test is again VGG. If the cross-entropy *loss* was what pinned VGG — saturated softmax gradients on
overconfident survivors — then the scale-invariant DLR member should flip samples CE-PGD could not even
get a gradient on, and the worst-case-over-ensemble VGG ASR should sit at or above PGD's 0.720 and
0.736. I genuinely expect a near-tie there rather than a large jump: PGD at `eps/4` over 40 steps is
already a strong CE maximizer, and AutoAttack's edge is *reliability* — it cannot be defeated by a
loss-scale quirk or a single attack's blind spot — more than raw ASR on these particular undefended
models, where most survivors really are architectural `L_inf` robustness no in-budget attack can touch.
So the honest claim for the endpoint is not "much higher than PGD" but "no worse than PGD on every
scenario, with the VGG number now trustworthy": a robustness measurement that a decision-preserving
rescaling or a single attack's failure mode can no longer inflate. If instead AutoAttack came in
*below* PGD on VGG, that would mean the fixed-step CE-PGD had been over-reporting ASR by exploiting a
surrogate-specific quirk — and the ensemble's lower, worst-case number would be the more honest one,
which is exactly the property that makes it the right place to stop the ladder.
