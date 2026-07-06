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

Concretely, start at `eta = 2·eps` — in `L_inf` a single signed step of `2·eps` can cross the whole
box before projection, so the projected point lands at an informative boundary location — and keep the
best point `x_max` and best objective seen so far, because the raw iterate is not monotone. Put
checkpoints into the budget at fractions `p_0 = 0`, `p_1 = 0.22`, `p_{j+1} = p_j + max(p_j − p_{j−1} −
0.03, 0.06)`: a long first window to explore, later windows shrinking until they hit a `0.06N` floor,
so progress is checked more often as the search becomes local. Let me actually run that recurrence so
the schedule is concrete rather than a formula. `p_2 = 0.22 + max(0.22 − 0.03, 0.06) = 0.41`,
`p_3 = 0.41 + max(0.16, 0.06) = 0.57`, `p_4 = 0.70`, `p_5 = 0.80`, `p_6 = 0.87`, `p_7 = 0.93`,
`p_8 = 0.99`. So over a 100-iteration budget the checkpoints fall at iterations `0, 22, 41, 57, 70, 80,
87, 93, 99`, and the window widths are `22, 19, 16, 13, 10, 7, 6, 6` — they shrink monotonically and
then pin at the `0.06N = 6`-step floor. That is exactly the shape I argued for: a long `22`-step opening
window where a large step is allowed to roam, then progressively tighter windows so that late in the
budget, when the search is local, a stall is caught within a handful of steps instead of wasting a
quarter of the budget. The `0.75` progress test reads off these widths too — early I need more than
`0.75 · 22 ≈ 16` of the 22 steps to have increased the objective or the step halves; late I need more
than `0.75 · 6 ≈ 5` of 6. It is the schedule I was approximating by hand with "40 steps, reach 10·eps",
now made a function of measured progress rather than a guess. At each checkpoint, halve `eta` if too
few steps in the window actually increased the objective (a count below `0.75` of the window — the step
is overshooting too often) or if the step was not reduced last checkpoint *and* the best objective has
not improved since (the search has stalled at a scale it cannot refine). Whenever I halve, restart the
iterate from `x_max`, because a smaller step should refine the best neighborhood the larger step found.
A momentum blend on the projected step stabilizes the deliberately-large early moves:
`z = P_S(x + eta·dir(grad))`, then `x_{next} = P_S(x + 0.75·(z − x) + 0.25·(x − x_prev))`. Read that
blend: `z − x` is the fresh projected gradient step and `x − x_prev` is the previous move, so the update
is a `0.75/0.25` heavy-ball mix of "where the current gradient wants to go" and "keep going the way I
was going." At `eta = 2·eps` a raw sign step is violent enough to ping-pong across the box; the `0.25`
carry-over damps that oscillation without the systematic staleness that made me *drop* momentum at rung
three — the difference is that here the blend is coupled to an *adaptive* step that halves when progress
stalls, so the momentum cannot keep pushing a wrong direction across a scale change the way a fixed-step
velocity could. And the halve-and-restart-from-`x_max` rule is what makes it safe: whenever the step
shrinks, the iterate is teleported back to the best point seen, so a smaller step always refines the
best neighborhood the larger step discovered rather than continuing from wherever the large step happened
to bounce to. This is an adaptive, step-size-free PGD whose only remaining knob is the iteration budget
— strictly a better version of the inner maximizer I built at rung three, fixing the schedule I had been
hand-tuning.

Second failure mode is subtler and is the one I think actually pins VGG: the *loss*. I have been
climbing cross-entropy throughout, and cross-entropy has a scale degree of freedom the decision rule
does not. `CE(x,y) = −z_y + log Σ_j exp(z_j)`, and its input gradient is `(p_y − 1)∇z_y + Σ_{i≠y}
p_i ∇z_i`. When the true-class probability `p_y` is near one — an overconfident, well-classified point,
which is exactly the VGG survivor — all those coefficients are near zero, and in finite precision can
*be* zero. The gradient vanishes not because the point is hard to move but because the softmax has
saturated, and a decision-preserving rescaling of the logits (`h = α·g`, same argmax for every input)
sharpens the softmax and can drive the CE gradient to zero while the classifier behaves identically.
Let me make the blindness a number. Take logits `z = (4, 1, 0)` on the true class (the `4`); the
softmax probability of the true class is `p_y = e^4/(e^4 + e^1 + e^0) = 54.6/58.3 = 0.936`, so the CE
gradient carries a coefficient `p_y − 1 = −0.064` on `∇z_y` — already small. Now rescale every logit by
`α = 4`, giving `z' = (16, 4, 0)`: the decision is identical (still argmax the first class), but
`p_y = e^16/(e^16 + e^4 + e^0) = 1 − 5.6·10^{-6}`, so the coefficient is now `−5.6·10^{-6}` and every
wrong-class coefficient `p_i` is comparably tiny. The CE gradient has shrunk by four orders of magnitude
under a transformation the classifier does not even notice. A fixed-step optimizer reading that gradient
sees a flat surface and reports the sample robust. So a CE-PGD run can report a VGG sample as robust
when the only thing robust is the *loss surface I chose to climb*, not the model. The fix is to climb a loss with the same invariances as the decision:
shift-invariant and positive-scale-invariant. A margin `−z_y + max_{i≠y} z_i` is shift-invariant
(difference of logits) but not scale-invariant; I need a *ratio* of logit differences so the scale
cancels. Sort the logits `z_{π1} ≥ z_{π2} ≥ z_{π3} ≥ …` and use
`DLR(x,y) = −(z_y − max_{i≠y} z_i) / (z_{π1} − z_{π3})`: the numerator is the decision margin (so the
loss turns positive exactly when a wrong class overtakes `y`), and the `1st−3rd` denominator keeps the
top-two competition the attack is trying to close *out* of the denominator (avoiding a `0/0` at the
boundary) while measuring the logit scale. Being a ratio of differences, it is invariant to the exact
shift-and-scale that blinds cross-entropy — so it keeps a usable gradient precisely on the saturated,
overconfident VGG points where CE goes flat. Check the invariance on the same example. With
`z = (4, 1, 0)`, the true class is `π_1 = 4`, the runner-up `π_2 = 1`, third `π_3 = 0`, and the margin
`z_y − max_{i≠y} z_i = 4 − 1 = 3`, so `DLR = −3/(4 − 0) = −0.75`. Rescale by `α = 4` to `z' = (16, 4,
0)`: margin `16 − 4 = 12`, denominator `16 − 0 = 16`, `DLR = −12/16 = −0.75` — identical, where the CE
gradient collapsed by `10^4`. The `α` cancels top and bottom because both are differences of logits, so
the ratio sees the same landscape at any softmax temperature. And the `π_1 − π_3` denominator, not
`π_1 − π_2`, is deliberate: as the attack succeeds and the runner-up `π_2` climbs to overtake the true
class, `z_{π_1} − z_{π_2}` would collapse toward zero and the ratio would blow up right at the boundary
I care about, whereas `z_{π_1} − z_{π_3}` keeps a stable third-place reference in the denominator and
stays well-conditioned exactly through the flip.

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
masking. The worst-case-over-ensemble aggregation is not a heuristic vote; it is the honest estimator of the
quantity I actually want. Robust accuracy is "the fraction no admissible attack can flip," a min over
attacks of the per-sample survival; with a finite toolbox the best unbiased-downward estimate of that
min is the min over the tools I have, which is exactly "robust only if *none* of them flips it." Adding a
member can only lower the reported robust accuracy (raise ASR) or leave it unchanged — never inflate it
— so a diverse ensemble is monotonically safer against over-reporting, and each member earns its place
by covering a failure mode the others share. That reframes the member choice as failure-mode coverage
rather than raw strength. APGD-CE and targeted APGD-DLR are both gradient attacks but fail on disjoint
sets: CE goes flat on the saturated overconfident points, DLR keeps its gradient there, while CE's
untargeted push on true-class confidence can matter under any output stochasticity where a single
targeted objective is narrow. FAB fails differently again — it linearizes the decision boundary and
minimizes perturbation *norm* to the nearest boundary rather than maximizing a fixed-radius loss, so it
does not care whether the loss surface is saturated, only where the boundary is; a point where both
APGD objectives are flat can still have a nearby boundary FAB walks to. And Square Attack uses *no
gradient at all* — a score-based random search over sign-pattern squares — so it is the backstop against
gradient masking, the failure mode where the model's gradients are deliberately or accidentally
uninformative and *every* first-order member is blind at once; if a defense ever hid its gradients,
Square is the member that would still flip samples and expose the inflated first-order number. Four
members, four distinct ways for an attack to fail, and the worst case over them cannot be defeated by
any single one of those ways.

Run them sequentially on the shrinking set of still-robust points — mark the initially-correct
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
members sweep — the standard ensemble aims each targeted member at the `9` highest-scoring wrong
classes, drawn from the `n_classes − 1` available, so `9` of `9` on CIFAR-10 and `9` of `99` on
CIFAR-100), and the `seed` is read from the environment so the run is reproducible under the harness's
single seed 42. This also sets the cost, which is worth pricing to see why the endpoint is expensive but
affordable. A targeted member runs one `100`-iteration APGD per target class, so `9 × 100 = 900`
gradient steps per sample for targeted DLR and again for targeted FAB, plus `100` for APGD-CE and `5000`
queries for Square — roughly `6000+` model evaluations per sample at the top of the funnel, against
PGD's `40`. That is two-plus orders of magnitude more than a single PGD run. But the sequential funnel
recovers most of it: APGD-CE alone flips the easy ~`0.99` of ResNet/MobileNetV2 samples on the first
`100`-step pass, so only the `~28%` VGG survivors ever reach the expensive targeted and Square stages,
and the per-benchmark cost is dominated by that small residual set rather than the full `1000` samples.
The spend is concentrated exactly where the measurement is contested, which is the right place to spend
it for a *trustworthy* ceiling. Unlike the prior
rungs, `device` is unused here because the package places tensors itself; I bind it to a throwaway. The
derivation above is *why* this call is the right endpoint — it is the worst case over a diverse,
adaptive, parameter-free ensemble — but the code is the invocation, and that is what the answer shows.
There is a real correctness argument for calling the library rather than reimplementing the four
members: the value of AutoAttack as a *reference* ceiling is that everyone runs the same reference
code, so an evaluation number is comparable across runs and cannot be quietly weakened by a subtle
reimplementation bug — a mis-set APGD checkpoint or a DLR denominator that divides by `π_1 − π_2`
instead of `π_1 − π_3` would silently *lower* ASR and over-report robustness, exactly the failure this
whole rung exists to prevent. Reimplementing a strong attack is precisely where I would risk
reintroducing the over-reporting I am trying to eliminate, so on this edit surface the disciplined move
is the standard-version call whose behavior is fixed, with only `n_classes` and `seed` wired through
from the harness.

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

And it is the place to stop for a reason beyond "it is strong." Every rung so far was a *correction to a
named approximation*: FGSM linearized, so MI-FGSM iterated; MI-FGSM carried a transfer-motivated
velocity, so PGD made the maximizer faithful; PGD fixed its step and its loss, so AutoAttack made the
step adaptive and the loss scale-invariant and then hedged the whole thing behind a no-gradient backstop.
Each fixed a specific, identifiable failure I could point to in the previous number. After AutoAttack I
cannot point to another. The step schedule is now adaptive and parameter-free; the loss is invariant to
the exact degrees of freedom the decision ignores; the gradient-masking escape hatch is closed by Square;
and the worst-case-over-ensemble aggregation means adding still more members can only ratify the number,
not overturn it. The remaining VGG survivors are, by elimination, the ones where the model genuinely does
not have a within-budget adversarial neighbor — architectural `L_inf` robustness, not a surrogate
artifact. There is no further correction with a mechanism I can state, and the standard reference the
whole field uses as the trustworthy ceiling is exactly this ensemble. So the ladder stops here not
because I ran out of ideas but because the next rung would have to be a *better-than-worst-case* attack,
which is a contradiction — the worst case over a diverse parameter-free ensemble is, by construction, the
honest ceiling for this white-box edit surface.
