The cosine run did two things at once and the split is the whole story. On ResNet-20 / CIFAR-10 it jumped
to 93.03, a clean lift over one-cycle's 92.31; on MobileNetV2 / FashionMNIST it rose to 94.70 from 93.93.
Both confirm the diagnosis from the last rung: dropping the sixty-epoch sub-`base_lr` warmup and annealing
all the way to zero reclaimed the productive high-rate epochs and removed the floor jitter, and the shallow
net and the grayscale mobile net rewarded that immediately. Let me put the deltas on paper because their
*pattern* is what I have to explain. Against one-cycle, cosine gained `+0.72` on CIFAR-10 and `+0.77` on
FashionMNIST — nearly the same lift on the two easy settings — but on ResNet-56 / CIFAR-100 it went the
*other* way, `71.07` against one-cycle's `71.57`, a loss of `0.50`. So two settings up by about three-
quarters of a point and one down by half a point; the mean crept from `85.94` to `86.27`, a `+0.33` that
hides a regression. That single negative delta is not noise I can wave away, because it is exactly the
sign the last rung told me to watch for. When I stripped the warmup I wrote down the falsifiable failure in
advance: if the bare full-rate start re-exposed an init instability on the deep net, CIFAR-100 would *fail
to clear* one-cycle's 71.57 despite the cleaner finish. It did precisely that. The prediction fired, so I
should take its mechanism seriously rather than reach for a different explanation.

Read the three settings together and the pattern is coherent under one cause. The two nets that improved
are the shallow ResNet-20 and the inverted-residual MobileNetV2; the one that regressed is the 56-layer
ResNet. Depth is the axis that separates them. ResNet-56 has `56` layers against ResNet-20's `20` — a
factor of `2.8` more composed nonlinear maps between input and loss — and curvature at initialization
compounds with depth: the loss Hessian's top eigenvalue is built from products of per-layer Jacobians, so
a deeper stack has a sharper, higher-`lambda_max` region right after Kaiming init before BatchNorm's
running statistics have settled. From the last rung's stability arithmetic, `base_lr = 0.1` diverges along
any direction with curvature past `2/0.1 = 20`; the deeper the net, the more likely `lambda_max` at init
sits above that line. So `0.1` from epoch 0 is most likely above the ceiling on ResNet-56 *specifically*,
where the first few full-rate steps overshoot, knock the net into a worse basin before it settles, and the
clean cosine anneal that follows cannot fully recover the damage — the deepest net loses the very accuracy
the cosine's better finish should have bought it. Meanwhile the shallow ResNet-20 and the mobile net have
gentler initial curvature (fewer composed layers, so a lower `lambda_max`), tolerate `0.1` from the start,
and keep the cosine's gains. So the deep net is telling me something the easy nets structurally cannot:
warmup *was* doing real curvature-protection work — just not sixty epochs of it.

I have to be careful here, because the last rung changed *two* things at once relative to one-cycle: it
removed the warmup *and* it dropped the floor from `base_lr/25` to `0`. So a regression on CIFAR-100 could
in principle come from either change, and before I build a whole rung on "restore the warmup" I should rule
out the floor as the culprit. Could annealing all the way to `0` have *hurt* the deep net? That runs
against the mechanism: a lower floor shrinks the stationary jitter monotonically, so going from `0.004` to
`0` can only make the final settling cleaner, and it demonstrably helped the two easy nets (both rose),
which is the direction a better floor predicts. There is no mechanism by which reaching a smaller final
rate damages a deep net specifically; the floor change is a strict improvement at the *finish*. The
regression, by contrast, is concentrated where the *start* is hardest — the deepest, highest-`lambda_max`
net — which is exactly the signature of an init overshoot, not a finish problem. And the two easy nets,
which got the same floor change, did not regress. So the floor is exonerated and the warmup removal is
implicated: the change that helped everywhere (floor to zero) cannot be what hurt one setting, and the
change that only mattered where curvature is sharpest (warmup removed) lines up with the one net that
suffered. One loose end is MobileNetV2, which is not shallow — it has plenty of depth — yet it improved
like the easy net rather than regressing like ResNet-56. I read that as its setting: inverted-residual
blocks with linear bottlenecks on grayscale FashionMNIST are an easier optimization than a 56-layer
residual stack on 100-way CIFAR, so its init curvature stays under the ceiling even without warmup. Depth
alone is not the trigger; depth *times* task difficulty is, and ResNet-56 / CIFAR-100 is the only setting
that maxes both.

That is the precise lesson, and it points at a precise fix. The mistake in rung one was not *having* a
warmup; it was making the warmup sixty epochs long and forcing the rate below `base_lr` for nearly a third
of the run, which cost the easy nets about half their early step-distance. The mistake in rung two was
throwing the warmup out entirely and re-exposing the deep net's init. The two failures bracket the answer:
one had too much warmup, the other too little, and the right move is the middle — a *short* warmup that
protects the sharp initial region for just a handful of epochs, then hands off to the same cosine anneal
that already won on the easy nets. I want the best of both runs — cosine's clean full-rate-to-zero body,
which won CIFAR-10 and FashionMNIST, plus a brief opening ramp that survives the deep net's high-curvature
init, which cosine lacked. So I split the problem cleanly: a start-of-training problem (survive the sharp
init) and a body problem (anneal smoothly to settle), handle them separately, and stitch.

Take the start first, because it is the one cosine got wrong on CIFAR-100. The disease is that `base_lr`
is above the curvature ceiling `2/lambda_max` while the deep net is in its sharp initial region. The cure
is to not use `base_lr` at the very start — use something smaller while curvature is high, then raise it.
Why does small-then-raise actually buy me anything rather than just delaying the problem to whenever I do
raise the rate? Because the sharp initial region is *transient*: even a few small, safe steps move the
weights into a flatter region where `lambda_max` has come down and `2/lambda_max` has risen, so the full
`base_lr` becomes admissible where it was not at epoch 0. The point of starting small is to survive long
enough for the curvature to relax under its own descent, and only then step up to the rate I actually want
for the body. So at the very start the curve should *rise*, from a safe small value to `base_lr`, over a
brief opening phase — and this is the same rise-then-run shape one-cycle had, just compressed from sixty
epochs to a handful and with no aggressive peak on the far side.

How to climb? The laziest version is a constant low prefix — hold `base_lr/5 = 0.02` for a few epochs,
then snap to `base_lr`. But the snap recreates the exact disease I am trying to cure. After the prefix the
net has descended somewhere, but I have no guarantee its curvature has come down *enough* that `base_lr` is
now below `2/lambda_max` there; if it has not, the snap from `0.02` to `0.1` is a five-fold jump straight
back over the ceiling, the same loss spike as before, merely postponed by a few epochs. The discontinuity
is the problem. The fix is forced: do not snap, climb *gradually* — increase the rate by a constant
increment each epoch so there is never a value jump. As `eta` rises the weights keep moving into flatter
regions and `2/lambda_max` keeps rising; a linear ramp keeps `eta` below that rising ceiling instead of
leaping over it, tracking the relaxation rather than betting on it having finished. So the opening phase is
a linear ramp: over the first `warmup` epochs, `eta` takes the sampled values `base_lr/warmup`,
`2*base_lr/warmup`, ..., `base_lr` — the k-th warmup epoch is `k*base_lr/warmup`, an honest arithmetic
ramp reaching `base_lr` at the last warmup epoch. I write it in the code as `base_lr*(epoch+1)/warmup` so
that the 0-indexed first epoch already does a little work at `base_lr/warmup` rather than sitting at zero.

Linear, not cosine — and I should say why, since one-cycle's up-leg was a cosine climb and I am
deliberately not reusing it here. Over a five-epoch window the shape of the ramp barely exists: I have only
five samples between `0.02` and `0.10`, and a cosine warmup over the same window would put them at about
`0.028, 0.048, 0.072, 0.092, 0.100` against the linear `0.020, 0.040, 0.060, 0.080, 0.100` — a maximum
difference of `0.012`, one epoch's worth of curvature that no run is going to resolve. A cosine ramp would
buy me a smoother slope at the very bottom of the climb in exchange for a shape parameter and a second
formula, for a difference smaller than the ramp's own step size. The linear ramp is the standard warmup,
carries no free knob, and over five epochs is indistinguishable from any smooth alternative, so I take it
and spend my attention on the length instead of the curve.

How long is `warmup`? This is where the two failed rungs pin the answer between them. One-cycle's sixty
epochs were far too long — 30% of the run, half the early step-distance thrown away, and the easy nets paid
for it. Cosine's zero was too short for the deep net. I want the *smallest* warmup that still relaxes the
deep net's initial curvature: long enough to matter, short enough to be a negligible prefix against the
200-epoch budget. Five epochs — `2.5%` of the run, one twelfth of one-cycle's warmup — is the natural
choice: it gives the sharp region a few steps to relax while costing the body almost nothing. Let me put
the "almost nothing" on paper so I know what I am trading. The five ramp epochs accumulate `0.02 + 0.04 +
0.06 + 0.08 + 0.10 = 0.30` in step-distance, against `0.50` if those same five epochs had run at the full
`0.1`; so the ramp costs `0.20` of travel, which against the run's total of about `10.05` is under `2%`.
That is the whole tax the easy nets pay — two percent of their total stepping, at the very start where a
couple of extra epochs at half-rate is invisible against a shallow net's easy descent. Contrast one-cycle's
warmup, which cost the easy nets roughly *half* their early step-distance; five epochs is a twenty-fifth of
that damage. And it directly tests the diagnosis: if five epochs of ramp recover CIFAR-100 to above one-
cycle's 71.57 *without* costing the easy nets the gains cosine won, then the deep net's loss really was an
init-overshoot that a short warmup fixes, and the lost accuracy was never about the cosine shape. It is
also worth seeing that this rung is literally a one-parameter bridge between the last two: `warmup = 0`
recovers rung two's pure cosine, and a `warmup` near `60` recovers a one-cycle-length opening; I am placing
the knob near the cosine end of that line, at `5`, close enough to keep the body cosine won and far enough
from zero to shield the deep net. Five it is.

Now the body — epochs 5 through 199 — where I want to start at the full `base_lr` (continuous with the top
of the ramp) and anneal to zero. This is exactly the cosine I already derived and already validated on the
easy nets: a half-cosine whose slope is zero at both ends and steepest in the middle, so the rate lingers
high while the gradient is informative, falls through the middle, and lingers low to settle below the noise
floor. I keep it unchanged; the only real question is the *clock*. The ramp occupies epochs 0–4 and ends
at exactly `base_lr` at epoch 4, so the cosine body must *begin* at `base_lr` at the seam — which means
the cosine's progress must be measured from the *end of the warmup*, spanning the remaining
`total_epochs - warmup = 195` epochs, not from epoch 0. Let me check both choices numerically to see that
this matters and by how much. Measuring from the end of warmup, `progress = (epoch - warmup)/(total -
warmup)`: at `epoch = 5` progress is `0`, `cos(0) = 1`, `eta = base_lr` — continuous with the top of the
ramp. Measuring instead from epoch 0 over the full 200, at `epoch = 5` the cosine would already read
`base_lr*0.5*(1 + cos(pi*5/200)) = 0.09985`, so the handoff would drop by about `1.5e-4` — a small value
jump, and more importantly the cosine would have "used up" five epochs of its descent range on the warmup
window, shortening the anneal. The value jump alone is negligible, but there is no reason to accept it when
the warmup-relative clock removes it exactly and hands the body its full 195-epoch budget. So
`progress = (epoch - warmup)/(total_epochs - warmup)`, and the body is `base_lr*0.5*(1 + cos(pi*progress))`.

Let me trace the seam itself, because stitching two formulas is where an off-by-one hides. Epoch 4 is the
last warmup epoch: `base_lr*(4+1)/5 = base_lr*5/5 = 0.10`. Epoch 5 is the first body epoch: progress `0`,
`base_lr*0.5*(1+cos 0) = 0.10`. So *both* epoch 4 and epoch 5 sit at exactly `base_lr` — the schedule holds
the peak for two epochs across the seam, then epoch 6 begins the descent at `base_lr*0.5*(1+cos(pi/195)) =
0.099994`, barely below. That two-epoch plateau at the top is harmless and even faintly desirable: it is a
value-continuous handoff with a slope change (from the ramp's constant positive slope to the cosine's flat
tangent) but no value discontinuity, so there is nothing to shock the dynamics or to mismatch against the
`10*`-amplified momentum velocity. And I should check the two-epoch plateau does not quietly re-trigger the
very overshoot I built the ramp to prevent: by epoch 4 the net has taken four gradually-rising steps
(`0.02, 0.04, 0.06, 0.08`) that, by the transient-curvature argument, have already moved it into a flatter
region where `2/lambda_max` has risen above `base_lr`; so arriving at `0.10` for epochs 4 and 5 lands on a
ceiling that has *already* risen to admit it, not on the sharp init ceiling that `0.1` violated at epoch 0.
The plateau is safe precisely because the ramp preceded it — the whole point of the ramp was to make
`base_lr` admissible by the time the schedule reaches it, and holding it for one extra epoch changes
nothing once it is admissible. The ramp's own samples are `0.02, 0.04, 0.06, 0.08, 0.10` — five equal
steps of `+0.02`, which are the largest epoch-to-epoch changes anywhere in the schedule (the cosine body's
are on the order of `base_lr/100`), but they are *upward* and gradual by construction, the exact opposite
of step decay's downward cliffs, and they are precisely where I want the rate to move deliberately.

Stitch the two phases: for `epoch < warmup`, the linear ramp `base_lr*(epoch+1)/warmup`; for
`epoch >= warmup`, the cosine body with the warmup-relative clock. One branch on the epoch index, two short
formulas, and I confirmed the last body epoch: at `epoch = 199`, progress `= 194/195 = 0.99487`, so
`eta = base_lr*0.5*(1+cos(0.99487*pi)) ≈ 6.6e-6`, just above zero, reaching exactly zero at the boundary
after training — the same settling convention the plain cosine had. It conditions on nothing in `config`:
the same curve serves all three settings, because the two arguments — survive the sharp init, then anneal
smoothly to settle — are architecture-agnostic. The shallow nets that already tolerated `base_lr` from
epoch 0 lose almost nothing to a five-epoch ramp (the ramp is brief and the body is identical to the
cosine they liked); the deep net that overshot gains the protection it needed. That asymmetry — nearly
costless on the easy nets, decisive on the hard one — is the whole point of putting a *short* warmup back
rather than a long one, and it is why I expect this to strictly dominate cosine rather than trade one
setting for another. One number makes the "costless" side concrete: summing the whole schedule's per-epoch
rate gives `10.10` in total travel against the pure cosine's `10.05` — a difference of `0.05` out of ten,
half a percent. Warmup+cosine is, in total step-distance, the same run as cosine, with the `0.20` ramp tax
at the front almost exactly offset by handing the body its full 195-epoch descent from the warmup-relative
clock. So on the nets that did not need the init protection, the two schedules are near-identical objects,
which is precisely why I expect CIFAR-10 and FashionMNIST to land within a hair of their cosine values —
they are running what is essentially the same curve — while the deep net, which alone cared about the first
five epochs, gets its overshoot fixed.

So the delta from rung two is one targeted addition: keep cosine's body verbatim — the part that won
CIFAR-10 (93.03) and FashionMNIST (94.70) — and prepend a five-epoch linear ramp from `base_lr/5` to
`base_lr` to protect the deep net's init, with the cosine clock shifted to start at the end of warmup. The
falsifiable expectations are pinned tightly by the previous two rungs, and I want them sharp enough to be
wrong. The decisive claim is on CIFAR-100: if the short warmup is the right fix, ResNet-56 / CIFAR-100
should clear *both* one-cycle's 71.57 and cosine's 71.07 — the warmup recovers what cosine's bare start
lost — and it should do so by the biggest margin of any setting, because CIFAR-100 is where the init
instability lived. Quantitatively I expect it back into the 72s, above both prior rungs by more than the
half-to-one-point deltas the ladder has been trading in, since it is recovering a genuine overshoot rather
than shaving a floor. On the easy nets the prediction is the *opposite* kind: I expect CIFAR-10 and
FashionMNIST to stay essentially where cosine put them (mid-93 and mid-94), moving by only a hair in either
direction — a five-epoch ramp on nets that already tolerated the full rate is nearly a no-op. This is a
real risk, not a hedge: if either easy net *drops* noticeably, then even five epochs of sub-`base_lr`
opening is costing the easy nets early step-distance and five is too long; a clean win requires CIFAR-100
up *and* the easy nets flat. The whole rung succeeds only if it buys CIFAR-100 at the deep net while
leaving CIFAR-10 and FashionMNIST untouched — strictly dominating cosine across the board, which is what
would make it the strongest schedule on the ladder so far.

The causal chain in one breath: cosine's split result — CIFAR-10 `+0.72` to 93.03 and FashionMNIST `+0.77`
to 94.70 (dropping the long warmup and the non-zero floor helped the easy nets) but CIFAR-100 `-0.50` to
71.07, below one-cycle's 71.57 (the bare full-rate start re-exposed the deep net's high-curvature init) —
fired exactly the failure I predicted and says warmup was doing real curvature-protection work, just not
sixty epochs of it → the regression tracks depth (`56` vs `20` layers, a sharper `lambda_max` at init on
the deep net, so `0.1` is above `2/lambda_max` there specifically) → split start from body: a *short*
warmup to survive the sharp initial region while `2/lambda_max` relaxes, then cosine to settle → the warmup
must be a *gradual* linear ramp, not a constant prefix that snaps (the `0.02 -> 0.1` snap re-creates the
overshoot), and five epochs (2.5%, one twelfth of one-cycle's) is the smallest ramp that relaxes the deep
net's init without eating the body → keep cosine's half-cosine body verbatim but measure its clock from the
end of warmup so the seam is value-continuous at `base_lr` (traced: epochs 4 and 5 both `0.10`, then a
gentle descent to `~0` at epoch 199) → expect CIFAR-100 to clear both prior rungs by the largest margin
(the init fix) while CIFAR-10 and FashionMNIST hold near cosine's values (a brief ramp is a near-no-op on
nets that already tolerated the full rate), which would make it strictly dominate cosine and top the ladder.
