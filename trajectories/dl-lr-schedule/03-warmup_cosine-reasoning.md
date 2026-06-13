The cosine run did two things at once and the split is the whole story. On ResNet-20 / CIFAR-10 it jumped
to 93.03, a clean lift over one-cycle's 92.31; on MobileNetV2 / FashionMNIST it rose to 94.70 from 93.93.
Both confirm the diagnosis from the last rung: dropping the sixty-epoch sub-`base_lr` warmup and annealing
all the way to zero reclaimed the productive high-rate epochs and removed the floor jitter, and the shallow
net and the grayscale mobile net rewarded that immediately. But ResNet-56 / CIFAR-100 went the *other*
way — 71.07, *below* one-cycle's 71.57, the one setting where cosine lost ground. That is exactly the open
risk I named when I stripped the warmup: that on the deep net, starting at the full `base_lr` from epoch 0
would re-expose an init instability the warmup had been quietly protecting against. The numbers say it did.
The deep 56-layer net on 100 classes is the most curvature-sensitive of the three at initialization — more
layers means a sharper, higher-`lambda_max` region right after Kaiming init, so the admissible step
`2/lambda_max(H)` is smallest there, and `base_lr = 0.1` from epoch 0 is most likely to be above that
ceiling on *this* net specifically. The first few full-rate steps overshoot, the net gets knocked into a
worse basin before it settles, and the clean cosine anneal that follows cannot fully recover the damage —
so the deepest net loses the very accuracy the cosine's better finish should have bought it. Meanwhile the
shallow ResNet-20 and the mobile net have gentler initial curvature, tolerate 0.1 from the start, and keep
the cosine's gains. So the deep net is telling me something the easy nets cannot: warmup *was* doing real
curvature-protection work — just not sixty epochs of it.

That is the precise lesson, and it points at a precise fix. The mistake in rung one was not *having* a
warmup; it was making the warmup *sixty epochs long* and forcing the rate *below* `base_lr` for nearly a
third of the run. The mistake in rung two was throwing the warmup out entirely and re-exposing the deep
net's init. The right move is the middle: a *short* warmup that protects the sharp initial region for just
a handful of epochs, then hands off to the same cosine anneal that already won on the easy nets. I want the
best of both runs — cosine's clean full-rate-to-zero body, which won CIFAR-10 and FashionMNIST, plus a
brief opening ramp that survives the deep net's high-curvature init, which cosine lacked. So I split the
problem cleanly: a start-of-training problem (survive the sharp init) and a body problem (anneal smoothly
to settle), handle them separately, and stitch.

Take the start first, because it is the one cosine got wrong on CIFAR-100. The disease is that `base_lr`
is above the curvature ceiling `2/lambda_max` while the deep net is in its sharp initial region. The cure
is to not use `base_lr` at the very start — use something smaller while curvature is high, then raise it.
Why does small-then-raise actually buy me anything rather than just delaying the problem? Because the sharp
initial region is *transient*: even a few small, safe steps move the weights into a flatter region where
`lambda_max` has come down and `2/lambda_max` has risen, so the full `base_lr` becomes admissible. The
point of starting small is to survive long enough for the curvature to relax, then ramp up to the rate I
actually want for the body. So at the very start the curve should *rise*, from a safe small value to
`base_lr`, over a brief opening phase.

How to climb? The laziest version is a constant low prefix — hold `base_lr/5` for a few epochs, then snap
to `base_lr`. But the snap recreates the exact disease I am trying to cure: after the prefix the net has
descended somewhere, but I have no guarantee its curvature has come down *enough* that `base_lr` is now
below `2/lambda_max` there, and if it has not, the snap to `base_lr` is the same loss spike, just postponed
by a few epochs. The snap is the problem. The fix is forced: do not snap, climb *gradually* — increase the
rate by a constant increment each epoch so there is never a value discontinuity. As `eta` rises the weights
keep moving into flatter regions and `2/lambda_max` keeps rising; a linear ramp keeps `eta` below that
rising ceiling instead of leaping over it. So the opening phase is a linear ramp: over the first `warmup`
epochs, `eta` takes the sampled values `base_lr/warmup, 2*base_lr/warmup, ..., base_lr` — the kth warmup
epoch is `k*base_lr/warmup`, an honest arithmetic ramp reaching `base_lr` at the last warmup epoch. I start
the sampled ramp at `base_lr/warmup` rather than literally 0 so the first epoch already does a little work.

How long is `warmup`? This is where the two failed rungs pin the answer between them. One-cycle's sixty
epochs were far too long — they ate the productive middle and the easy nets paid for it. Cosine's zero was
too short for the deep net. I want the *smallest* warmup that still relaxes the deep net's initial
curvature: long enough to matter, short enough to be a negligible prefix against the 200-epoch budget. Five
epochs — 2.5% of the run — is the natural choice: it gives the sharp region a few steps to relax while
costing the body almost nothing, the opposite of one-cycle's 30%. Five it is. Note this directly tests the
diagnosis: if five epochs of ramp recover CIFAR-100 to above one-cycle's 71.57 *without* costing the easy
nets the gains cosine won, then the deep net's loss really was an init-overshoot that a short warmup fixes,
and the lost accuracy was never about the cosine shape.

Now the body — epochs 5 through 199 — where I want to start at the full `base_lr` (continuous with the top
of the ramp) and anneal to zero. This is exactly the cosine I already derived and already validated on the
easy nets: a half-cosine whose slope is zero at both ends and steepest in the middle, so the rate lingers
high while the gradient is informative, falls through the middle, and lingers low to settle below the
noise floor. I keep it unchanged; the only question is the *clock*. The ramp occupies epochs 0–4 and ends
at exactly `base_lr` at epoch 4, so the cosine body must *begin* at `base_lr` at the seam — which means the
cosine's progress must be measured from the *end of the warmup*, spanning the remaining
`total_epochs - warmup` epochs, not from epoch 0. If I measured progress from epoch 0 over the full 200,
the cosine at the seam (epoch 5) would already be slightly below `base_lr` and I would have a value drop at
the handoff, and worse, the warmup epochs would eat into the cosine's budget. Measuring from the end of
warmup makes the seam continuous: at `epoch = warmup`, progress is 0, `cos(0) = 1`, `eta = base_lr` —
continuous with the top of the ramp; at the last in-loop epoch progress is just under 1 so `eta` is just
above 0, and the cosine reaches 0 exactly at the boundary after training, the usual convention. So
`progress = (epoch - warmup) / (total_epochs - warmup)`, and the body is `base_lr * 0.5 * (1 + cos(pi *
progress))`.

Stitch the two phases: for `epoch < warmup`, the linear ramp `base_lr * (epoch + 1) / warmup`; for
`epoch >= warmup`, the cosine body with the warmup-relative clock. One branch on the epoch index, two short
formulas. The seam has no learning-rate jump (both equal `base_lr` there), though it does have a slope
change — from the ramp's constant positive slope into the cosine's flat tangent — which is harmless: there
is no value discontinuity to shock the dynamics or to mismatch the momentum velocity. And it conditions on
nothing in `config`: the same curve serves all three settings, because the two arguments — survive the
sharp init, then anneal smoothly to settle — are architecture-agnostic. The shallow nets that already
tolerated `base_lr` from epoch 0 lose almost nothing to a five-epoch ramp (the ramp is brief and the body
is identical to the cosine they liked); the deep net that overshot gains the protection it needed. That
asymmetry — costless on the easy nets, decisive on the hard one — is the whole point of putting a *short*
warmup back. (The full schedule body is in the answer.)

I should also be careful about the per-epoch granularity, since the schedule is sampled once per epoch and
held constant within it. The ramp's epoch values are `base_lr/5, 2*base_lr/5, ..., base_lr` — equal steps
of `base_lr/5 = 0.02`, the largest epoch-to-epoch jumps anywhere in the schedule, but they are *upward* and
gradual by construction, the opposite of step decay's downward cliffs. The cosine body is sampled at 195
points with per-epoch changes on the order of `base_lr/100`, far smaller. So the only non-trivial
transition is the gentle five-step climb, which is exactly where I want the rate to move deliberately.

So the delta from rung two is one targeted addition: keep cosine's body verbatim — the part that won
CIFAR-10 (93.03) and FashionMNIST (94.70) — and prepend a five-epoch linear ramp from `base_lr/5` to
`base_lr` to protect the deep net's init, with the cosine clock shifted to start at the end of warmup. The
falsifiable expectations are pinned tightly by the previous two rungs. The decisive claim is on CIFAR-100:
if the short warmup is the right fix, ResNet-56 / CIFAR-100 should clear *both* one-cycle's 71.57 and
cosine's 71.07 — the warmup recovers what cosine's bare start lost, by the biggest margin of any setting,
because CIFAR-100 is where the init instability lived. On the easy nets the prediction is the *opposite*
kind: I expect CIFAR-10 and FashionMNIST to stay essentially where cosine put them (mid-93 and mid-94),
moving by only a hair in either direction — a five-epoch ramp on nets that already tolerated the full rate
is nearly a no-op, so if either of those *drops* noticeably, the warmup is costing the easy nets something
and five epochs is too long. The whole rung succeeds if it buys CIFAR-100 at the deep net while leaving
CIFAR-10 and FashionMNIST untouched — strictly dominating cosine across the board, which is what would make
it the strongest schedule on the ladder.

The causal chain in one breath: cosine's split result — CIFAR-10 up to 93.03 and FashionMNIST up to 94.70
(dropping the long warmup and the non-zero floor helped the easy nets) but CIFAR-100 *down* to 71.07 below
one-cycle's 71.57 (the bare full-rate start re-exposed the deep net's high-curvature init) — says warmup
was doing real curvature-protection work, just not sixty epochs of it → split start from body: a *short*
warmup to survive the sharp initial region while `2/lambda_max` relaxes, then cosine to settle → the warmup
must be a *gradual* linear ramp, not a constant prefix that snaps (the snap re-creates the overshoot), and
five epochs (2.5%) is the smallest ramp that relaxes the deep net's init without eating the body → keep
cosine's half-cosine body verbatim but measure its clock from the end of warmup so the seam is continuous
at `base_lr` → expect CIFAR-100 to clear both prior rungs by the largest margin (the init fix) while
CIFAR-10 and FashionMNIST hold near cosine's values (a brief ramp is a near-no-op on nets that already
tolerated the full rate), which would make it strictly dominate cosine and top the ladder.
