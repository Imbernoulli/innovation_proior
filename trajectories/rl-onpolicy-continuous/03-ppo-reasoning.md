AWR confirmed both halves of my prediction, and the second half is the one that matters. Let me read the
feedback quantitatively, because the per-seed structure tells me exactly what to fix. It won where I
expected it to win: HalfCheetah jumped to a mean of 1996.7 (up from the penalty rung's 1676.6, a $+19\%$
gain) and InvertedDoublePendulum rose to 7299.2 (up from 6877.1, $+6\%$), so abandoning the importance
ratio for a weighted regression did buy real gains on the environments with a clean advantage signal. But
Swimmer collapsed to 90.2, *below* the penalty rung's 101.4, and the per-seed numbers are the tell: 124.4
/ 46.0 / 100.1 across seeds {42, 123, 456}, with seed-123 cratering to 46 — a best-to-worst ratio of
$124.4/46.0\approx2.7$, worse spread than anything the penalty rung produced. That is exactly the fixed-
noisy-target failure I flagged: the sharp $\beta=0.05$ near-binary selection commits to whichever half of
a minibatch scored above its noisy mean, freezes that as the regression target for all ten epochs, and on
Swimmer's high-variance advantages that target is frequently spurious; with no ratio and no KL to pull it
back, one seed regresses confidently toward noise and never recovers. And notice the HalfCheetah "win" is
itself a one-seed story — 1358 / 1331 / 3301 — where two seeds sit near 1350 and the mean is dragged up
to 1996.7 entirely by the lucky 3301 seed; the *median* behaviour barely beat the penalty rung. So AWR is
the *least balanced* update rule: strong-on-average on two environments but built on a single lucky seed,
and worst-of-all on the third. Because the task scores by geometric mean across the three, that Swimmer
90.2 caps AWR hard — the gmean is dominated by the weakest environment, and a method that gave up AWR's
lucky HalfCheetah tail but never produced a Swimmer 46 would beat it overall.

So the next move is constrained from two sides at once. It is *not* "abandon the ratio" — AWR tried that
and it bought instability on the low-signal environment, because a ratio-free regression has no mechanism
to correct a commitment to a noisy action. And it is *not* "soft-penalize the ratio" — the penalty rung
tried that and it bought seed variance on the dense environment, because a soft penalty prices a bad move
instead of forbidding it. Both failures point the same way: keep a ratio surrogate (so the update has a
notion of "I moved too far in this direction and should pull back"), but replace both the soft penalty and
the closed-form regression with a trust region that is *hard and built directly into the loss*, so it is
non-negotiable on every environment at once. Let me go back to what governs whether the surrogate is
honest, because the fix has to attach there.

The surrogate $L(\theta)=\hat{\mathbb E}_t[r_t(\theta)\hat A_t]$ with $r_t=\pi_\theta/\pi_{old}$ is the
Kakade–Langford lower bound on true improvement, and its honesty degrades with the KL distance from
$\pi_{old}$. The penalty rung enforced "stay close" by *subtracting* $\beta\,\mathrm{KL}$ and servoing
$\beta$ — soft, reactive, tradeable. AWR enforced it by *projecting* the closed-form tilted policy onto
the Gaussian — baked in, but blind to noise because nothing corrects an overshoot once the weights
commit. What I want is the trust region of the penalty form without the coefficient that will not hold
still, and without the noise-blindness of the regression. The thing both of those fight is the same: the
optimizer is *rewarded* for pushing $r_t$ far from 1 on a positive-advantage sample, because inflating
the ratio is the cheapest way to raise the objective. So instead of penalizing the distance after the
fact, let me *remove the reward for moving the ratio outside a band at all*.

There are a few ways to build a "hard band into the loss," so let me lay them out and eliminate the ones
that do not survive contact with the frozen Adam loop. One option is a hard projection onto a KL ball
after each step — but that is the constrained second-order solve I already ruled out at rung 1, because
the interface hands me one scalar loss and one first-order optimizer with no hook for a projection step.
A second option is a *barrier*: add a penalty that is zero inside the band $[1-\epsilon,1+\epsilon]$ and
shoots to infinity at the edges. But think about what that does to Adam's gradient. Inside the band the
barrier's gradient is zero, so it shapes nothing until the ratio is right at the edge, where the gradient
explodes; a $320$-step first-order optimizer with a global gradient-norm clip at $0.5$ cannot use an
exploding edge gradient — the clip would swallow it and the step before the edge would still be
unconstrained. A barrier is the right *shape* but the wrong *smoothness* for this optimizer. The third
option is to *clip the objective itself* so that the incentive to leave the band simply vanishes — a flat
spot in the loss rather than a wall. That one is differentiable almost everywhere, has bounded gradient,
and needs nothing the interface does not already give me. So the clip is not one arbitrary choice among
many; it is the one member of the "hard band" family that fits a first-order, gradient-clipped, scalar-
loss optimizer.

Here is the move. Pick a small $\epsilon$ — say $0.2$ — and clip the ratio inside the objective:
$\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\,\hat A_t$. Feel out what that does. With $\hat A_t>0$ (a good
action I want more likely, raising $r_t$), once $r_t$ exceeds $1+\epsilon$ the term flattens at
$(1+\epsilon)\hat A_t$ and its gradient with respect to $r_t$ goes to zero — no more incentive to crank
that action's probability up. With $\hat A_t<0$ (a bad action I want less likely, lowering $r_t$), once
$r_t$ drops below $1-\epsilon$ the term flattens at $(1-\epsilon)\hat A_t$ and again the gradient dies.
So a flat clip kills the incentive to leave the band. That is the trust region I wanted at rung 1 —
expressed as a *flat spot in the loss*, with no KL term, no Lagrange coefficient to servo, and no closed-
form projection. It is unit-free (it constrains the ratio directly), which is why a single $\epsilon=0.2$
can be reliable across HalfCheetah, Swimmer, and InvertedDoublePendulum at once where the penalty rung's
$\beta$ had to chase three different advantage scales — and, from the rung-1 dimensionality arithmetic,
three different per-coordinate KL budgets across $d\in\{1,2,6\}$ — and AWR's $\beta$ over-concentrated on
the noisy one.

But a plain clip is not safe, and the unsafe case is exactly the kind of overshoot that wrecked AWR on
Swimmer, so let me stress it with concrete numbers. Imagine an action that was actually bad, $\hat A_t=
-1$, but the new policy has *already* moved its ratio way up to $r_t=3$ — maybe a noisy earlier minibatch
pushed it there, the same way AWR's sharp weights pushed Swimmer's policy toward spurious actions. With
$\hat A_t<0$ the *unclipped* term $r_t\hat A_t=3\times(-1)=-3$ is very negative, so its gradient wants to
pull $r_t$ back down — it corrects the overshoot. But the *clipped* term $\mathrm{clip}(3,0.8,1.2)\times
(-1)=1.2\times(-1)=-1.2$ sits in the flat region with zero gradient, so a pure clip would *freeze in* the
overshoot it should be undoing. That is precisely AWR's disease — commit to a move and never correct it.
So pure clipping is too forgiving. I want the clip to remove the *incentive to overshoot* but keep the
*gradient that corrects an overshoot*. That means I need both the unclipped and the clipped term and I
should keep the more pessimistic one — the one that gives the objective *less* credit: the minimum.

$$L^{CLIP}(\theta)=\hat{\mathbb E}_t\big[\min\big(r_t\hat A_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\,\hat A_t\big)\big].$$

Let me verify the min does what I want, case by case with numbers, because this is the whole method. Take
$\hat A_t=+1$. Inside the band, say $r_t=1.1$: both terms equal $1.1$, min picks either, gradient alive —
ordinary ascent. Above the band, $r_t=1.5$: unclipped $1.5$, clipped $1.2$; min picks $1.2$ — capped,
gradient zero, "stop paying for going higher." Below the band the *wrong* way, $r_t=0.7$ (the policy made
a good action *less* likely): unclipped $0.7$, clipped $0.8$; min picks the unclipped $0.7$ — gradient
alive, pulling the probability back up. Now $\hat A_t=-1$. Inside, $r_t=1.1$: both $-1.1$, equal. Below
the band the favored way, $r_t=0.7$ (bad action suppressed past the band): unclipped $-0.7$, clipped
$-0.8$; min picks $-0.8$, flat, no gradient — stop paying to suppress further. Above the band, $r_t=3$
(the overshoot from my Swimmer worry): unclipped $-3$, clipped $-1.2$; min keeps the $-3$, gradient alive,
*pulls the overshoot back down*. So across all four corners the min clips away the incentive to push the
ratio further in the direction the advantage already favors, but never clips away the gradient that
corrects a move in the wrong direction. It is a pessimistic lower bound on the unclipped surrogate:
ignore the full ratio change only when including it would make the objective look better, keep it when it
makes the objective look worse. This is the exact property AWR lacked — AWR committed to its weighted
target and had no mechanism to undo a commitment to a noisy action; the min *does*. And to first order at
$\theta_{old}$, where every $r_t=1$, the clip is inactive and $L^{CLIP}$ equals the plain policy-gradient
surrogate, so the first epoch is ordinary ascent and the brake engages only as the policy moves through
the ten epochs.

Why $\epsilon=0.2$ and not $0.05$ or $0.5$? $\epsilon$ is the half-width of the band the policy may move
within per update, in ratio space — a $\pm20\%$ change in any action's probability before the brake fully
engages. Let me tie that to the KL budget the penalty rung was targeting, to check $0.2$ lands in the same
small-move regime rather than a wildly different one. A ratio sitting at the clip edge $1+\epsilon=1.2$
has log-ratio $\ln 1.2\approx0.182$, and from the rung-1 estimator $\mathrm{KL}\approx\tfrac12(\Delta
\log\pi)^2$ that is a per-coordinate KL of $\tfrac12(0.182)^2\approx0.017$; the lower edge $0.8$ gives
$\ln 0.8\approx-0.223$, $\tfrac12(0.223)^2\approx0.025$. So a fully-clipped step corresponds to a realized
KL on the order of $0.017$–$0.025$ — the *same* $\sim0.01$–$0.02$ small-move regime the penalty rung was
*targeting* with its servo, but reached here by a fixed, unit-free constant instead of a feedback loop
that lagged behind every overshoot. Too small an $\epsilon$ and the brake bites immediately, every update
is microscopic, and I waste the ten epochs of reuse — the penalty rung's microscopic-step failure mode in
a different dress. Too large and the band rarely engages across the 320 steps, the ratios fan out, and I
am back to the placeholder blow-up. Around $0.2$ the policy moves a useful amount per iteration while the
realized KL after a full update stays in that regime. That is the structural reason I expect the clip to
fix the HalfCheetah seed variance: the penalty rung hit $0.01$ KL only *on average and reactively*, so
individual minibatches overshot and the seeds diverged; the clip enforces the band on *every* minibatch
*before* the step, so the seeds should track each other far more tightly.

Now the harness. The scaffold hands me already-minibatch-normalized advantages and asks for one
`compute_losses`; the clipped surrogate is naturally three lines — `pg_loss1 = -mb_advantages * ratio`,
`pg_loss2 = -mb_advantages * clamp(ratio, 1-clip_coef, 1+clip_coef)`, `pg_loss = max(pg_loss1,
pg_loss2).mean()` — where the `max` of the two *negatives* is the pessimistic `min` of the two products,
because the loop minimizes the loss. This sign flip is the one place a transcription slip would silently
invert the method, so it is worth pinning: I want the pessimistic (smaller) objective, which after negation
is the *larger* loss, hence `max`. `clip_coef` is the loop's `0.2`. The `get_action_and_value` head stays
the standard Gaussian, identical to both prior rungs; the entire contribution lives in the loss, which is
the point of this edit surface.

There is one more piece the harness exposes that neither prior rung used, and the symmetry argument says
I should: the value clip. The whole reason the policy surrogate is clipped is that across ten epochs the
network drifts from the data-generating one and large moves are destructive. But the value head is on the
*same* drifting network getting the *same* 320 gradient steps, so a single minibatch could yank
$V_\theta(s)$ far from the `mb_values` the rollout recorded — an analogous overshoot, and exactly the kind
of thing that hurt the plain-MSE value loss the two prior rungs used. So I clip the value update too, by
the same pessimism logic: take `max((newvalue - mb_returns)**2, (clip(newvalue, mb_values±clip_coef) -
mb_returns)**2)`. Let me check the pessimism direction with a quick example. Say the recorded value was
`mb_values=5`, the target `mb_returns=6`, and one epoch has already pushed `newvalue=9`. The unclipped
squared error is $(9-6)^2=9$; the clipped prediction is $5+\mathrm{clip}(9-5,-0.2,0.2)=5.2$, squared error
$(5.2-6)^2=0.64$. The `max` keeps the $9$ — so the loss does *not* let the far-moved prediction off the
hook; it keeps rewarding a return toward the target even though the raw move was large, while never
letting the clipped form *reduce* the loss below what the honest error would be. That is the critic-side
analogue of the policy min: cap how far the value prediction is rewarded for moving per update, but never
suppress the gradient that corrects it. The loop gates this on `args.clip_vloss` (default `True`), so I
honor that flag. A more accurate, more stable critic feeds back into better advantages, which feeds back
into a better policy update — so the value clip is not cosmetic; it is the critic-side half of the same
trust region, and it is the one structural ingredient the two prior rungs left entirely on the table.
The full module is in the answer.

The loop's `clipfrac` diagnostic — the fraction with $|r-1|>\epsilon$ — is the cheapest read on whether
the band is doing real work: near zero would mean it never engages (the unclipped placeholder, fan-out
back), near one would mean nearly everything is clipped and the ten epochs waste on flat gradients (the
microscopic-step failure). A modest intermediate fraction, and a similar one across the three
environments, is the signature that a single $\epsilon=0.2$ is transferring.

This reliability pays off specifically in the aggregator. The geometric mean $(x_1 x_2 x_3)^{1/3}$ is
dominated by its smallest factor — halving the weakest environment roughly halves the score, while
doubling the strongest lifts it only by a cube-root — and AWR's ledger is the wrong shape for that: a
lucky-seed HalfCheetah gain bought at a Swimmer collapse to 90.2 (seed 46) sank its weakest factor.
Giving back the lucky tail to lift Swimmer out of collapse and tighten every seed is worth more to the
geometric mean than the peak surrendered.

Within this harness: the 2×64 critic shares the optimizer, so the `vf_coef` term is real; `ent_coef=0` on
MuJoCo (the learned log-std supplies exploration), so the entropy term is included for
contract-completeness but contributes nothing; the GAE scan, per-minibatch normalization, LR anneal, and
gradient-norm clip are all the frozen loop's. My contribution is exactly the clipped surrogate plus the
clipped value loss.

One elegance is worth noting because it is the contrast with the penalty rung's whole apparatus. The
frozen loop anneals the learning rate linearly to zero, so late in training each Adam step moves the
policy less, which means the ratios drift less per epoch and fewer samples reach the clip band — the
`clipfrac` naturally decays and the trust region quietly loosens its grip as the policy converges, with
*zero* control logic on my side. The penalty rung had to actively halve $\beta$ to track that same LR
anneal; the clip gets the identical behaviour for free, because a fixed band in ratio space combined with
a shrinking step size is self-scheduling. There is no coefficient to persist on the `agent`, no servo to
tune, no state to carry across the free-function calls — the entire trust region is three lines that read
only the current minibatch. That statelessness is not just tidy; it is why one $\epsilon$ can hold across
three environments and a full run without ever being adjusted, which is exactly the fragility the penalty
rung's per-minibatch mutable $\beta$ introduced.

Closing on falsifiable expectations against AWR's numbers. My central bet is *balance*: the hard,
per-minibatch, unit-free band should give up a little of AWR's HalfCheetah peak — I would not be shocked
if PPO's HalfCheetah mean lands *below* AWR's 1996.7, since AWR's 3301 seed inflated that mean and PPO
trades the lucky tail for consistency — but should not collapse on Swimmer, which I expect to clear AWR's
90.2 and the penalty rung's 101.4 with much tighter seed spread, because the min-clip corrects the
spurious-action overshoots that cratered AWR's seed-123 to 46. On InvertedDoublePendulum I expect it at
least competitive with 7299.2. The falsifiable claim is about the *geometric mean*, not any single
environment: no environment where PPO is the worst by a wide margin, every seed tracking its neighbours,
the weakest factor lifted out of the collapse that capped both prior rungs — so even without winning any
environment outright, its three-environment geometric mean should be the highest yet.
