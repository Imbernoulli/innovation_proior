AWR confirmed both halves of my prediction, and the second half is the one that matters. It won where I
expected it to win: HalfCheetah jumped to a mean of 1996.7 (up from the penalty rung's 1676.6) and
InvertedDoublePendulum rose to 7299.2 (up from 6877.1), so abandoning the importance ratio for a
weighted regression did buy real gains on the environments with a clean advantage signal. But Swimmer
collapsed to 90.2, *below* the penalty rung's 101.4, and the per-seed numbers are the tell: 124.4 /
46.0 / 100.1, with seed 123 cratering to 46 — almost half the other seeds. That is exactly the failure I
flagged. The sharp $\beta=0.05$ temperature concentrates the regression on whichever actions got the
top advantages on a minibatch, and on Swimmer's long-horizon, noisy advantages those top advantages are
frequently *noise*; with no ratio and no KL to pull it back, the policy confidently regresses toward
spurious actions on some seeds and never recovers. And notice HalfCheetah's win is itself a one-seed
story — 1358 / 1331 / 3301 — so even the gain is fragile. So AWR is the *least balanced* update rule:
strong on two environments, worst-of-all on the third. Because the task scores by geometric mean across
the three, that Swimmer 90.2 caps AWR hard — a method that gave up a little peak on HalfCheetah but
never collapsed on Swimmer would beat it overall. The next move is therefore not "abandon the ratio"
(AWR tried that and it bought instability on the low-signal environment) and not "soft-penalize the
ratio" (the penalty rung tried that and it bought seed variance on the dense one). It is: *keep* a ratio
surrogate, but replace both the soft penalty and the closed-form regression with a trust region that is
*hard and built directly into the loss*, so it is non-negotiable on every environment at once.

Let me go back to what governs whether the surrogate is honest, because the fix has to attach there. The
surrogate $L(\theta)=\hat{\mathbb E}_t[r_t(\theta)\hat A_t]$ with $r_t=\pi_\theta/\pi_{old}$ is the
Kakade–Langford lower bound on true improvement, and its honesty degrades with the KL distance from
$\pi_{old}$. The penalty rung enforced "stay close" by *subtracting* $\beta\,\mathrm{KL}$ and servoing
$\beta$ — soft, reactive, tradeable. AWR enforced it by *projecting* the closed-form tilted policy onto
the Gaussian — baked in, but blind to noise because nothing corrects an overshoot once the weights
commit. What I want is the trust region of the penalty form without the coefficient that will not hold
still, and without the noise-blindness of the regression. The thing both of those fight is the same: the
optimizer is *rewarded* for pushing $r_t$ far from 1 on a positive-advantage sample, because inflating
the ratio is the cheapest way to raise the objective. So instead of penalizing the distance after the
fact, let me *remove the reward for moving the ratio outside a band at all*.

Here is the move. Pick a small $\epsilon$ — say $0.2$ — and clip the ratio inside the objective:
$\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\,\hat A_t$. Feel out what that does. With $\hat A_t>0$ (a good
action I want more likely, raising $r_t$), once $r_t$ exceeds $1+\epsilon$ the term flattens at
$(1+\epsilon)\hat A_t$ and its gradient with respect to $r_t$ goes to zero — no more incentive to crank
that action's probability up. With $\hat A_t<0$ (a bad action I want less likely, lowering $r_t$), once
$r_t$ drops below $1-\epsilon$ the term flattens at $(1-\epsilon)\hat A_t$ and again the gradient dies.
So a flat clip kills the incentive to leave the band. That is the trust region I wanted at rung 1 —
expressed as a *flat spot in the loss*, with no KL term, no Lagrange coefficient to servo, and no
closed-form projection. It is unit-free (it constrains the ratio directly), which is why a single
$\epsilon=0.2$ can be reliable across HalfCheetah, Swimmer, and InvertedDoublePendulum at once where the
penalty rung's $\beta$ had to chase three different advantage scales and AWR's $\beta$ over-concentrated
on the noisy one.

But a plain clip is not safe, and the unsafe case is exactly the kind of overshoot that wrecked AWR on
Swimmer, so let me stress it. Imagine an action that was actually bad, $\hat A_t<0$, but the new policy
has *already* moved its ratio way up to $r_t=3$ — maybe a noisy earlier minibatch pushed it there, the
same way AWR's sharp weights pushed Swimmer's policy toward spurious actions. With $\hat A_t<0$ the
*unclipped* term $r_t\hat A_t=3\hat A_t$ is very negative, so its gradient wants to pull $r_t$ back down
— it corrects the overshoot. But the *clipped* term $\mathrm{clip}(3,\dots)\hat A_t=(1+\epsilon)\hat A_t$
sits in the flat region with zero gradient, so a pure clip would *freeze in* the overshoot it should be
undoing. That is precisely AWR's disease — commit to a move and never correct it. So pure clipping is too
forgiving. I want the clip to remove the *incentive to overshoot* but keep the *gradient that corrects
an overshoot*. That means I need both the unclipped and the clipped term and I should keep the more
pessimistic one — the one that gives the objective *less* credit: the minimum.

$$L^{CLIP}(\theta)=\hat{\mathbb E}_t\big[\min\big(r_t\hat A_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\,\hat A_t\big)\big].$$

Let me verify the min does what I want, case by case, because this is the whole method. Take $\hat A_t>0$.
Inside the band the two terms are equal. Above $1+\epsilon$ the clipped term $(1+\epsilon)\hat A_t$ is
smaller, so `min` picks it — capped, gradient zero, "stop paying for going higher." Below $1-\epsilon$
(the policy made a good action *less* likely — wrong direction) the clipped term $(1-\epsilon)\hat A_t$
is *larger*, so `min` picks the unclipped $r_t\hat A_t$ — gradient alive, pulling the probability back
up. Now $\hat A_t<0$. Inside the band, equal. Below $1-\epsilon$ (bad action suppressed past the band —
the favored direction) the clipped $(1-\epsilon)\hat A_t$ is more negative, `min` picks it, flat, no
gradient — stop paying to suppress further. Above $1+\epsilon$ (bad action made *more* likely — the
overshoot from my Swimmer worry) the unclipped $r_t\hat A_t$ is more negative, `min` keeps it, gradient
alive, *pulls the overshoot back down*. So the min clips away the incentive to push the ratio further in
the direction the advantage already favors, but never clips away the gradient that corrects a move in the
wrong direction. It is a pessimistic lower bound on the unclipped surrogate: ignore the full ratio change
only when including it would make the objective look better, keep it when it makes the objective look
worse. This is the exact property AWR lacked — AWR committed to its weighted target and had no mechanism
to undo a commitment to a noisy action; the min *does*. And to first order at $\theta_{old}$, where every
$r_t=1$, the clip is inactive and $L^{CLIP}$ equals the plain policy-gradient surrogate, so the first
epoch is ordinary ascent and the brake engages only as the policy moves through the K epochs.

Why $\epsilon=0.2$ and not $0.05$ or $0.5$? $\epsilon$ is the half-width of the band the policy may move
within per update, in ratio space — a $\pm20\%$ change in any action's probability before the brake fully
engages. Too small and the brake bites immediately, every update is microscopic, and I waste the ten
epochs of reuse the loop is built for — the penalty rung's microscopic-step failure mode in a different
dress. Too large and the band rarely engages across the K epochs, the ratios fan out, and I am back to
the placeholder blow-up. Around $0.2$ the policy moves a useful amount per iteration while the realized
KL after a full update stays $\sim0.01$–$0.02$ — the same small-move regime the penalty rung was
*targeting* with its servo, but reached here by a fixed, unit-free constant instead of a feedback loop
that lagged behind every overshoot. That is the structural reason I expect the clip to fix the
HalfCheetah seed variance: the penalty rung hit $0.01$ KL only *on average and reactively*, so individual
minibatches overshot and the seeds diverged; the clip enforces the band on *every* minibatch *before* the
step, so the seeds should track each other far more tightly.

Now the harness. The scaffold hands me already-minibatch-normalized advantages and asks for one
`compute_losses`; the clipped surrogate is naturally three lines — `pg_loss1 = -mb_advantages * ratio`,
`pg_loss2 = -mb_advantages * clamp(ratio, 1-clip_coef, 1+clip_coef)`, `pg_loss =
max(pg_loss1, pg_loss2).mean()` — where the `max` of the two *negatives* is the pessimistic `min` of the
two products, because the loop minimizes the loss. `clip_coef` is the loop's `0.2`. The `get_action_and_value`
head stays the standard Gaussian, identical to both prior rungs; the entire contribution lives in the loss,
which is the point of this edit surface.

There is one more piece the harness exposes that neither prior rung used, and the symmetry argument says
I should: the value clip. The whole reason the policy surrogate is clipped is that across ten epochs the
network drifts from the data-generating one and large moves are destructive. But the value head is on the
*same* drifting network getting the *same* ten epochs, so a single minibatch could yank $V_\theta(s)$ far
from the `mb_values` the rollout recorded — an analogous overshoot, and exactly the kind of thing that
hurt the plain-MSE value loss the two prior rungs used. So I clip the value update too, by the same
pessimism logic: take `max((newvalue - mb_returns)**2, (clip(newvalue, mb_values±clip_coef) -
mb_returns)**2)`, never letting the clipped form *reduce* the loss, only capping how far the value
prediction is rewarded for moving per update. The loop gates this on `args.clip_vloss` (default `True`),
so I honor that flag. A more accurate, more stable critic feeds back into better advantages, which feeds
back into a better policy update — so the value clip is not cosmetic; it is the critic-side half of the
same trust region, and it is the one structural ingredient the two prior rungs left entirely on the
table. The full module is in the answer.

Let me be precise about what this rung is *within this harness*, so I land its implementation and not a
generic one. There is no separate value network — the 2×64 critic is its own head, so the `vf_coef`
combine is a real shared-optimizer term, not a no-op. There is no entropy bonus in play
(`ent_coef=0` by default on MuJoCo, where the Gaussian's learned log-std supplies exploration), so I
include the entropy term in the loss for contract-completeness but expect it to contribute nothing. The
GAE advantages, the per-minibatch normalization, the LR anneal, and the global gradient-norm clip are all
the loop's, frozen; my contribution is exactly the clipped surrogate plus the clipped value loss. This is
PPO-clip as the scaffold realizes it — no more, no less.

Closing on falsifiable expectations against AWR's numbers, since this is the rung that should top the
ladder and the finale will have to clear it. My central bet is *balance*: the hard, per-minibatch,
unit-free band should give up a little of AWR's HalfCheetah peak (I would not be shocked if PPO's
HalfCheetah mean lands *below* AWR's 1996.7, because AWR's 3301 seed inflated that mean and PPO trades
that lucky tail for consistency) but should *not* collapse on Swimmer the way AWR did — I expect PPO's
Swimmer to clear AWR's 90.2 and the penalty rung's 101.4, because the min-clip corrects the spurious-action
overshoots that cratered AWR's seed-123 to 46, and to do so with much tighter seed spread than either
prior rung. On InvertedDoublePendulum I expect PPO to be at least competitive with AWR's 7299.2. The
concrete, falsifiable claim is about the *geometric mean*, not any single environment: PPO should be the
most balanced of the three — no environment where it is the worst by a wide margin — so even if it does
not win every environment outright, its three-environment geometric mean should be the highest of the
baselines, because the task rewards exactly the reliability the hard clip is built to deliver. If that
holds, the only way past PPO is to keep its loss exactly as-is and improve the *exploration* the fixed
Gaussian head supplies — which is where the next move points.
