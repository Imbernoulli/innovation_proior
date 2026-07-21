PPO topped the ladder by being the most balanced rung, and the numbers show exactly why and exactly
where its remaining weakness lives. Its Swimmer is the tightest and highest of any baseline — 111.9 /
109.7 / 118.1 for a mean of 113.2, a seed range of barely $8.4$ points where AWR's Swimmer ranged 46–124
(a range of $78$) and the penalty rung's 82–113 — so the hard per-minibatch clip did precisely what I
predicted: it corrected the spurious-action overshoots that cratered AWR's seed-123, and the seeds now
track each other. Its InvertedDoublePendulum is steady at 7048.4 (7502.6 / 6968.1 / 6674.5), a best-to-
worst ratio of only $1.12$. And its HalfCheetah, as I bet, came in *below* AWR's mean — 1757.6 vs 1996.7
— because AWR's mean was inflated by one lucky 3301 seed while PPO traded that tail for consistency
(1470.6 / 1361.3 / 2441.0). So PPO wins on the metric that matters here, the geometric mean across all
three environments, by refusing to be the worst anywhere. But look at where it is *not* the best: its
HalfCheetah, at 1757.6, still trails AWR's lucky-seed peak, and its own three seeds are themselves split —
two near 1400 and one at 2441, a $1.79\times$ gap. That single-seed tail is the tell. On HalfCheetah the
difference between a 1400 run and a 2441 run is which gait the policy found, and *that* is decided early,
by whether the policy stayed stochastic long enough to explore into the better basin before committing.
The diagnosis is no longer about the trust region — the clip nailed that; the Swimmer tightness proves it.
It is about *exploration*. The clip controls how far the policy moves per step; it does nothing to keep
the policy *stochastic* enough to keep finding the high-return tail it occasionally hits. That is the
weakness the next move has to attack, and it has to do it without touching the loss the clip earned and
without adding capacity, because the parameter count is frozen by the runtime assertion.

Let me localize the exploration problem in this exact substrate, because the fix has to fit the edit
surface. Exploration here flows entirely through one object: the state-independent learned log-std vector
`actor_logstd`, which sets the width of the Gaussian the policy samples from. Early in training that std
is reasonably wide and the policy explores; but the policy gradient, the clip, and especially the GAE
advantages all *reward* sharpening the policy onto whatever has been working, and the cleanest way the
optimizer raises return is to shrink the std — make the Gaussian narrow and near-deterministic around the
current best action. Let me see this in the gradient. The entropy of a diagonal Gaussian is
$\sum_i(\log\sigma_i + \tfrac12\log2\pi e)$, so it depends *only* on `actor_logstd`, and with `ent_coef=0`
nothing in the loss opposes its decrease; meanwhile the surrogate's gradient with respect to $\log\sigma$
is generally negative once the policy has found actions with positive advantage, because a tighter
distribution assigns those actions higher probability. So $\sigma$ ratchets down monotonically, and once
it collapses, exploration is over: the policy commits to a basin. On HalfCheetah that is exactly why two
of three seeds plateau around 1400 while one occasionally finds the 2441 gait — the difference is whether
the policy stayed stochastic long enough to find the better basin before its std collapsed. PPO's clip
does not arrest this collapse at all; it only bounds the *step*, not the *entropy*.

So what are my options for injecting exploration without touching the earned loss or the parameter count?
Let me lay them out and eliminate. The obvious lever is the entropy bonus the loop already exposes:
set `ent_coef>0` so the loss gains $-c\,H(\pi)$ and the optimizer is paid to keep $\sigma$ wide. But this
is the blunt instrument, and it fails for a computable reason. A fixed `ent_coef` adds an entropy reward
measured in nats, and it competes against the policy-gradient term measured in *advantage-scaled return*.
The three environments live at wildly different return scales — HalfCheetah around $1500$, Swimmer around
$110$, InvertedDoublePendulum around $7000$ — so a single `ent_coef` that supplies enough entropy pressure
to keep Swimmer exploring (where returns, and hence advantage magnitudes, are small) would be swamped on
InvertedDoublePendulum, and one large enough to matter on the pendulum would over-inflate Swimmer's std
and wreck its long-horizon credit assignment — and worse, on the unstable pendulum too much entropy
literally knocks the pole over. That is the *same* per-environment-coefficient disease the penalty rung's
$\beta$ had: a single scalar cannot balance three reward scales at once. A second option is to floor or
anneal `actor_logstd` directly — clamp it from below so it cannot collapse. But that floor is itself a
per-environment number in *action* units, and it fights the optimizer head-on rather than shaping the
landscape, so it trades the collapse for a hand-tuned constant that will be wrong on at least one of the
three. I want exploration that does not need a per-environment knob — something that measures itself
against the policy's *own* current width rather than against the reward scale.

Here is the move I want to make, and it is a published method — Robust Policy Optimization (Rahman & Xue,
2022) — and it is exactly the shape I just described. Keep PPO's clipped loss *completely unchanged* and
instead perturb the policy's action-mean during the update with a small uniform noise, $z\sim\mathcal
U(-\alpha,\alpha)$ added to `action_mean` before the log-probability is evaluated. Crucially the
perturbation is applied **only during the update** — when an action is being re-scored, i.e. when `action
is not None` in `get_action_and_value` — and *not* during data collection, when the policy samples actions
to step the environment. So the agent still acts with its clean, unperturbed Gaussian (the rollout is
undisturbed, the actions stored in the buffer are honest samples from $\pi_{old}$), but every time the
loss re-evaluates $\log\pi_\theta(a|s)$ over the ten epochs, it does so under a *family* of slightly
shifted means.

Let me derive why this maintains entropy without an entropy coefficient, because the mechanism is the
whole justification and I want it on paper, not asserted. Re-score a stored action $a$ under a mean
perturbed by $z$: the per-dimension negative log-likelihood is $\tfrac12(a-\mu-z)^2/\sigma^2 + \log\sigma
+ \text{const}$. Average over $z\sim\mathcal U(-\alpha,\alpha)$, which is zero-mean with variance
$\mathrm{Var}(z)=\alpha^2/3$. Since $\mathbb E_z[(a-\mu-z)^2]=(a-\mu)^2+\mathrm{Var}(z)$, the expected NLL
gains a term relative to the unperturbed loss of exactly
$$\Delta = \frac{\mathrm{Var}(z)}{2\sigma^2} = \frac{\alpha^2}{6\sigma^2}\quad\text{per action dimension.}$$
Put $\alpha=0.5$: $\Delta = 0.25/(6\sigma^2)\approx 0.0417/\sigma^2$. Read what this expression *does*. It
is an extra cost added to the policy loss that blows up as $1/\sigma^2$ when $\sigma\to0$ — so the closer
the policy drifts to deterministic, the harder this term pushes back, and its gradient with respect to
$\log\sigma$ is $-2\Delta = -\alpha^2/(3\sigma^2)$, i.e. it *always* pushes $\sigma$ up, with a force that
grows without bound as the std collapses. That is an implicit entropy regularizer — and here is the part
that answers the `ent_coef` objection: its strength is $(\alpha/\sigma)^2$, measured in units of the
policy's *own* std $\sigma$, not in nats against the reward scale. So it self-normalizes to each
environment: wherever $\sigma$ has collapsed, the pressure is strong; wherever $\sigma$ is already wide,
$\Delta$ is small and it barely intrudes. One $\alpha$ therefore transfers across HalfCheetah, Swimmer,
and InvertedDoublePendulum far better than one `ent_coef` could, because it never has to be commensurated
against three different return magnitudes. That is the exact defect of the entropy-bonus option, removed
by construction.

Let me pin $\alpha=0.5$ with a limit check on both ends, because the same $1/\sigma^2$ blow-up that makes
the regularizer useful also makes it dangerous if $\alpha$ is mis-scaled. As $\alpha\to0$, $\Delta\to0$
and the method degrades continuously to plain PPO — good, so there is no discontinuity and $\alpha$ is a
genuine dial with PPO at one end. At the other end, the perturbation must stay small *relative to the
action range*: these MuJoCo policies act into a bounded action space (roughly $[-1,1]$ per coordinate
before the env rescales), so a mean shift of $\alpha=0.5$ is half of one side of that range — large enough
to matter, but not so large that the re-scored action lands in a part of action space the policy never
visits. If I pushed $\alpha$ up toward, say, $2$, the shifted mean $\mu+z$ would routinely sit outside the
plausible action region, the re-scored `newlogprob` would be dominated by perturbation noise rather than
by the surrogate's advantage signal, and the ratio would swing so hard that the clip is fighting the noise
instead of the policy drift — learning would break. So $\alpha=0.5$ is bounded above by the action scale
and below by "large enough that $\Delta$ resists collapse," and the reported default sits in that window.

That window has a sharp interpretation if I ask where $\sigma$ settles. The surrogate pushes $\log\sigma$
*down* with some environment-dependent force $g$ (its gradient once positive-advantage actions have been
found); the perturbation pushes *up* with $\alpha^2/(3\sigma^2)$. They balance at
$\sigma^\* \approx \alpha/\sqrt{3g}$ — so the maintained std scales *with* $\alpha$, and the method holds
the policy at a stochasticity floor set by $\alpha$ rather than by a per-environment number. Put the
numbers in to see how sharp the floor is. At $\sigma=1$, $\Delta=0.0417$ per dim — negligible against an
$O(1)$ advantage-scaled surrogate, so the regularizer is essentially off and the policy is free to
sharpen. At $\sigma=0.2$, $\Delta=0.0417/0.04\approx1.04$ — now comparable to the surrogate, so the
up-pressure is real. At $\sigma=0.1$, $\Delta\approx4.2$ — overwhelming; the policy simply cannot afford
to be that deterministic. So the perturbation acts like a soft floor that is inert above $\sigma\approx0.5$
and dominant below $\sigma\approx0.2$, catching the std exactly in the collapse regime that traps PPO's
HalfCheetah seeds and nowhere else. That is the property `ent_coef` cannot buy: a *state-of-the-policy*-
triggered pressure rather than a flat reward-scale-weighted one.

There is a second, subtler benefit that falls out of the same averaging: it *smooths the loss landscape*.
The optimizer no longer sees the single sharp surrogate at the current mean; it sees an average of the
surrogate over a $\pm\alpha$ neighborhood of means, which damps the narrow, over-confident maxima the
GAE-plus-clip machinery would otherwise sharpen onto. That is directly aimed at the HalfCheetah tail: a
smoothed landscape is less likely to trap a seed in the 1400 basin, so more seeds should reach the 2441-
class gait, which is the specific place PPO left return on the table.

Why uniform noise and not Gaussian, and why perturb the *mean* and not the *std* directly? Uniform noise
gives a hard, bounded perturbation — every update sees a mean shifted by at most $\alpha$, no heavy tail —
which keeps the perturbation from ever being large enough to break the trust region the clip is
maintaining; the perturbation feeds into `newlogprob`, hence into the ratio $r$, and the clip still bounds
$|r-1|$ on the *perturbed* policy, so the two mechanisms compose rather than fight. A Gaussian perturbation
would occasionally throw a large shift that spikes the ratio and collides with the clip. And I perturb the
mean rather than directly inflating the std because perturbing the std would change the *sampling*
distribution and corrupt the honest $\pi_{old}$ rollout if applied at collection time, whereas perturbing
the mean only at *re-scoring* time leaves the rollout untouched and acts purely as a regularizer on the
gradient. This is also why the perturbation must be gated on `action is not None`: in the data-collection
call `action is None`, the policy samples cleanly; only in the re-scoring call, where the stored action is
passed back in, does the noise enter. Getting that gate wrong — perturbing at collection — would inject
noise into the behavior policy and silently break the on-policy assumption that the whole ladder rests on,
so the `if action is None: sample else: perturb-then-rescore` structure is load-bearing, not incidental.

There is a structural reason this is the natural next move. Every previous rung fought over
`compute_losses` and left `get_action_and_value` at the default; this one is the mirror image —
`compute_losses` stays byte-for-byte PPO and only `get_action_and_value`, the *distribution*, changes. The
loss has been walked from a soft KL penalty to a ratio-free regression to a hard clip, exhausting what it
alone can do; the return PPO still leaves on the table lives not in the loss but in the policy's own
stochasticity, which only `get_action_and_value` controls. So this composes with PPO rather than replacing
it — reaching the one part of the substrate no prior rung touched while preserving every gain.

Now the harness fit, which is the cleanest of any rung on this ladder. RPO changes *only*
`get_action_and_value`; `compute_losses` is byte-for-byte PPO's clipped surrogate plus clipped value loss
— the loss the previous rung earned, untouched, which is exactly what I wanted: improve exploration
without disturbing the trust region. The edit adds three lines to the `else` branch of
`get_action_and_value`: sample $z=$ `torch.FloatTensor(action_mean.shape).uniform_(-rpo_alpha, rpo_alpha)`
on the observation's device, set `action_mean = action_mean + z`, and rebuild `probs = Normal(action_mean,
action_std)` before computing the log-prob, entropy, and value against the *perturbed* distribution. The
default $\alpha=0.5$ is the value the method reports as robust across a broad continuous-control suite, and
I keep it hardcoded as a local since the fixed `Args` does not expose it. Critically for this task's
parameter-count guard: $z$ is *sampled noise*, not a learnable parameter, so it adds nothing to the count
the runtime assertion checks — the contribution is algorithmic, exactly as the frozen-capacity constraint
demands. No replay buffer, no extra network, no new hyperparameter in the config; it slots into the frozen
loop exactly. The full module is in the answer.

The bar this must clear is PPO's numbers: HalfCheetah 1757.6, Swimmer 113.2, InvertedDoublePendulum
7048.4. The claim is that a scale-free implicit-entropy perturbation improves over PPO while never
destabilizing — the balanced-reliability axis the task scores on — so the falsifiable bar is three-fold.
HalfCheetah: I expect the mean above 1757.6 and, more tellingly, the per-seed *minimum* to rise as the
smoothed landscape and arrested std-collapse let more seeds escape the 1400 basin into the 2441-class gait
— fewer trapped seeds rather than a higher ceiling. Swimmer: *do no harm* — the $\alpha^2/(6\sigma^2)$
pressure must not inflate the std enough to reintroduce the long-horizon collapse, so it should stay in
PPO's tight band. InvertedDoublePendulum is the risk environment: unstable balancing where too much
exploration knocks the pole over, and $\Delta$ is largest exactly when $\sigma$ is small — a converged
pendulum policy — so I watch the worst seed for a drop below 7048.4. The geometric-mean bet is that the
fixed perturbation clears PPO by lifting the exploration-limited environment while holding the other two;
if instead it drags the pendulum or destabilizes Swimmer, that falsifies $\alpha=0.5$ transferring across
these three dynamics, and the move past it would be to anneal $\alpha$ or make it state-dependent so the
pressure eases where the policy must stay decisive. This is the reliability the clip bought, extended from
the *step* to the *entropy*.
