FastSAC did what I expected on the easy axis and hinted at the trouble on the hard one. As an
off-policy method it reuses the replay buffer, so it cleared PPO comfortably — the consensus is
that an entropy-driven actor-critic beats on-policy PPO across stand, walk, and run under this
budget, closing the sample-efficiency shortfall the PPO feedback diagnosed. But the same consensus
flags the ceiling: on HumanoidBench locomotion the SAC family sits *below* the deterministic
FastTD3 backbone, and the reason is exactly the two-sided expectation I closed the last step with.
A stochastic tanh-Gaussian actor has to maximize entropy over tens of contact-rich action
dimensions, which is a genuinely hard optimization, and it pays an entropy tax — the `−α log π`
term inside the bootstrap and the entropy penalty in the actor loss — that pulls the policy away
from greedily exploiting the value function. On stand, where exploration is cheap and exploitation
is the whole game, that tax is pure cost; on walk and run the entropy helps less than I hoped
because the limiting factor is precision of a learned gait, not breadth of exploration. So the
diagnosis pointing to this step is: I am paying for exploration I can get for free elsewhere, and
that payment is capping the policy below where a pure-exploitation actor would land. The move is to
drop the stochastic actor and the entropy machinery entirely and go deterministic — *if* I can
solve the exploration problem that made me reach for entropy in the first place. The bet of this
rung is that the substrate already solves it: 128 parallel environments, each running the same
deterministic actor with its own independent noise and its own random start, manufacture a broad
data distribution without any entropy objective at all.

Let me build the deterministic off-policy fill carefully, because every piece is a reaction to
either FastSAC's tax or DDPG's brittleness, and the substrate constrains the form. The backbone is
TD3: a deterministic actor `μ(s)` pushed uphill on the critic by the deterministic policy gradient,
twin critics with a clipped-min bootstrap, target-policy smoothing, and delayed actor updates. The
deterministic policy gradient is the source of the exploitation advantage — I differentiate the
critic with respect to the action and backprop into the actor, so the actor moves straight toward
the critic's argmax with no score-function variance and no entropy term holding it back. That is
exactly the greedy exploitation FastSAC sacrificed. But a deterministic actor that just maximizes
the critic is precisely what made DDPG diverge: the critic, bootstrapped through a target that
effectively maximizes over actions, overestimates, and the actor eagerly chases the inflated value.
TD3's clipped double-Q is the fix and it transfers directly to this substrate's distributional
critic the same way FastTD3 already does it: keep two categorical critics, project both target
distributions, read each one's scalar mean `Σ_i p_i z_i`, and keep the *whole distribution*
belonging to whichever critic has the smaller mean — the min selects a distribution by its
expectation, clamping the more-overestimated of the two, and that selected distribution is the
cross-entropy target for both critics. Underestimation, unlike overestimation, does not get chased
and amplified by the policy, which is why the min is the right pessimism.

The crucial difference from FastSAC's critic target is what enters the projection's reward. FastSAC
subtracted `α·log π(a'|s')` to fold in entropy; here there is no entropy term, so the projection
operates on the raw reward, and the next action fed to the target is *not* a stochastic sample but
the deterministic actor's output with target-policy-smoothing noise added: `a' = clip(μ(s') + ε)`
with `ε` a small clipped Gaussian. Target smoothing averages the critic over a small neighborhood
of the target action so the policy cannot exploit a needle-thin spurious peak in the approximate
distributional value — it is a SARSA-flavored regularizer saying nearby actions should have similar
value, and it matters more for a deterministic actor than a stochastic one because the deterministic
actor will otherwise drive straight at whatever sharp maximum the critic happens to have. The actor
objective is then the pure deterministic-policy-gradient one: read the support-weighted means of the
two critics at the actor's (noise-free) action, take the clipped-double-Q minimum, and ascend it —
`actor_loss = −min(Q1, Q2).mean()`. No entropy, no log-prob, no temperature. And the actor updates
on the *delayed* schedule, every `policy_frequency` critic steps, not every step: the two-timescale
trick lets the critic settle before each policy move so the actor is not chasing a critic that is
still thrashing — the opposite of SAC's every-step actor update, and appropriate because a
deterministic actor has nothing damping it but the delay.

Now the exploration that justifies dropping entropy. The actor is deterministic, so on its own it
would trace one thin trajectory; the substrate fixes this by parallelism. Each of the 128
environments runs the same actor but adds its own Gaussian exploration noise, and — the FastTD3
touch — each environment draws its *own* noise scale once from a range `[std_min, std_max]` and
resamples that scale when its episode ends. So the fleet spans a spread of exploration
aggressiveness simultaneously: some envs explore timidly, some boldly, and I never have to find the
single right noise scale for a task because the fleet covers the range. This is mixed per-env noise,
and it is the deterministic answer to FastSAC's entropy: exploration comes from a fleet of noisy
copies of a deterministic policy rather than from the policy itself being stochastic, which means I
keep the clean deterministic-policy-gradient exploitation *and* get a broad data distribution. On a
high-dimensional humanoid action this is the better trade, because injecting noise into 128 parallel
rollouts is trivial whereas maximizing entropy over the action space is hard — exactly the asymmetry
that should let FastTD3 edge past FastSAC.

The architecture is where I make the other deliberate departure from FastSAC, and it goes the other
way than instinct might suggest. FastSAC used LayerNorm and SiLU to stabilize its off-policy critic.
FastTD3's design philosophy is that the stabilization should come from *data*, not architecture:
with a fast-filling, diverse replay buffer (128 envs filling it every step) and the substrate's huge
32,768 batch, each gradient update is low-variance and close to on-distribution, so the deadly triad
is tamed by data diversity rather than by normalization. So the actor and critic are plain
descending-width MLPs with ReLU and no LayerNorm — `512→256→128` for the actor, `1024→512→256` for
the critic — and the deterministic actor's final tanh head is small-initialized so it starts near
zero action. This is lighter and faster than FastSAC's normalized body, and the claim is that on
this substrate it loses nothing in stability because the diversity is doing that job. Everything
else stays matched to the substrate exactly as FastSAC had it — AdamW with weight decay `0.1`,
cosine LR annealing, `num_updates=2` per env step, fast target update `tau=0.1`, the categorical
critic with 101 atoms over `[−250, 250]` — because those are the fast-and-stable settings the loop
is built for.

It is worth naming what the harness does *not* expose, so I do not import machinery that is not
here. There is no separate distributional value head per stream, no n-step machinery beyond the
buffer's `effective_n_steps` discount, no asymmetric critic observations (the loop passes the same
normalized obs to actor and critic), and the bootstrap mask is the substrate's
`(truncations | ~dones)` so that time-limit truncations still bootstrap while true terminations do
not. The FastTD3 fill is therefore the *default* the scaffold ships with — this rung is the
substrate's own baseline — and that is the honest framing: I am not adding to FastTD3, I am
recovering it as the strongest of the three fills by arguing my way from PPO through FastSAC to it.

The falsifiable expectations against the prior numbers. Against FastSAC, FastTD3 should win because
it removes the entropy tax while keeping exploration through the fleet, so I expect it to land
*above* FastSAC on all three tasks, with the clearest margin on **stand** — the task where
exploitation matters most and FastSAC's entropy was pure cost — and a narrower but still positive
margin on walk and run, where some of FastSAC's entropy was actually doing useful exploration so
the gap should shrink. Against PPO the margin should be the largest of all three methods, since
FastTD3 combines experience reuse (beating PPO's on-policy waste) with aggressive deterministic
exploitation (beating PPO's undirected wandering). The risk that would falsify the ordering is if
the plain MLP without LayerNorm proves *less* stable than FastSAC's normalized body on the
harder-to-fit walk/run value surfaces — if the data-diversity-tames-the-triad claim fails at this
scale, FastTD3's critic could be noisier and it could lose to FastSAC on run even while winning on
stand. That precise failure mode — a plain MLP critic that does not convert its capacity into a
clean value fit on the hardest task — is the opening the next rung will attack. The distilled
FastTD3 fill, which is the scaffold default, is in the answer.
