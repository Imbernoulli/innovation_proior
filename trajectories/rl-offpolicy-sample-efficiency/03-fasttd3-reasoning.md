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
because the limiting factor is precision of a learned gait, not breadth of exploration.

Before I commit to ripping out the entropy machinery, I owe the alternative a fair hearing, because
"drop it" is a big move. The cheaper option is to *keep* FastSAC and simply detune the exploration:
lower the target entropy `H̄`, or cap `α` so the dual variable cannot spend much on entropy. But
walk through what that actually buys. The dual update servos `α` toward whatever entropy the target
demands, so lowering `H̄` lets `α` fall and the policy sharpen — yet the *machinery* is still there:
I am still sampling a stochastic next action inside the distributional bootstrap, still paying the
`−α log π` correction, still carrying the log-std head and the tanh-Jacobian log-prob, still optimizing
a tanh-Gaussian over tens of dimensions. As `α → 0` the soft actor-critic degenerates toward a
deterministic one, but by a noisy, indirect route — a reparameterized sampled action whose gradient
has score-function-like variance even at small `α`, and an entropy accounting that is now doing
nothing but adding variance. If the endpoint I want is "greedy exploitation with clean gradients,"
detuning SAC reaches it asymptotically and expensively, whereas building the deterministic method
directly reaches it exactly and cheaply. So the elimination is not aesthetic: keeping SAC and turning
`α` down keeps every one of the stochastic actor's optimization costs while discarding the only thing
those costs bought — exploration — which I am about to get elsewhere for free. The move is to drop the
stochastic actor and the entropy machinery entirely and go deterministic — *if* I can solve the
exploration problem that made me reach for entropy in the first place. The bet of this rung is that
the substrate already solves it: 128 parallel environments, each running the same deterministic actor
with its own independent noise and its own random start, manufacture a broad data distribution without
any entropy objective at all.

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

TD3's clipped double-Q is the fix, and I want to verify the *direction* of its bias correction
rather than invoke it as a spell, because getting the sign wrong would make it a de-stabilizer. Keep
two categorical critics, project both target distributions, read each one's scalar mean `Σ_i p_i z_i`,
and keep the *whole distribution* belonging to whichever critic has the smaller mean. Why the min and
not the max or the mean? The failure mode is overestimation: because the actor is trained to maximize
the critic, any state where one of the two critics happens to sit high is a state the actor will drive
toward, so positive noise in the value estimate is *selected for* and amplified through the policy,
while negative noise is not — an actor never chases a value that looks too low. Taking the elementwise
`min` of the two critics' means clamps the more-overestimated of the pair on every state, so the
target the actor optimizes is systematically pulled toward the pessimistic estimate, and the amplifying
feedback loop is cut. Underestimation, by contrast, is self-correcting: an undervalued action is simply
not chosen, and the next time better data arrives its value rises without any runaway. So the min is the
*right* pessimism, not merely *a* pessimism, and the direction check confirms it. The selected
distribution is then the cross-entropy target for both critics, so both are trained toward the same
pessimistic bootstrap.

The crucial difference from FastSAC's critic target is what enters the projection's reward. FastSAC
subtracted `α·log π(a'|s')` to fold in entropy; here there is no entropy term, so the projection
operates on the raw reward, and the next action fed to the target is *not* a stochastic sample but
the deterministic actor's output with target-policy-smoothing noise added: `a' = clip(μ(s') + ε)`
with `ε` a small clipped Gaussian, `ε = clip(N(0, policy_noise²), −noise_clip, noise_clip)`. Target
smoothing averages the critic over a small neighborhood of the target action so the policy cannot
exploit a needle-thin spurious peak in the approximate distributional value — it is a SARSA-flavored
regularizer saying nearby actions should have similar value. Check its limit: as the smoothing noise
`ε → 0` the target reduces to `a' = clip(μ(s'))`, the plain deterministic-policy-gradient target, so
smoothing is a strict softening of exactly the sharp-argmax target that lets a deterministic actor
drive at a spurious critic peak — it can only help stability and reduces to the base method when
turned off. It matters more for a deterministic actor than a stochastic one because the deterministic
actor will otherwise go straight at whatever sharp maximum the critic happens to have, where the
stochastic actor's own sampling already blurred the target action. The actor objective is then the
pure deterministic-policy-gradient one: read the support-weighted means of the two critics at the
actor's (noise-free) action, take the clipped-double-Q minimum, and ascend it —
`actor_loss = −min(Q1, Q2).mean()`. No entropy, no log-prob, no temperature. And the actor updates
on the *delayed* schedule, every `policy_frequency=2` critic steps, not every step: the two-timescale
trick lets the critic settle before each policy move so the actor is not chasing a critic that is
still thrashing — the opposite of SAC's every-step actor update, and appropriate because a
deterministic actor has nothing damping it but the delay. Where FastSAC could update its actor every
step because entropy damped it, here the delay *is* the damper, and dropping the actor's update rate
in half is the price of removing the entropy term.

Now the exploration that justifies dropping entropy, and it is worth doing the arithmetic that
convinces me the fleet really substitutes for the entropy objective. The actor is deterministic, so
on its own it would trace one thin trajectory; the substrate fixes this by parallelism. Each of the
128 environments runs the same actor but adds its own Gaussian exploration noise, and — the key
touch — each environment draws its *own* noise scale once from a range `[std_min, std_max] =
[0.001, 0.4]` and resamples that scale when its episode ends. So at any instant the fleet holds 128
noise scales spread roughly uniformly across `[0.001, 0.4]`, mean about `0.2`: some dozens of envs
explore almost deterministically near `0.001` and refine the current policy, some dozens explore
boldly near `0.4` and probe far from it, and the rest tile the range in between. The data the buffer
sees each step is therefore a *mixture* over exploration temperatures, not a single one — and because
each env resamples its scale at episode end, the assignment reshuffles constantly so no env is stuck
timid or stuck bold. This is the deterministic answer to FastSAC's entropy: exploration comes from a
fleet of noisy copies of a deterministic policy spanning a range of aggressiveness, rather than from
the policy itself being stochastic, which means I keep the clean deterministic-policy-gradient
exploitation *and* get a broad data distribution. I never have to find the single right noise scale
for a task because the fleet covers the range and the buffer averages over it. On a high-dimensional
humanoid action this is the better trade, because injecting scalar-scaled Gaussian noise into 128
parallel rollouts is trivial whereas maximizing entropy over the whole action space is hard — exactly
the asymmetry that should let a deterministic backbone edge past FastSAC.

It is worth being precise about *why* the deterministic policy gradient is lower-variance than the
stochastic one FastSAC used, because that is the concrete mechanical payoff of this whole rung. The
stochastic actor's gradient is a reparameterized expectation over a sampled action — even with the
reparameterization trick, the gradient is evaluated at a *sample* `a = tanh(μ + σ·ξ)`, so the noise
`ξ` enters the gradient and contributes variance that only vanishes as the batch grows. The
deterministic policy gradient evaluates the critic at the single point `μ(s)` and differentiates
through it: `∇_θ Q(s, μ(s)) = ∇_a Q|_{a=μ(s)} · ∇_θ μ(s)`, an exact derivative with no sampling noise
in the actor update at all. So for the same batch, the deterministic actor's gradient is strictly
less noisy, which means it can move more decisively toward the critic's argmax each step. That is the
"exploits the value function more aggressively" claim made mechanical: it is not that the deterministic
actor tries harder, it is that its gradient is cleaner, so more of each update is signal. The cost is
that this clean gradient points only at whatever the critic currently believes — which is exactly why
the clipped-min and the target smoothing above are load-bearing, and why the fleet must supply the
exploration the clean gradient will not.

The distributional critic earns its keep in this deterministic setting for a reason worth spelling
out, because it is what makes the clipped-double-Q selection coherent. On a locomotion task a state's
return is genuinely bimodal near the edge of stability: from here the humanoid either recovers and
banks a long tail of future reward, or falls and collects almost nothing, and a scalar critic would
average those two futures into a single number that describes neither. The categorical critic keeps
the two modes as separate atom masses, so the value estimate carries the *shape* of the return, not
just its mean. When clipped double-Q reads each critic's scalar mean and keeps the whole distribution
of the smaller-mean critic, it is selecting a coherent distribution — one critic's entire belief about
the survive/fall split — rather than splicing a mean from one critic onto a shape from another. That
is why "keep the whole distribution whose mean is smaller" is the right operation and not just a
convenient one: the pessimism is applied to a self-consistent object. Under the fast target update
`tau=0.1` this distributional target moves quickly toward the online critic, and the two-timescale
delay on the actor keeps the policy from chasing that fast-moving target before it settles — the pieces
interlock.

The architecture is where I make the other deliberate departure from FastSAC, and it goes the other
way than instinct might suggest. FastSAC used LayerNorm and SiLU to stabilize its off-policy critic.
The design philosophy here is that the stabilization should come from *data*, not architecture: with
a fast-filling, diverse replay buffer (128 envs filling it every step) and the substrate's huge
32,768 batch, each gradient update is low-variance and close to on-distribution, so the deadly triad
is tamed by data diversity rather than by normalization. Let me size the claim honestly. The critic
body is a plain descending MLP `1024→512→256`, so `(n_in·1024) + (1024·512) + (512·256) ≈ n_in·1024 +
655k` weights plus the atom head — comparable parameter scale to FastSAC's normalized body, but with
ReLU and no LayerNorm, so cheaper per forward and with no feature-scale controller. The bet is that a
32,768-sample batch drawn from a buffer that 128 envs refresh every step gives a gradient whose
variance is low enough, and whose distribution is close enough to the current policy's, that the
bootstrapped critic does not need LayerNorm to keep from amplifying its own scale drift — the diversity
does that job. The actor is the same shape one width-tier down, `512→256→128` with ReLU, and its final
tanh head is small-initialized (`init_scale=0.01`) so it starts near zero action, keeping the policy in
the well-conditioned middle of the `[−1, 1]` box at the start of training. This is lighter and faster
than FastSAC's normalized body, and the claim is that on this substrate it loses nothing in stability
because the diversity is doing that job. Everything else stays matched to the substrate exactly as
FastSAC had it — AdamW with weight decay `0.1`, cosine LR annealing, `num_updates=2` per env step,
fast target update `tau=0.1`, the categorical critic with 101 atoms over `[−250, 250]` — because those
are the fast-and-stable settings the loop is built for.

I can make the fleet-coverage argument quantitative enough to trust it as a real substitute for
entropy. The 128 noise scales are drawn uniformly on `[0.001, 0.4]`, so the expected fraction landing
in any sub-band is just that band's width over the total `0.399`. The bottom tenth of the range,
scales below `≈0.04`, holds about a tenth of the envs — a dozen or so near-deterministic rollouts that
refine the current policy tightly — while the top half, scales above `0.2`, holds about 64 envs
probing aggressively, and the middle tiles the rest. So every step the buffer receives, in expectation,
roughly `13` refinement-grade rollouts, `~50` moderate ones, and `~64` bold ones, all under the same
deterministic policy. That simultaneous coverage is the thing entropy gave FastSAC — a spread over how
far the data strays from the current policy — except here it costs a single `torch.rand` per episode
instead of a log-std head, a Jacobian-corrected log-prob, and a dual temperature. And because each env
reshuffles its scale at episode end, a rollout that was refining this episode may be probing the next,
so no region of the exploration spectrum is permanently assigned to the same states. This is why the
fleet is not a poor man's entropy but a genuinely different and cheaper mechanism for the same breadth.

It is worth naming what the harness does *not* expose, so I do not import machinery that is not
here. There is no separate distributional value head per stream, no n-step machinery beyond the
buffer's `effective_n_steps` discount, no asymmetric critic observations (the loop passes the same
normalized obs to actor and critic), and the bootstrap mask is the substrate's
`(truncations | ~dones)` so that time-limit truncations still bootstrap while true terminations do
not. That mask is a small correctness point I want to be sure of: a humanoid that hits the 1000-step
time limit while still upright has *not* reached a terminal state, so its future value should be
bootstrapped; only a genuine fall (`done` without `truncation`) should have its future zeroed. The
expression `(truncations | ~dones)` is 1 exactly when the episode was truncated or was not done, i.e.
whenever the transition should bootstrap, and 0 only on a true termination — which is the correct
time-limit-aware bootstrap and matters more here than for PPO because a wrong mask poisons every
replayed target, not just one discarded batch. The deterministic fill is therefore the *default* the
scaffold ships with — this rung is the substrate's own baseline — and that is the honest framing: I am
not adding to it, I am recovering it as the strongest of the three fills by arguing my way from PPO
through FastSAC to it.

The falsifiable expectations against the prior methods. Against FastSAC, this deterministic backbone
should win because it removes the entropy tax while keeping exploration through the fleet, so I expect
it to land *above* FastSAC on all three tasks, with the clearest margin on `h1hand-stand-v0` — the
task where exploitation matters most and FastSAC's entropy was pure cost — and a narrower but still
positive margin on `h1hand-walk-v0` and `h1hand-run-v0`, where some of FastSAC's entropy was actually
doing useful exploration so the gap should shrink. Against PPO the margin should be the largest of all
three methods, since this combines experience reuse (beating PPO's on-policy waste) with aggressive
deterministic exploitation (beating PPO's undirected wandering). The risk that would falsify the
ordering is if the plain MLP without LayerNorm proves *less* stable than FastSAC's normalized body on
the harder-to-fit walk/run value surfaces — if the data-diversity-tames-the-triad claim fails at this
scale, the plain critic could be noisier and it could lose to FastSAC on run even while winning on
stand. That precise failure mode — a plain MLP critic that does not convert its capacity into a clean
value fit on the hardest task — is the opening the next rung will attack. The distilled deterministic
fill, which is the scaffold default, is in the answer.
