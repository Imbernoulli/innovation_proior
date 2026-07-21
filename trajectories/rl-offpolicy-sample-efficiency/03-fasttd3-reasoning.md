FastSAC did what I expected on the easy axis and hinted at the trouble on the hard one. As an
off-policy method it reuses the buffer, so it cleared PPO comfortably across stand, walk, and run,
closing the sample-efficiency shortfall the PPO feedback diagnosed. But the same consensus flags the
ceiling: on HumanoidBench locomotion the SAC family sits *below* the deterministic FastTD3 backbone,
for exactly the two-sided reason I closed the last step with. A stochastic tanh-Gaussian actor must
maximize entropy over tens of contact-rich action dimensions — a genuinely hard optimization — and it
pays an entropy tax (the `−α log π` inside the bootstrap and the entropy penalty in the actor loss)
that pulls the policy away from greedily exploiting the value function. On stand, where exploration is
cheap and exploitation is the whole game, that tax is pure cost; on walk and run the entropy helps
less than I hoped because the limiting factor is precision of a learned gait, not breadth of
exploration.

Before I rip out the entropy machinery I owe the alternative a fair hearing, because "drop it" is a big
move. The cheaper option is to *keep* FastSAC and detune exploration — lower the target entropy `H̄`,
or cap `α`. But walk through what that buys. The dual update servos `α` toward whatever entropy the
target demands, so lowering `H̄` lets `α` fall and the policy sharpen — yet the *machinery* is still
there: I am still sampling a stochastic next action inside the distributional bootstrap, still paying
the `−α log π` correction, still carrying the log-std head and the tanh-Jacobian log-prob, still
optimizing a tanh-Gaussian over tens of dimensions. As `α → 0` the soft actor-critic degenerates
toward a deterministic one, but by a noisy, indirect route — a reparameterized sampled action whose
gradient has score-function-like variance even at small `α`, and an entropy accounting now doing
nothing but adding variance. Building the deterministic method directly reaches "greedy exploitation
with clean gradients" exactly and cheaply, whereas detuning SAC reaches it asymptotically and
expensively. So the elimination is not aesthetic: keeping SAC and turning `α` down keeps every one of
the stochastic actor's optimization costs while discarding the only thing they bought — exploration —
which I am about to get elsewhere for free. The move is to drop the stochastic actor and the entropy
machinery entirely and go deterministic — *if* I can solve the exploration problem that made me reach
for entropy. The bet of this rung is that the substrate already solves it: 128 parallel environments,
each running the same deterministic actor with its own independent noise and random start, manufacture
a broad data distribution without any entropy objective.

The backbone is TD3: a deterministic actor `μ(s)` pushed uphill on the critic by the deterministic
policy gradient, twin critics with a clipped-min bootstrap, target-policy smoothing, and delayed actor
updates. The deterministic policy gradient is the source of the exploitation advantage — I
differentiate the critic w.r.t. the action and backprop into the actor, so it moves straight toward
the critic's argmax with no score-function variance and no entropy term holding it back. But a
deterministic actor that just maximizes the critic is what made DDPG diverge: the critic, bootstrapped
through a target that effectively maximizes over actions, overestimates, and the actor eagerly chases
the inflated value.

Clipped double-Q is the fix, and I want the *direction* of its bias correction right rather than invoke
it as a spell. Keep two categorical critics, project both target distributions, read each one's scalar
mean `Σ_i p_i z_i`, and keep the whole distribution of whichever has the smaller mean. Why min and not
max or mean? The failure mode is overestimation: because the actor is trained to maximize the critic,
any state where one critic sits high is a state the actor drives toward, so positive noise in the value
is *selected for* and amplified through the policy while negative noise is not — an actor never chases a
value that looks too low. The elementwise min clamps the more-overestimated critic on every state, so
the target is pulled toward the pessimistic estimate and the amplifying loop is cut. Underestimation,
by contrast, is self-correcting: an undervalued action is simply not chosen, and better data later
lifts its value without runaway. So the min is the *right* pessimism, and the selected distribution is
the cross-entropy target for both critics.

The crucial difference from FastSAC's critic target is what enters the projection's reward. FastSAC
subtracted `α·log π(a'|s')`; here there is no entropy term, so the projection operates on the raw
reward, and the next action fed to the target is not a stochastic sample but the deterministic actor's
output with target-policy-smoothing noise: `a' = clip(μ(s') + ε)`, `ε = clip(N(0,policy_noise²),
−noise_clip, noise_clip)`. Smoothing averages the critic over a small neighborhood of the target action
so the policy cannot exploit a needle-thin spurious peak in the approximate value — a SARSA-flavored
regularizer that nearby actions should have similar value, reducing to the plain deterministic target
as `ε → 0`. It matters more for a deterministic actor than a stochastic one because the deterministic
actor goes straight at whatever sharp maximum the critic has, where the stochastic actor's own sampling
already blurred the target action. The actor objective is then pure deterministic policy gradient: read
the support-weighted means of the two critics at the actor's noise-free action, take the min, ascend it
— `actor_loss = −min(Q1,Q2).mean()`, no entropy, no log-prob, no temperature. And the actor updates on
the *delayed* schedule, every `policy_frequency=2` critic steps: the two-timescale trick lets the
critic settle before each policy move so the actor is not chasing a critic that is still thrashing.
Where FastSAC's entropy damped the every-step actor update, here the delay *is* the damper — dropping
the actor's update rate in half is the price of removing the entropy term.

Now the exploration that justifies dropping entropy, with the arithmetic that convinces me the fleet
really substitutes for it. Each of the 128 environments runs the same actor but adds its own Gaussian
noise, and — the key touch — each draws its *own* noise scale from `[std_min, std_max] = [0.001, 0.4]`
and resamples it when its episode ends. The scales are drawn uniformly, so the expected fraction in any
sub-band is that band's width over `0.399`: the bottom tenth (scales below `≈0.04`) holds about a dozen
near-deterministic rollouts refining the current policy tightly, the top half (above `0.2`) holds about
64 probing aggressively, and the middle tiles the rest. So at any instant the buffer receives a
*mixture* over exploration temperatures, not a single one, and because each env reshuffles at episode
end no env is stuck timid or bold — a rollout refining this episode may be probing the next. This is the
deterministic answer to FastSAC's entropy: the same spread over how far the data strays from the
current policy, except it costs a single `torch.rand` per episode instead of a log-std head, a
Jacobian-corrected log-prob, and a dual temperature. On a high-dimensional humanoid action, noising 128
rollouts is trivial where maximizing entropy is hard — exactly the asymmetry that should let the
deterministic backbone edge past FastSAC, keeping the clean deterministic-policy-gradient exploitation
*and* a broad data distribution.

It is worth being precise about *why* that gradient is lower-variance, because it is the mechanical
payoff. The stochastic actor's gradient is a reparameterized expectation evaluated at a *sample*
`a = tanh(μ + σ·ξ)`, so the noise `ξ` enters the gradient and contributes variance that only vanishes
as the batch grows. The deterministic policy gradient evaluates the critic at the single point `μ(s)`
and differentiates through it: `∇_θ Q(s, μ(s)) = ∇_a Q|_{a=μ(s)} · ∇_θ μ(s)`, an exact derivative with
no sampling noise in the actor update. So for the same batch the deterministic actor's gradient is
strictly less noisy, and more of each update is signal — "exploits the value function more aggressively"
made mechanical. The cost is that this clean gradient points only at what the critic currently believes,
which is exactly why the clipped-min and target smoothing are load-bearing and why the fleet must supply
the exploration the clean gradient will not.

The distributional critic (established last rung) keeps a locomotion state's genuinely bimodal return —
recover and bank a long tail, or fall and collect almost nothing — as separate atom masses rather than
one averaged number, so clipped double-Q selects a self-consistent object: one critic's entire belief
about the survive/fall split, not a mean from one spliced onto a shape from another. Under the fast
target update `tau=0.1` this target moves quickly toward the online critic, and the two-timescale delay
keeps the policy from chasing it before it settles.

The architecture is the other deliberate departure from FastSAC, and it goes the opposite way from
instinct. FastSAC used LayerNorm and SiLU to stabilize its critic; here the design philosophy is that
stabilization should come from *data*, not architecture. With a fast-filling, diverse buffer (128 envs
every step) and the substrate's huge 32,768 batch, each gradient update is low-variance and close to
on-distribution, so the deadly triad is tamed by data diversity. The critic body is a plain descending
ReLU MLP `1024→512→256` — comparable parameter scale to FastSAC's normalized body but cheaper per
forward and with no feature-scale controller — and the actor is the same shape one tier down,
`512→256→128`, its tanh head small-initialized (`init_scale=0.01`) so the policy starts near zero action
in the well-conditioned middle of the `[−1,1]` box. The bet is that a 32,768-sample batch from a
constantly-refreshed buffer gives a gradient whose variance is low enough and whose distribution is
close enough to the current policy's that the bootstrapped critic does not need LayerNorm to keep from
amplifying its own scale drift. Everything else stays matched to the substrate exactly as FastSAC had it
— AdamW with weight decay `0.1`, cosine annealing, `num_updates=2`, `tau=0.1`, the categorical critic
with 101 atoms over `[−250,250]`.

One correctness point I want to be sure of: the bootstrap mask is the substrate's `(truncations |
~dones)`. A humanoid that hits the 1000-step time limit while still upright has not reached a terminal
state, so its future value should be bootstrapped; only a genuine fall (`done` without `truncation`)
should have its future zeroed. `(truncations | ~dones)` is 1 exactly when the transition should
bootstrap and 0 only on a true termination — the correct time-limit-aware mask, and it matters more here
than for PPO because a wrong mask poisons every replayed target, not just one discarded batch. The
deterministic fill is in fact the default the scaffold ships with — this rung recovers the substrate's
own baseline as the strongest of the three fills by arguing from PPO through FastSAC to it.

The falsifiable expectations. Against FastSAC this backbone should win by removing the entropy tax while
keeping fleet exploration, so I expect it above FastSAC on all three tasks, with the clearest margin on
`h1hand-stand-v0` — where exploitation matters most and FastSAC's entropy was pure cost — and a narrower
but positive margin on walk and run, where some of FastSAC's entropy was doing useful exploration. Against
PPO the margin should be the largest of the three, since this combines experience reuse with aggressive
deterministic exploitation. The risk that would falsify the ordering is if the plain MLP without LayerNorm
proves *less* stable than FastSAC's normalized body on the harder-to-fit walk/run value surfaces — if the
data-diversity-tames-the-triad claim fails at this scale, the plain critic could be noisier and lose to
FastSAC on run even while winning on stand. That precise failure mode — a plain MLP critic that does not
convert its capacity into a clean value fit on the hardest task — is the opening the next rung will
attack. The distilled deterministic fill, which is the scaffold default, is in the answer.
