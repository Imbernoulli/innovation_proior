The rung I start from is the one a robotics practitioner reaches for by default, because it is
the honest floor for "train a humanoid policy under a fixed interaction budget": Proximal Policy
Optimization. It is the floor not because it is bad in absolute terms — on a massively parallel
simulator PPO is a workhorse — but because the budget here is the thing that decides everything,
and PPO spends the budget in the most expensive currency there is. So before I design anything
off-policy, I want to be precise about *why* on-policy is the weak baseline on this task, what it
should and should not be able to do on stand, walk, and run, and exactly how it sits relative to
the editable algorithm surface, because that relationship is itself informative.

Let me write down what PPO optimizes so the budget argument is exact. PPO maximizes a clipped
surrogate `L^CLIP = Ê_t[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)]`, where `r_t = π_θ/π_old` is the
probability ratio and `Â_t` is a generalized-advantage estimate. The clip is a trust region
expressed as a flat spot in the loss: once the policy moves a sample's ratio past `1±ε` in the
direction the advantage favors, the gradient through that sample dies, so I can safely run several
epochs of minibatch SGD over a single batch of rollouts without the policy collapsing. The
advantage is `Â_t = Σ_l (γλ)^l δ_{t+l}` with `δ_t = R_t + γV(s_{t+1}) − V(s_t)`, the value head
trades bias against variance through λ, and an entropy bonus keeps the Gaussian policy from
committing too early. This is a clean, reliable algorithm. The catch is the word *on-policy*: the
`Â_t` are computed under the policy that generated the rollouts, and the surrogate is only an
honest proxy for improvement while `π_θ` stays close to `π_old`. After a handful of epochs I have
extracted what I safely can from this batch, and I must throw it away and collect fresh rollouts
with the updated policy. Every transition feeds at most a few gradient steps and then it is gone.

I want to put a number on "at most a few," because that is the whole argument and it should be
arithmetic, not a slogan. Concretely the answer runs PPO with `n_steps=2048` across 16 parallel
envs, so each rollout batch is `2048 × 16 = 32,768` transitions; it does `n_epochs=10` passes over
that batch at `batch_size=256`, which is `32,768 / 256 = 128` minibatch updates per epoch, `1,280`
gradient steps per rollout — and then the batch is discarded and never seen again. So the *reuse
factor* of any given frame under PPO is exactly a constant: 10, one per epoch, hard-capped by the
on-policy contract. That constant is the ceiling PPO cannot raise without breaking its own trust
region, because doing more epochs on the same batch pushes `π_θ` further from `π_old`, the
importance ratios drift past `1±ε`, the clip flattens the gradient to zero on more and more
samples, and the extra epochs buy nothing. Now hold that against what the editable off-policy
surface can do with the same frame. The fixed loop keeps every transition in a GPU replay buffer
and runs `num_updates=2` gradient steps *per env step* for the life of the run; a transition
persists in the buffer across its whole residency and is eligible to be resampled on every one of
those updates. Its reuse factor is not a small constant — it grows with how long it lives in the
buffer and how often the sampler draws it, and under a fast-filling buffer that is many times ten.
That single ratio — 10, fixed, versus a large and buffer-bounded number — is the entire reason the
editable surface on this task is an *off-policy* actor-critic and not a policy-gradient loop, and
it is why PPO is the floor: it is the one method here that structurally cannot touch the buffer.

Now put that against the budget in the other units too. The task gives 100,000 gradient steps with
128 parallel environments — about 12.8M environment frames of interaction. PPO cannot reuse a
frame across iterations, so its entire learning signal is whatever it can wring out of each fresh
batch of 32,768 before discarding it. To stay stable it wants large, decorrelated batches and a
modest number of epochs; to fit the humanoid's high-dimensional, contact-rich dynamics it wants
*many* such batches. Under a tight frame budget those two demands fight: I can have stable updates
or many updates, not both. If I grow the batch to decorrelate the gradient I burn frames faster
and get fewer policy improvements inside the budget; if I shrink it to get more improvements I
raise the gradient variance and the clip starts fighting noisy ratios. An off-policy method sidesteps
the trade entirely because it decouples the gradient count from the frame count — it revisits stored
transitions — which is exactly the decoupling PPO forbids itself.

There is a second, quieter tax hiding inside "12.8M frames" that I want to make explicit, because
it compounds the reuse ceiling. PPO's value head is trained only on the returns of the batch in
front of it, and it is discarded-and-refit along with the policy every iteration. Count the value
updates the same way I counted the policy ones: 128 minibatch value regressions per epoch, 10
epochs, so `1,280` value SGD steps per rollout, on `32,768` fresh targets that will never be seen
again. The advantage estimate `Â_t` leans on `V` — through every `δ_{t+l} = R + γV(s_{t+1}) − V(s)`
in the GAE sum — so an undertrained value head feeds a biased advantage straight into the policy
gradient. Off-policy learning breaks this too: its critic is trained continuously on the whole
buffer with a bootstrapped target, so it accumulates a value estimate across the entire run instead
of refitting from scratch each iteration. PPO pays twice for being on-policy — once in the policy's
reuse ceiling, once in a value head that never gets to converge — and both bills come due precisely
on the long-horizon tasks where an accurate long-horizon value is what walking and running need.

The exploration side has its own geometric cost that is worth stating concretely, because it
explains why the descending stand-walk-run pattern is not incidental. PPO explores by sampling from
a factorized Gaussian over the action, one independent coordinate per actuator, so for a humanoid
with tens of action dimensions the exploration is an isotropic cloud in a very high-dimensional box
`[−1, 1]^d`. Undirected noise in `d` dimensions spreads its budget across all `d` coordinates at
once, but a coordinated gait lives on a thin, correlated manifold — the joints must move together in
a specific phase relationship — and the chance that independent per-coordinate jitter happens to
trace that correlated pattern for long enough to earn reward falls off fast as `d` grows and as the
required behavior gets longer-horizon. Standing needs almost no coordination across the manifold, so
the isotropic cloud finds it; running needs a tight, sustained coordination, so the same cloud
mostly wastes its samples off-manifold. The entropy bonus `ent_coef=0.01` keeps that cloud from
collapsing prematurely, which is the right instinct, but it is still *undirected* — it widens the
search without pointing it — and under a capped frame budget widening an isotropic search in high
dimensions is exactly the wrong economy. This is the concrete mechanism behind "PPO is worst where
coordination matters most," and it is the second thing the off-policy surface is built to fix: not
just to reuse frames, but to let exploration be *directed by the value/objective* rather than
sprayed uniformly across the action box.

It is worth walking the on-policy design space to confirm the floor is *on-policy learning itself*
and not this particular on-policy algorithm, because if a different policy-gradient method escaped
the trap I would reach for it instead. Vanilla REINFORCE takes one gradient step per sample from
Monte-Carlo returns; it is even more sample-hungry than PPO because it has no value baseline to cut
variance and no clip to license multi-epoch reuse, so its reuse factor is 1, strictly worse. TRPO
enforces the trust region exactly with a KL constraint solved by Fisher-vector products; it is more
principled than PPO's clip but it pays a conjugate-gradient inner loop of Hessian-vector products
per update and is still, definitionally, on-policy — it recomputes advantages under the current
policy and discards the batch just the same, so it spends *more* compute to buy the *same* frame
efficiency PPO has. Tuning PPO harder — bigger batches, a KL-adaptive clip, a better GAE λ — moves
the operating point along the stability-versus-throughput curve but never lifts the reuse ceiling
off 10. Every road inside the on-policy family dead-ends at the same wall: the rollouts are thrown
away. So the floor is not "PPO is a weak algorithm," it is "on-policy learning is the wrong currency
for a fixed frame budget," and the only exit is experience reuse — the replay buffer — which is off
the on-policy map and squarely on the editable surface.

That structural fact also shows up in how PPO is run here, and it is worth being explicit because
it tells me what PPO is *not*. PPO does not fill the editable `Actor`/`Critic`/`update_*`
contract at all — those functions assume a replay buffer, a distributional critic, target
networks, and a per-env exploration scheme, none of which an on-policy method uses. Trying to shoe-
horn PPO into the off-policy contract would mean sampling old buffer transitions and correcting
their staleness with importance weights `π_θ/π_behavior`, and those weights explode as the buffer
ages and the behavior policy recedes — the very instability the clip was invented to avoid, now
unbounded because the data is arbitrarily off-policy. So PPO runs as its own self-contained
on-policy script: a stable-baselines3 `MlpPolicy` with separate `[256, 256]` tanh policy and value
heads, `n_steps=2048` per env across 16 parallel environments, 10 epochs per batch, `gae_lambda=0.95`,
`clip_range=0.2`, `ent_coef=0.01`, learning rate `3e-4`, trained for 800,000 PPO steps to match the
same 12.8M-frame budget. It bypasses the off-policy infrastructure entirely. That is the right way
to give PPO its fair shot, and it is also exactly why PPO is the floor: it opts out of the one
mechanism — experience reuse — that the rest of the ladder is built to exploit.

Before I trust any of this, let me sanity-check the two knobs that carry the argument, by looking at
their limits. The clip `ε`: as `ε → 0` the flat spot collapses onto `r_t = 1`, so the moment the
policy moves at all the gradient dies and PPO makes no progress — a trust region shrunk to a point.
As `ε → ∞` the `min` and `clip` become inert, `L^CLIP` degenerates to the raw surrogate `Ê_t[r_t Â_t]`,
and PPO is back to unconstrained importance-weighted policy gradient with all its instability. The
chosen `0.2` sits between those two degeneracies, which is the entire design intent of the clip, and
confirms my reading of it as a trust region rather than a loss trick. The GAE `λ` checks out the
same way: at `λ = 0`, `Â_t = δ_t = R_t + γV(s_{t+1}) − V(s_t)` is the one-step TD residual — low
variance, high bias, leaning hard on a value head that under this budget is itself half-trained; at
`λ = 1`, `Â_t = Σ_l γ^l δ_{t+l}` telescopes to the Monte-Carlo advantage `Σ γ^l R − V(s_t)` —
unbiased but high variance. The chosen `0.95` sits near the Monte-Carlo end, trading a little bias
for the unbiasedness that matters when the value head is undertrained. Both limits land where the
theory says they should, so I trust the surrogate is doing what I think and the weakness is not a
mis-set knob — it is the on-policy contract itself.

Let me trace one full iteration concretely, because it makes the waste tangible and rules out any
hope that a scheduling trick recovers it. The 16 envs step for 2048 steps under `π_old`, filling a
buffer of 32,768 transitions; I compute GAE advantages and value targets against the current `V`;
I do 10 epochs of 128 minibatches, each minibatch nudging `π_θ` uphill on the clipped surrogate and
`V` toward its return targets. By the later epochs a growing fraction of samples have ratios outside
`1±ε` and contribute zero policy gradient, so the marginal value of epoch 8, 9, 10 is already
decaying — this is why pushing `n_epochs` higher does not help and can hurt. Then the entire batch
is deleted. The next iteration must re-collect 32,768 *new* frames just to take its next 1,280 steps.
Nothing carries over except the network weights; the data — every state the humanoid actually
visited, every reward it actually earned — is gone. Multiply by the number of iterations that fit in
the 12.8M-frame budget and the picture is a policy that has seen an enormous amount of interaction
but retained almost none of it. An off-policy critic, by contrast, would still have all of it in the
buffer, available to be regressed against again and again. That is the difference between spending
the budget and *keeping* it, and it is the single fact this whole ladder is organized around.

So what should PPO actually achieve on stand, walk, and run within this budget? The three tasks
differ along the axis PPO is most sensitive to: how much directed, sustained exploration the
reward demands before the policy finds the behavior. Standing balance is the gentlest — the reward
is dense (stay upright, stay near the target posture), the initial random policy already collects
informative gradients because almost any small correction changes the height, and the behavior to
discover is close to "do not fall," which a stochastic policy stumbles into early. So I expect PPO
to make the most progress on `h1hand-stand-v0`, though "most progress" under a starved on-policy
budget is still likely well below what an off-policy method reaches, because PPO is re-deriving the
value function from scratch out of non-reused data. Walking is harder: the reward rewards forward
velocity, but to get forward velocity the humanoid must first not fall *and* coordinate a gait,
and the gait is a longer-horizon, more fragile behavior that the policy must hold together for
many steps before the reward confirms it. PPO's undirected Gaussian exploration plus entropy bonus
has to chance upon a proto-gait and then reinforce it, and with no experience reuse it gets fewer
attempts per frame to lock it in. Running is hardest still — it is walking pushed to a regime where
small coordination errors are catastrophic, so it most rewards the sustained, sample-efficient
refinement that on-policy learning is worst at. I therefore expect a descending pattern across the
three, `h1hand-stand-v0` best and `h1hand-run-v0` worst, with all three pressed down by the budget,
and meaningful seed-to-seed variance because whether a fragile gait is found in time is partly luck
under undirected exploration.

There is a subtler point I want to flag now because it sets up the diagnosis for the next rung.
PPO's weakness here is not instability — the clip makes PPO reliable, it will not blow up. Its
weakness is *throughput of learning per frame*. So when I read PPO's numbers, the question I will
ask is not "did it diverge" but "how far short of the off-policy methods did it land, and on which
tasks did the gap open widest." If the gap is largest on walk and run and smallest on stand, that
confirms the story that the binding constraint is sample efficiency on the harder-to-explore
behaviors, and it points the next step squarely at the replay buffer: the cheapest way to beat PPO
is to stop throwing transitions away. That is precisely what the off-policy fills on the editable
surface do, and it is why the ladder climbs from PPO into the SAC/TD3 family rather than into a
better on-policy method.

It is also worth being explicit about what PPO is *not* failing at, because that shapes how I read
its numbers and what I do next. PPO is not a hard-exploration failure in the sparse-reward sense:
the HumanoidBench locomotion rewards are dense and shaped, so the policy gradient always has a
nonzero signal to ascend, and PPO will not sit at zero return the way a bonus-less agent does on a
sparse Atari game. Nor is it an instability failure: the clipped surrogate is a trust region
expressed as a flat spot in the loss, so the policy does not collapse and the runs do not diverge.
What PPO fails at is *rate* — how much usable policy improvement it extracts per frame of
interaction — and that is a quieter, more insidious failure, because the agent looks like it is
learning (the return curve rises) and simply tops out lower than it should within the budget. So
the right way to read PPO here is not "did it work" but "where did its curve flatten relative to
the off-policy methods, and how much budget did it burn re-deriving a value function it could not
keep." That distinction is exactly why the next rung is an off-policy method and not a tuned PPO:
no amount of clip-tuning or GAE-tuning changes the fact that the rollouts are discarded — as the
reuse-factor arithmetic already showed, the ceiling is a hard 10.

One more thing the floor has to establish: a calibration of the return scale. The humanoid tasks
produce returns that can run into the hundreds over an episode, and the editable critic's
distributional support is fixed at `[−250, 250]` over 101 atoms precisely to cover that range.
PPO, with its own value head and no such support, gives me an independent, architecture-free read
of roughly what return each task admits within the budget — a sanity anchor for the off-policy
numbers to come. If an off-policy method later reports a return far outside what PPO's own value
head implies is reachable, that is a flag to check the critic's support and the bootstrap mask,
not a free win. The `[−250, 250]` window itself is a soft check on my whole plan: an episode of a
thousand steps at a per-step shaped reward of order tenths sums to a few hundred, which lands inside
that window, so the support is scaled for these tasks and PPO's value head and the off-policy
critic are measuring the same object on the same axis.

So the expectations I will hold PPO to, falsifiably, are these. It should learn *something* on
all three tasks — it is reliable, not broken — with the clearest traction on stand, where dense
reward and short-horizon behavior suit undirected exploration, and the weakest on run, where
long-horizon coordination needs the sample efficiency PPO structurally lacks. Across the board it
should land below what an off-policy actor-critic reaches on the same frame budget, and the size
of that shortfall — largest on the harder-to-explore tasks — is the measurement that motivates the
entire rest of the ladder. The fix is not a better policy-gradient trick; it is to start reusing
experience, which means moving onto the editable off-policy surface and lifting the reuse factor off
its hard ceiling of 10. The distilled floor — PPO run as its own on-policy script, bypassing the
editable contract — is in the answer.
