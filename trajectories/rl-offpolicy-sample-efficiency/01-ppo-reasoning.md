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

Now put that against the budget. The task gives 100,000 gradient steps with 128 parallel
environments — about 12.8M environment frames of interaction. PPO cannot reuse a frame across
iterations, so its entire learning signal is whatever it can wring out of each fresh batch before
discarding it. To stay stable it wants large, decorrelated batches and a modest number of epochs;
to fit the humanoid's high-dimensional, contact-rich dynamics it wants *many* such batches. Under
a tight frame budget those two demands fight: I can have stable updates or many updates, not both.
An off-policy method, by contrast, keeps a replay buffer and revisits each transition many times,
so it converts the same frames into far more gradient signal — which is the entire reason the
editable surface on this task is an *off-policy* actor-critic and not a policy-gradient loop. PPO
is the baseline that demonstrates, by being the one method here that *cannot* touch that buffer,
what the buffer is worth.

That structural fact also shows up in how PPO is run here, and it is worth being explicit because
it tells me what PPO is *not*. PPO does not fill the editable `Actor`/`Critic`/`update_*`
contract at all — those functions assume a replay buffer, a distributional critic, target
networks, and a per-env exploration scheme, none of which an on-policy method uses. So PPO runs as
its own self-contained on-policy script: a stable-baselines3 `MlpPolicy` with separate
`[256, 256]` tanh policy and value heads, `n_steps=2048` per env across 16 parallel environments,
10 epochs per batch, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`, learning rate `3e-4`,
trained for 800,000 PPO steps to match the same 12.8M-frame budget. It bypasses the off-policy
infrastructure entirely. That is the right way to give PPO its fair shot, and it is also exactly
why PPO is the floor: it opts out of the one mechanism — experience reuse — that the rest of the
ladder is built to exploit.

So what should PPO actually achieve on stand, walk, and run within this budget? The three tasks
differ along the axis PPO is most sensitive to: how much directed, sustained exploration the
reward demands before the policy finds the behavior. Standing balance is the gentlest — the reward
is dense (stay upright, stay near the target posture), the initial random policy already collects
informative gradients because almost any small correction changes the height, and the behavior to
discover is close to "do not fall," which a stochastic policy stumbles into early. So I expect PPO
to make the most progress on stand, though "most progress" under a starved on-policy budget is
still likely well below what an off-policy method reaches, because PPO is re-deriving the value
function from scratch out of non-reused data. Walking is harder: the reward rewards forward
velocity, but to get forward velocity the humanoid must first not fall *and* coordinate a gait,
and the gait is a longer-horizon, more fragile behavior that the policy must hold together for
many steps before the reward confirms it. PPO's undirected Gaussian exploration plus entropy bonus
has to chance upon a proto-gait and then reinforce it, and with no experience reuse it gets fewer
attempts per frame to lock it in. Running is hardest still — it is walking pushed to a regime where
small coordination errors are catastrophic, so it most rewards the sustained, sample-efficient
refinement that on-policy learning is worst at. I therefore expect a descending pattern across the
three, stand best and run worst, with all three pressed down by the budget, and meaningful
seed-to-seed variance because whether a fragile gait is found in time is partly luck under
undirected exploration.

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
no amount of clip-tuning or GAE-tuning changes the fact that the rollouts are discarded.

One more thing the floor has to establish: a calibration of the return scale. The humanoid tasks
produce returns that can run into the hundreds over an episode, and the editable critic's
distributional support is fixed at `[−250, 250]` over 101 atoms precisely to cover that range.
PPO, with its own value head and no such support, gives me an independent, architecture-free read
of roughly what return each task admits within the budget — a sanity anchor for the off-policy
numbers to come. If an off-policy method later reports a return far outside what PPO's own value
head implies is reachable, that is a flag to check the critic's support and the bootstrap mask,
not a free win.

So the expectations I will hold PPO to, falsifiably, are these. It should learn *something* on
all three tasks — it is reliable, not broken — with the clearest traction on stand, where dense
reward and short-horizon behavior suit undirected exploration, and the weakest on run, where
long-horizon coordination needs the sample efficiency PPO structurally lacks. Across the board it
should land below what an off-policy actor-critic reaches on the same frame budget, and the size
of that shortfall — largest on the harder-to-explore tasks — is the measurement that motivates the
entire rest of the ladder. The fix is not a better policy-gradient trick; it is to start reusing
experience, which means moving onto the editable off-policy surface. The distilled floor — PPO run
as its own on-policy script, bypassing the editable contract — is in the answer.
