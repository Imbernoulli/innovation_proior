The rung I start from is the one a robotics practitioner reaches for by default, because it is the
honest floor for "train a humanoid policy under a fixed interaction budget": Proximal Policy
Optimization. It is the floor not because it is bad in absolute terms — on a massively parallel
simulator PPO is a workhorse — but because the budget here decides everything, and PPO spends the
budget in the most expensive currency there is. So before I design anything off-policy I want to be
precise about *why* on-policy is the weak baseline on this task, and exactly how it sits relative to
the editable algorithm surface.

PPO maximizes a clipped surrogate `L^CLIP = Ê_t[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)]`, where
`r_t = π_θ/π_old` is the probability ratio and `Â_t` is a generalized-advantage estimate. The clip
is a trust region expressed as a flat spot in the loss: once a sample's ratio moves past `1±ε` in
the direction the advantage favors, the gradient through that sample dies, so I can run several
epochs of minibatch SGD over one batch of rollouts without the policy collapsing. The catch is the
word *on-policy*: the `Â_t` are computed under the policy that generated the rollouts, and the
surrogate is only an honest proxy for improvement while `π_θ` stays near `π_old`. After a handful of
epochs I have extracted what I safely can, and I must throw the batch away and collect fresh
rollouts. Every transition feeds at most a few gradient steps and then it is gone.

Put a number on "at most a few," because that is the whole argument. The answer runs PPO with
`n_steps=2048` across 16 envs, so each rollout batch is `2048 × 16 = 32,768` transitions; `n_epochs=10`
passes at `batch_size=256` is `128` minibatch updates per epoch, `1,280` gradient steps per rollout —
then the batch is discarded forever. So the *reuse factor* of any frame is a hard constant: 10, one
per epoch, capped by the on-policy contract. Doing more epochs pushes `π_θ` further from `π_old`, the
ratios drift past `1±ε`, the clip flattens the gradient on more and more samples, and the extra
epochs buy nothing. Hold that against the editable off-policy surface: the fixed loop keeps every
transition in a GPU replay buffer and runs `num_updates=2` gradient steps *per env step* for the life
of the run, so a transition is eligible to be resampled on every update between storage and eviction —
under a fast-filling buffer, many times ten. That single ratio — 10 fixed versus a large
buffer-bounded number — is the entire reason the editable surface is an off-policy actor-critic and
not a policy-gradient loop, and why PPO is the floor: it is the one method here that structurally
cannot touch the buffer.

There is a second tax inside the same budget. PPO's value head is trained only on the batch in front
of it and refit from scratch every iteration — `1,280` value SGD steps per rollout on `32,768` fresh
targets that are never seen again. The advantage `Â_t = Σ_l (γλ)^l δ_{t+l}` with
`δ_t = R_t + γV(s_{t+1}) − V(s_t)` leans on `V` through every term, so an undertrained value head
feeds a biased advantage straight into the policy gradient. An off-policy critic is trained
continuously on the whole buffer with a bootstrapped target, accumulating a value estimate across the
run instead of refitting each iteration. PPO pays twice for being on-policy — once in the policy's
reuse ceiling, once in a value head that never converges — and both bills come due on the
long-horizon tasks where an accurate long-horizon value is exactly what walking and running need.

The exploration side has its own geometric cost, and it explains why a descending stand-walk-run
pattern is not incidental. PPO explores by sampling a factorized Gaussian over the action, one
independent coordinate per actuator, so for a humanoid with tens of action dimensions the exploration
is an isotropic cloud in `[−1,1]^d`. A coordinated gait lives on a thin, correlated manifold — the
joints must move together in a specific phase relationship — and the chance that independent
per-coordinate jitter traces that pattern long enough to earn reward falls off fast as `d` grows and
the behavior gets longer-horizon. Standing needs almost no coordination, so the isotropic cloud finds
it; running needs tight sustained coordination, so the same cloud wastes its samples off-manifold.
The `ent_coef=0.01` bonus keeps the cloud from collapsing early, the right instinct, but it *widens*
the search without pointing it — the wrong economy under a capped budget. This is the second thing
the off-policy surface is built to fix: to let exploration be *directed by the value* rather than
sprayed uniformly across the action box.

Is the floor on-policy learning itself, or just this particular algorithm? REINFORCE has no baseline
and no clip, so its reuse factor is 1 — strictly worse. TRPO enforces the trust region exactly with a
KL constraint and Fisher-vector products, but it still recomputes advantages under the current policy
and discards the batch, spending *more* compute for the *same* frame efficiency PPO has. Tuning PPO —
bigger batches, a KL-adaptive clip, a better λ — moves the operating point along the
stability-versus-throughput curve but never lifts the reuse ceiling off 10. Every road inside the
on-policy family dead-ends at the same wall: the rollouts are thrown away. So the floor is not "PPO is
weak," it is "on-policy learning is the wrong currency for a fixed frame budget," and the only exit is
experience reuse — the replay buffer, which is off the on-policy map and squarely on the editable
surface.

That is also why PPO does not fill the editable `Actor`/`Critic`/`update_*` contract at all — those
functions assume a replay buffer, a distributional critic, target networks, and per-env noise, none
of which an on-policy method uses. Shoehorning PPO in would mean sampling old buffer transitions and
correcting their staleness with importance weights `π_θ/π_behavior`, and those weights explode as the
buffer ages and the behavior policy recedes — the very instability the clip was invented to avoid, now
unbounded because the data is arbitrarily off-policy. So PPO runs as its own self-contained on-policy
script: a stable-baselines3 `MlpPolicy` with separate `[256,256]` tanh policy and value heads,
`n_steps=2048` per env across 16 envs, 10 epochs, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`,
lr `3e-4`, trained for 800,000 PPO steps to match the 12.8M-frame budget. Giving it its own script is
the fair shot, and it is exactly why it is the floor: it opts out of experience reuse, the one
mechanism the rest of the ladder exploits.

So what should PPO achieve on stand, walk, and run? The tasks differ along the axis PPO is most
sensitive to — how much directed, sustained exploration the reward demands before the policy finds the
behavior. Standing is gentlest: the reward is dense, and almost any small correction changes the
height, so a stochastic policy stumbles into "do not fall" early. Walking is harder — forward velocity
requires first not falling *and* coordinating a longer-horizon gait the policy must hold together many
steps before the reward confirms it, and with no reuse PPO gets fewer attempts per frame to lock it
in. Running is hardest, walking pushed to a regime where small coordination errors are catastrophic.
So I expect a descending pattern, `h1hand-stand-v0` best and `h1hand-run-v0` worst, all three pressed
down by the budget, with meaningful seed-to-seed variance because whether a fragile gait is found in
time is partly luck under undirected exploration.

The point that sets up the next rung: PPO's weakness here is not instability — the clip makes it
reliable, it will not diverge — nor a hard-exploration failure, since the HumanoidBench rewards are
dense and shaped and the policy gradient always has a nonzero signal to ascend. Its weakness is
*throughput of learning per frame*: the agent looks like it is learning (the return curve rises) and
simply tops out lower than it should within the budget. So the right way to read PPO's numbers is not
"did it work" but "where did its curve flatten relative to the off-policy methods, and on which tasks
did the gap open widest." If the gap is largest on walk and run and smallest on stand, that confirms
the binding constraint is sample efficiency on the harder-to-explore behaviors, and it points the next
step at the replay buffer — no amount of clip- or GAE-tuning changes the fact that the rollouts are
discarded.

One calibration the floor establishes: PPO's own value head gives an architecture-free read of roughly
what return each task admits within the budget — a sanity anchor for the off-policy critic, whose
distributional support is fixed at `[−250,250]` over 101 atoms. An episode of a thousand steps at a
per-step shaped reward of order tenths sums to a few hundred, inside that window, so the support is
scaled for these tasks and PPO's value head and the off-policy critic are measuring the same object on
the same axis. If an off-policy method later reports a return far outside what PPO implies is
reachable, that flags the critic's support or the bootstrap mask, not a free win. The distilled floor —
PPO run as its own on-policy script, bypassing the editable contract — is in the answer.
