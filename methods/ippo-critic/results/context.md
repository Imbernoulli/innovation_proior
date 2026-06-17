## Research question

A team of `N` agents acts in a partially observable, stochastic environment and shares a single
scalar reward. Formally this is a decentralised partially observable Markov decision process
(Dec-POMDP) `⟨ N, S, U, P, r, Z, O, ρ, γ ⟩`: the world is in state `s`, every agent `a`
simultaneously picks `u^a` from the joint action `u = (u^1,…,u^N)`, the world transitions
`s' ∼ P(·|s,u)` and emits one team reward `r(s,u)`, and each agent sees only a local observation
`z^a ∼ O(s,a)` (so its action-observation history `τ^a` is all it can condition on). The agents must
learn a *joint* policy `π(u|τ) = ∏_a π^a(u^a|τ^a)` that maximises the expected discounted team
return `J = E[Σ_t γ^t r_t]`. The execution constraint is hard: at run time each agent's action may
depend only on its own `τ^a` — the policies must be **decentralisable**.

Training, however, need not be decentralised. It usually happens in simulation or a lab, where the
learner may legally use information no agent could access at execution: the global state `s`, every
agent's observation, shared gradients and shared parameters. This is **centralised training with
decentralised execution (CTDE)**. The standard CTDE actor-critic recipe is settled in its skeleton:
one decentralised actor `π^a(u^a|τ^a)` per agent (often parameter-shared), trained by a policy
gradient with a learned value function as a variance-reducing baseline. What is *not* settled — and
is the entire question here — is the **value function (critic)**: what it should condition on and how
it should be built. The actor is fixed in form; the critic's input set is the free design variable,
and that single choice fixes the bias-variance behaviour of every advantage estimate and decides
whether the method survives the pathologies of multi-agent learning. A viable solution has to (1) give a
low-variance, low-bias advantage to each decentralised actor, (2) remain learnable as the number of
agents and the observation dimension grow, and (3) not be destabilised by the fact that, from any one
agent's viewpoint, the *other* learning agents make the environment non-stationary.

## Background

The engine underneath is single-agent policy gradient. The score-function estimator
`ĝ = E[ ∇_θ log π_θ(a|s) Â ]` (Williams, 1992) is unbiased but high-variance; subtracting a learned
state-value baseline `V(s)` and using the advantage `Â` in its place leaves the gradient unbiased
while cutting variance — the actor-critic pattern (Mnih et al., 2016, A3C/A2C). The variance of the
gradient is dominated by the variance of `Â`, so the quality of the critic *is* the quality of the
learning signal.

Trust-region policy optimisation (TRPO; Schulman et al., 2015) sharpened the policy step. It maximises
the importance-weighted surrogate `E[ (π_θ/π_θold) A ]` subject to a KL trust region
`E[ KL(π_θold, π_θ) ] ≤ δ`, which guarantees approximately monotone improvement. The price is that the
KL constraint is enforced through conjugate gradient on Hessian-vector products — computationally
heavy, and awkward to combine with parameter sharing or stochastic regularisation. Proximal policy
optimisation (PPO; Schulman et al., 2017) keeps TRPO's empirical behaviour with only first-order
updates by replacing the constraint with a clipped surrogate. With the probability ratio
`r_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t)`,

```
L^CLIP(θ) = E_t[ min( r_t(θ) Â_t,  clip(r_t(θ), 1-ε, 1+ε) Â_t ) ],
```

the `min` makes the objective a pessimistic lower bound: it only ignores a ratio move when that move
would *improve* the surrogate, and keeps it when the move would worsen it, so there is no incentive to
push `r_t` outside `[1-ε, 1+ε]`. Equivalently, for `Â_t > 0` the objective is flat once
`r_t > 1+ε`, and for `Â_t < 0` it is flat once `r_t < 1-ε`; the unclipped term is still used on the
side that would make the objective worse. The full PPO objective adds a value-function error and an
entropy bonus, `L^CLIP - c_1 L^VF + c_2 S[π]`, and several epochs of minibatch updates are run on each
batch of trajectories. The advantage is computed by generalized advantage estimation (GAE; Schulman
et al., 2016): for a finite rollout segment,
`Â_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}` with TD residual
`δ_t = r_t + γV(x_{t+1}) - V(x_t)`, where `x_t` is the value input. Small `λ` relies more on short
bootstraps and has lower variance but more bias from the value estimate; large `λ` relies more on
long returns and has lower bootstrap bias but higher variance. At `λ=1` the truncated GAE sum
telescopes to `Σ_{l=0}^{T-t-1} γ^l r_{t+l} + γ^{T-t} V(x_T) - V(x_t)`. Some PPO implementations also
clip the value update, `L^VF = max{ (V - V̂)², (V_old + clip(V - V_old, -ε, +ε) - V̂)² }`, to keep the
critic inside its own trust region; others use an unclipped squared TD error. A practical `n`-step target has the same bootstrapped-return shape,
`R_t^{(n)} = Σ_{l=0}^{n-1} γ^l r_{t+l} + γ^n V(x_{t+n})`, with advantage
`Â_t^{(n)} = R_t^{(n)} - V(x_t)`.

Carrying this into the multi-agent case raises the difficulties that have shaped the field:

- **Non-stationarity from co-learners.** If one agent treats the other `N-1` agents as part of the
  environment, then because those agents are themselves learning and exploring, the effective
  transition and reward dynamics drift over training. The single-agent convergence guarantees no
  longer hold (Tan, 1993), and value estimates go stale as peers change.
- **Confounded stochasticity.** An independent learner cannot always tell environment stochasticity
  apart from another agent's exploration, which provably blocks optimal play in some coordination
  matrix games (Claus & Boutilier, 1998).
- **Joint-action explosion.** A genuinely centralised joint policy or joint `Q` over `U^N` grows
  exponentially in `N`, and joint policies are not inherently decentralisable.
- **Partial observability vs. the critic's information set.** The actor must act on history `τ^a`,
  because in a Dec-POMDP the world is not fully observable. A diagnostic well documented for this
  setting (the Dec-Tiger problem; Nair et al., 2003): two underlying states but exponentially many
  histories, where the value of a history must rise as repeated observations accumulate evidence. A
  value function keyed on the global state `s` alone cannot represent this — it averages over all the
  histories that map to `s`, so its value is `E_{h|s}[·]` and it loses exactly the history information
  the actor needs; information-gathering actions look valueless to it. So conditioning the critic on
  the ground-truth state is not automatically an improvement: it can be *biased* relative to the
  history-conditioned objective the actor is optimising.

Together these constraints make the critic's input set a real design problem rather than a matter of
feeding it every variable available during training.

## Baselines

**Independent actor-critic / independent Q-learning (IAC, IQL; Tan, 1993; Foerster et al., 2018
ablation).** Decompose the `N`-agent problem into `N` single-agent problems: each agent runs vanilla
actor-critic or Q-learning on its own `(τ^a, u^a, r)`, treating peers as environment. Trivially
decentralisable and cheap. *Limitation:* on cooperative deep benchmarks these independent learners are
observed to be unstable and to converge to poor policies — the non-stationarity from co-learners and
the confounded-stochasticity pathology bite in practice, and a single bad update can be catastrophic
because nothing restrains the per-step policy change.

**Centralised joint critic (Central-V, COMA; Foerster et al., 2018) and MADDPG (Lowe et al., 2017).**
Keep decentralised actors but train a critic that conditions on centralised information — the global
state `s`, or the concatenation of all agents' observations/actions. Because the critic is discarded
at execution, the policies stay decentralisable. COMA additionally uses a counterfactual baseline
(comparing an agent's action to a marginal over its other actions, holding peers fixed) to sharpen
credit assignment. The premise is that seeing the whole picture removes the non-stationarity from the
critic's view and makes value learning easier. *Limitations:* (i) when the critic conditions on
information outside one agent's actor input, the actor update has to average over that extra
information; in practice this marginalisation is often estimated by sampling and raises variance in
the per-agent update; (ii) it scales poorly as the number of agents and the observation/action
dimensions grow, both in input width and in learnability; (iii) a critic keyed on the bare state `s`
is biased in partial observability because it discards history information, so the central information helps only when
the critic retains history information rather than discarding it for `s`.

**Value-function factorisation (VDN, QMIX; Sunehag et al., 2018; Rashid et al., 2018).** Sidestep the
joint-action explosion by writing the joint `Q_tot` as a function of per-agent utilities —
`Q_tot = Σ_a Q_a` (VDN) or a state-conditioned monotonic mixing network with `∂Q_tot/∂Q_a ≥ 0`
(QMIX) — so that greedy decentralised action selection is consistent with the centralised value.
*Limitation:* the monotonicity that makes factorisation decentralisable also restricts what team-
reward functions it can represent; it is prone to *relative overgeneralisation*, where the policies
converge to a suboptimal joint action because a non-monotonic reward surface (coordinated success is a
needle in a haystack, partial coordination is penalised) cannot be represented by a monotonic mix.

Across all three families the unexamined common assumption is that *more centralised information in
the value function is better*. The cost each pays — instability without any restraint on the policy
step (IAC/IQL), variance and poor scaling and state-induced bias from centralising the critic
(Central-V/COMA/MADDPG), representational limits from monotone factorisation (VDN/QMIX) — is the gap
left open.

## Evaluation settings

The natural yardstick for cooperative, partially observable, common-reward control is the StarCraft
Multi-Agent Challenge (SMAC; Samvelyan et al., 2019), a suite of StarCraft II unit-micromanagement
maps spanning easy to very hard, where a team of allied units (melee, ranged, and healer types) must
defeat an enemy team under the hardest built-in AI; winning needs precisely coordinated movement such
as focus-fire and kiting. Each agent observes a local field of view; a global state is available
during training only. The protocol: train roughly 5-20M environment steps per map; the primary metric
is the **test win rate** (fraction of test episodes won under the greedy policy), with the test
episode return as a secondary metric, evaluated separately per map. A pure-Python reimplementation of
SMAC that removes the StarCraft II binary dependency makes the same maps runnable as a lightweight
benchmark. The GAE-based PPO setup uses `γ = 0.99`, GAE `λ = 0.95`, clip `ε = 0.2`, gradient-
norm clipping, advantage normalisation, parallel actors, and Adam; the common team reward is
broadcast to all agents. The bundled reference harness uses the same clipped policy objective but
computes fixed `q_nstep = 5` bootstrapped targets, standardises rewards by default, leaves return
standardisation off in the local config, and fits the critic with an unclipped masked squared TD error.

## Code framework

The substrate is the existing CTDE actor-critic harness: a parameter-shared decentralised actor (the
recurrent agent network), a learner that runs the PPO clipped surrogate on the actor and a regression
loss on the critic, the episode buffer, Adam, the GAE or harness `n`-step target machinery, and
reward standardisation. All of that is fixed. The one open slot is the critic module — what it conditions on
and how it maps that input to a per-agent value. The learner calls `critic(batch)` over a whole
sequence and then does `.squeeze(3)`, so the critic must return shape `(B, T, n_agents, 1)`; it must
set `self.output_type = "v"`. The body is left empty.

```python
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """Value function for a CTDE actor-critic learner. The actor is fixed and
    decentralised; this module is the free design slot. Returns one scalar value
    per agent per timestep, shape (B, T, n_agents, 1)."""

    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_type = "v"

        # Available pieces of the batch the critic MAY condition on:
        #   scheme["state"]["vshape"] : global state dim   (training-only info)
        #   scheme["obs"]["vshape"]   : per-agent obs dim   (also available at execution)
        # TODO: choose the critic's input set and architecture, and build the layers.

    def forward(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        # batch["state"] : (B, T, state_dim)
        # batch["obs"]   : (B, T, n_agents, obs_dim)
        # TODO: assemble the chosen inputs and map them to a per-agent value.
        values = None                                                     # (B, T, n_agents, 1)
        return values


# existing learner loop the critic plugs into (fixed)
def train_step(batch, actor, critic, target_critic, actor_opt, critic_opt, args):
    # advantage from the critic (GAE / n-step), then PPO clipped actor update
    advantages = compute_advantages(batch, target_critic, args)          # uses critic values
    ratios = th.exp(actor.logp(batch) - actor.old_logp(batch).detach())
    surr1 = ratios * advantages
    surr2 = th.clamp(ratios, 1 - args.eps_clip, 1 + args.eps_clip) * advantages
    pg_loss = -th.min(surr1, surr2).mean()                               # PPO clipped surrogate
    actor_opt.zero_grad(); pg_loss.backward(); actor_opt.step()

    v = critic(batch)[:, :-1].squeeze(3)                                 # critic regression
    critic_loss = ((target_returns(batch, target_critic, args) - v) ** 2).mean()
    critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
```

The actor, learner, optimiser, advantage estimator, and environment interface are all already in
place; the critic class is the single piece to be filled in.
