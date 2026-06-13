# Context: on-policy policy optimization for continuous control (circa 2016-2017)

## Research question

We want to train a neural-network control policy for a continuous-action task — a simulated
robot that must learn to run, hop, or walk — directly from its own interaction with the
environment. The policy `π_θ(a | s)` is stochastic (a diagonal-Gaussian over the
real-valued action vector, mean produced by an MLP), and the only signal is the scalar
reward stream. We measure progress by the discounted expected return
`η(π_θ) = E[Σ_t γ^t r_t]`, and we improve `θ` by gradient ascent on a Monte-Carlo estimate
of `∇_θ η`.

The painful fact about that gradient is data efficiency. The unbiased policy-gradient
estimate is valid only for data drawn from the *current* policy, so the textbook recipe is:
roll out the current policy, form one gradient estimate, take one small step, throw the data
away, and roll out again. Each batch of expensive environment interaction buys exactly one
gradient step. If we instead try to squeeze several optimization steps out of one batch — the
obvious way to be more sample-efficient — the policy moves away from the one that generated
the data, the estimate stops being valid, and in practice the updates become destructively
large: a single batch can push the policy off a cliff it never recovers from.

So the precise goal is a single first-order algorithm that simultaneously: (1) reuses each
batch of on-policy data for *several* epochs of minibatch SGD rather than one update, to be
sample-efficient; (2) never lets a single update move the policy so far that the surrogate it
optimized stops predicting the true return — i.e. keeps each step inside a region where the
local approximation is trustworthy; (3) needs no second-order machinery (no Fisher matrix, no
conjugate gradient, no line search) so it stays simple and cheap and is compatible with
architectures that share parameters between policy and value function or that inject noise;
and (4) remains reliable across environments with very different dynamics rather than needing
a freshly engineered optimizer for each task. The methods below each get a subset of these;
none gets all four at once.

## Background

By this time the dominant way to train continuous-control policies with deep nets is
policy-gradient ascent. The core identity (Williams 1992; Sutton et al. 2000) is that for a
parameterized stochastic policy,

```
∇_θ η = E[ Σ_t Ψ_t · ∇_θ log π_θ(a_t | s_t) ],
```

where `Ψ_t` can be the trajectory return, the return-to-go, or — with almost the lowest
variance — the **advantage** `A_π(s_t, a_t) = Q_π(s_t,a_t) − V_π(s_t)`, which measures
whether the action taken was better or worse than the policy's default behavior at that
state. Intuitively the gradient should raise the probability of better-than-average actions
and lower the probability of worse-than-average ones, which is exactly what plugging in `A`
does. The advantage is not known and must be estimated from a learned value function `V(s)`.

Two background frames are load-bearing here.

**Advantage estimation as a bias-variance dial.** Define the TD residual
`δ_t = reward_t + γ V(s_{t+1}) − V(s_t)`. If `V = V_π`, then `E[δ_t] = A_π(s_t,a_t)`, so a single
`δ_t` is a low-variance but biased advantage estimate (biased whenever `V` is imperfect).
Summing `k` residuals, `Â_t^(k) = Σ_{l=0}^{k-1} γ^l δ_{t+l}`, telescopes to
`−V(s_t) + Σ_{l=0}^{k-1} γ^l reward_{t+l} + γ^k V(s_{t+k})`: larger `k` trusts the empirical rewards more
(less bias from `V`, more variance), `k→∞` is the Monte-Carlo return (unbiased, highest
variance). The generalized advantage estimator (Schulman et al. 2015b) takes the
exponentially-weighted average of all the `Â_t^(k)`, which collapses to the strikingly simple
`Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`, with `λ ∈ [0,1]` continuously interpolating between the
`λ=0` one-step residual and the `λ=1` Monte-Carlo return. `λ` is the variance-reduction knob.

**The surrogate objective and why the step must be bounded.** The expected return of a
candidate policy `π̃` relative to the current `π` decomposes exactly as
`η(π̃) = η(π) + Σ_s ρ_{π̃}(s) Σ_a π̃(a|s) A_π(s,a)`, where `ρ_{π̃}` is the (unnormalized)
discounted state-visitation distribution under `π̃`. The dependence of `ρ_{π̃}` on `π̃` makes
this intractable, so one uses the **local approximation** (Kakade & Langford 2002)

```
L_π(π̃) = η(π) + Σ_s ρ_π(s) Σ_a π̃(a|s) A_π(s,a),
```

which replaces `ρ_{π̃}` by the *old* policy's visitation `ρ_π`. Using importance sampling
with the probability ratio `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)`, this becomes the
sample-estimable `L^CPI(θ) = Ê_t[ r_t(θ) Â_t ]` (the superscript marks conservative policy
iteration, where the ratio surrogate originates). `L_π` matches `η` to first order at
`θ = θ_old` (where `r_t = 1`), but only there — it ignores the shift in state visitation, so
the further `θ` moves from `θ_old`, the less `L^CPI` predicts the true return. Maximizing
`L^CPI` with no leash gives an excessively large, often catastrophic policy update. This is
the empirically observed failure mode that motivates the whole design: it is well documented
that doing several gradient steps on the raw `L^PG`/`L^CPI` with one batch produces
destructively large updates.

The state of the field: deep Q-learning works on discrete-action Atari but not reliably on
continuous control; vanilla policy gradient has poor data efficiency and robustness; and the
trust-region line (below) is data-efficient and stable but relatively complicated, second
order, and awkward with parameter-sharing or noisy architectures. The open space is a method
that keeps trust-region-grade stability and data efficiency using only first-order
optimization.

## Baselines

**Vanilla policy gradient / REINFORCE (Williams 1992; advantage actor-critic, Mnih et al.
2016).** Estimate `∇_θ η = Ê_t[∇_θ log π_θ(a_t|s_t) Â_t]` by differentiating the surrogate
`L^PG(θ) = Ê_t[log π_θ(a_t|s_t) Â_t]`, take one ascent step per batch, discard, repeat.
A2C/A3C run `N` actors in parallel, use a finite-horizon advantage
`Â_t = −V(s_t) + Σ_{l=0}^{T-t-1} γ^l reward_{t+l} + γ^{T-t} V(s_T)`, and add an entropy bonus for
exploration. **Gap:** one gradient step per batch is sample-inefficient; and the moment you
try to do several steps on `L^PG` with the same batch, there is nothing in the objective that
notices the policy has drifted, so the updates blow up. There is no mechanism keeping the
update inside the region where the surrogate is valid.

**Trust region policy optimization (Schulman et al. 2015a).** Make the "stay near `θ_old`"
requirement explicit and exact. Starting from the CPI lower bound for mixture policies,
`η(π_new) ≥ L_π(π_new) − (2εγ/(1−γ)²) α²`, extend it to *all* stochastic policies by
replacing the mixture weight `α` with a divergence between `π` and `π̃`. Using total
variation and the inequality `D_TV(p,q)² ≤ D_KL(p,q)`, one gets the central minorization
bound

```
η(π̃) ≥ L_π(π̃) − C · max_s KL[π(·|s), π̃(·|s)],     C = 4εγ/(1−γ)²,   ε = max_{s,a}|A_π|.
```

Because `L` and the bound agree with `η` to first order at `π`, maximizing the right-hand
side is guaranteed to not decrease `η` — a monotonic-improvement (minorize-maximize)
guarantee. To take usefully large steps, TRPO replaces the penalty by a hard constraint on
the *average* KL and solves, each iteration,

```
maximize_θ  Ê_t[ r_t(θ) Â_t ]   subject to   Ê_t[ KL[π_old(·|s_t), π_θ(·|s_t)] ] ≤ δ,
```

via a linear approximation to the objective and a quadratic approximation to the constraint,
which reduces to a natural-gradient step computed with conjugate gradient on
Fisher-vector products plus a line search. **Gap:** it is second order and relatively
complicated (conjugate gradient, Fisher-vector products, backtracking line search), and the
constrained/Hessian machinery does not mix cleanly with architectures that share parameters
between the policy and value function or that include noise such as dropout. It also performs
one constrained solve per batch rather than cheap repeated minibatch SGD.

**Fixed-penalty surrogate (the form TRPO's own theory points at).** The minorization bound is
literally a *penalty*: `maximize_θ L_π(θ) − C · max_s KL`. One could simply pick a scalar
coefficient and run SGD on the ratio surrogate with a fixed old-new KL penalty. **Gap:** with the
theoretically-derived `C = 4εγ/(1−γ)²` the steps are far too small to be practical; and a
single hand-chosen `β` that works does not transfer — the right penalty differs across
problems and even *within* one problem as the advantage scale and the policy's sensitivity
change over the course of learning. A penalty coefficient that is right early is wrong late.
This is precisely why TRPO abandoned the fixed penalty for a hard constraint; the fixed
penalty leaves the practical step size at the mercy of an arbitrary scalar.

## Evaluation settings

The natural yardsticks for continuous control at this time:

- **MuJoCo continuous-control tasks via OpenAI Gym** (Todorov et al. 2012; Brockman et al.
  2016; benchmark protocol from Duan et al. 2016): simulated robots such as HalfCheetah,
  Hopper, Walker2d, Swimmer, Reacher, InvertedPendulum, InvertedDoublePendulum. State is the
  robot's joint configuration and velocities; action is a real-valued torque vector; reward
  rewards forward progress with control-effort penalties.
- **Policy/value architecture**: a fully-connected MLP with two hidden layers of 64 units and
  tanh nonlinearities, outputting the mean of a diagonal Gaussian over
  actions, with a separate learned log-standard-deviation; a separate value-function MLP.
- **Protocol**: a fixed interaction budget (e.g. one million environment timesteps per task);
  several random seeds per environment; the metric is mean episodic return averaged over
  evaluation episodes, higher is better. A method is judged on remaining reliable *across*
  environments with different dynamics, not on being tuned to one. Optimization with Adam,
  GAE for advantages (`γ = 0.99`, `λ = 0.95`), advantage standardization per batch.

## Code framework

A standard on-policy actor-critic harness already has the data pipeline (collect a fixed-length
segment of `N` parallel actors × `T` steps), the Gaussian-MLP policy/value networks, the GAE
advantage computation, the Adam optimizer, and the K-epoch minibatch loop. The single empty
slot is the per-minibatch loss that turns one batch of freshly collected on-policy data into a
policy update: the rule that decides how to extract many SGD steps from one batch without the
policy drifting destructively.

```python
import torch
from torch.distributions import Normal


class Agent(torch.nn.Module):
    """Gaussian-MLP actor-critic. Architecture is fixed; only the update rule is open."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = mlp(obs_dim, 1)                       # value function V(s)
        self.actor_mean = mlp(obs_dim, act_dim)             # mean of the action Gaussian
        self.actor_logstd = torch.nn.Parameter(torch.zeros(1, act_dim))  # state-indep log std

    def get_action_and_value(self, obs, action=None):
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return action, logprob, entropy, self.critic(obs)


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """Generalized advantage estimation: Â_t = Σ_l (γλ)^l δ_{t+l}, δ_t = reward_t+γV'−V."""
    adv = torch.zeros_like(rewards)
    lastgae = 0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        v_next = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next * nonterminal - values[t]
        adv[t] = lastgae = delta + gamma * lam * nonterminal * lastgae
    returns = adv + values
    return adv, returns


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages,
                   mb_returns, mb_values, args):
    """The per-minibatch loss is exactly what we will design: take one batch of
    on-policy data (old log-probs, advantages, returns) and produce a scalar loss
    whose gradient updates the policy and value function — while keeping each of the
    several SGD epochs on this batch from moving the policy too far to trust."""
    # TODO: the policy-update objective we will design.
    pass


def train(agent, envs, args):
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    for iteration in range(args.num_iterations):
        # 1) roll out the current policy for N actors x T steps, store
        #    obs, actions, logprobs (under π_old), rewards, dones, values
        rollout = collect_rollout(agent, envs, args)
        # 2) advantages + returns via GAE
        advantages, returns = compute_gae(
            rollout.rewards, rollout.values, rollout.dones,
            rollout.next_value, args.gamma, args.gae_lambda)
        # 3) K epochs of minibatch SGD on the freshly collected batch
        for epoch in range(args.update_epochs):
            for mb in minibatches(rollout, advantages, returns, args.minibatch_size):
                loss, *_ = compute_losses(
                    agent, mb.obs, mb.actions, mb.logprobs,
                    mb.advantages, mb.returns, mb.values, args)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
```

The rollout collector, GAE, optimizer, and K-epoch loop are fixed; `compute_losses` is the
one slot left open.
